"""The only module that opens a WAV.

One pass over the 24 GB of audio produces, per segment, the per-microphone short-time
spectra of every 100 ms frame plus the aligned labels. Everything downstream (microphone
subsets, direction grids, sub-band counts, frame stacking) is then a re-steer over this
cache rather than another read of the audio.

Why spectra and not finished energy maps: a map cache costs ~3.4 GB per microphone subset
and freezes the subset, the grid and the bin count into storage, while steering a batch
costs about 2% of a training step. See design_notes section 10.
"""
import struct
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from .config import (BYTES_PER_SAMPLE, CACHE_BAND, CH_PER_DEVICE, CHUNK, N_CHANNELS,
                     SR, Segment, segments)
from .labels import frame_labels

# Above 200 Hz is where the drone's harmonics live, and the low end is dominated by wind.
# The FFT band selection does most of the filtering; this removes the sub-200 Hz energy
# first so it cannot leak up through the sidelobes of a rectangular analysis window.
FILTER_BAND = (200.0, 8000.0)
FILTER_ORDER = 5


def wav_info(path: Path) -> tuple[int, int, int, int]:
    """(data offset, channels, sample rate, bytes per sample) by walking the RIFF chunks.

    The header is 44 bytes on every file in this dataset, but a WAV is allowed extra
    chunks and a silent 4-byte misread would shift one recorder's audio relative to the
    others, so the chunks are parsed rather than assumed.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF" or f.read(8)[4:] != b"WAVE":
            raise ValueError(f"not a RIFF/WAVE file: {path}")
        ch = sr = bits = None
        while True:
            head = f.read(8)
            if len(head) < 8:
                raise ValueError(f"no data chunk in {path}")
            cid, size = struct.unpack("<4sI", head)
            if cid == b"fmt ":
                fmt = f.read(size)
                ch, sr = struct.unpack("<H", fmt[2:4])[0], struct.unpack("<I", fmt[4:8])[0]
                bits = struct.unpack("<H", fmt[14:16])[0]
            elif cid == b"data":
                return f.tell(), ch, sr, bits // 8
            else:
                f.seek(size + (size & 1), 1)


class SegmentAudio:
    """Memory-mapped 24-channel view of one segment, with per-recorder offsets applied.

    The four recorders were started by hand seconds apart, so each has its own sample
    correction. Sample 0 here is the first sample of the segment on every channel.
    """

    def __init__(self, seg: Segment):
        align = seg.align
        self.paths = [seg.session_dir / w for w in align["wavs_nosync"]]
        skip = int(round(seg.skip_seconds * SR))
        self.starts = [skip + c for c in align["corrections_samples"]]
        self.maps, avail = [], []
        for path, start in zip(self.paths, self.starts):
            offset, ch, sr, bps = wav_info(path)
            if (ch, sr, bps) != (CH_PER_DEVICE, SR, BYTES_PER_SAMPLE):
                raise ValueError(f"unexpected format in {path}: {ch}ch {sr}Hz {bps}B")
            n = (path.stat().st_size - offset) // (bps * ch)
            self.maps.append(np.memmap(path, dtype="<i4", mode="r", offset=offset,
                                       shape=(n, ch)))
            avail.append(n - start)
        self.n_samples = min(avail)

    @property
    def n_frames(self) -> int:
        return self.n_samples // CHUNK

    def read(self, first: int, count: int) -> np.ndarray:
        """(count, 24) float64, sample-aligned across recorders."""
        out = np.empty((count, N_CHANNELS), dtype=np.float64)
        for d, (mm, start) in enumerate(zip(self.maps, self.starts)):
            s = start + first
            out[:, d * CH_PER_DEVICE:(d + 1) * CH_PER_DEVICE] = mm[s:s + count]
        return out


def build_cache(seg: Segment, band=CACHE_BAND, block_frames: int = 600,
                overwrite: bool = False, verbose: bool = True) -> Path:
    """Write cache/<segment id>/{stft.npy, meta.npz}. Returns the segment's cache dir."""
    out_dir = seg.cache_dir
    stft_path, meta_path = out_dir / "stft.npy", out_dir / "meta.npz"
    if stft_path.exists() and meta_path.exists() and not overwrite:
        if verbose:
            print(f"{seg.id}: cache present, skipping")
        return out_dir

    audio = SegmentAudio(seg)
    labels = frame_labels(seg)
    n = min(len(labels["has_drone"]), audio.n_frames)
    if n < len(labels["has_drone"]) and verbose:
        print(f"{seg.id}: audio is short, {len(labels['has_drone'])} labelled frames "
              f"-> {n} usable")
    labels = {k: v[:n] for k, v in labels.items()}

    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    b, a = butter(FILTER_ORDER, [FILTER_BAND[0] / (SR / 2), FILTER_BAND[1] / (SR / 2)],
                  btype="band")

    out_dir.mkdir(parents=True, exist_ok=True)
    stft = np.lib.format.open_memmap(stft_path, mode="w+", dtype=np.complex64,
                                     shape=(n, N_CHANNELS, int(sel.sum())))
    pad = CHUNK                      # filter context, discarded after filtering
    t0 = time.time()
    for f0 in range(0, n, block_frames):
        nf = min(block_frames, n - f0)
        lo = max(0, f0 * CHUNK - pad)
        hi = min(audio.n_samples, (f0 + nf) * CHUNK + pad)
        blk = filtfilt(b, a, audio.read(lo, hi - lo), axis=0)
        head = f0 * CHUNK - lo
        blk = blk[head:head + nf * CHUNK].reshape(nf, CHUNK, N_CHANNELS)
        stft[f0:f0 + nf] = np.fft.rfft(blk, axis=1)[:, sel].transpose(0, 2, 1)
        if verbose:
            done = f0 + nf
            print(f"\r{seg.id}: {done}/{n} frames  "
                  f"{done / max(time.time() - t0, 1e-9):.0f} fps", end="", flush=True)
    stft.flush()
    if verbose:
        print(f"\r{seg.id}: {n} frames, {stft_path.stat().st_size / 2**30:.2f} GiB, "
              f"{time.time() - t0:.0f}s")

    np.savez(meta_path, freqs=freqs[sel], n_frames=n, split=seg.split,
             session=seg.session, flight_id=seg.flight_id, name=seg.name,
             has_drone_segment=seg.has_drone, band=np.array(band),
             skip_seconds=seg.skip_seconds, starts=np.array(audio.starts),
             **labels)
    return out_dir


def load_cache(seg: Segment, mmap: bool = True) -> tuple[np.ndarray, dict]:
    """(n, 24, F) complex64 spectra and the meta dict, index-aligned by construction."""
    stft = np.load(seg.cache_dir / "stft.npy", mmap_mode="r" if mmap else None)
    meta = dict(np.load(seg.cache_dir / "meta.npz", allow_pickle=False))
    if len(stft) != int(meta["n_frames"]):
        raise ValueError(f"{seg.id}: {len(stft)} frames cached, meta says {meta['n_frames']}")
    return stft, meta


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the STFT cache for every segment.")
    p.add_argument("--only", help="substring filter on segment id")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    total = 0.0
    for seg in segments():
        if args.only and args.only not in seg.id:
            continue
        d = build_cache(seg, overwrite=args.overwrite)
        total += (d / "stft.npy").stat().st_size / 2**30
    print(f"cache total {total:.2f} GiB")
