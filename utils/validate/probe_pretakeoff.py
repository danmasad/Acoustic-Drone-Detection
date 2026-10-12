"""Is the audio before takeoff usable as no-drone data?

Every session's `skip_seconds` exists because the recorders were started before the flight,
so each WAV begins with minutes of audio we have never touched. If that audio is genuinely
ambient, it is real negatives from the same array at the same site — which is exactly what
the detection head is starving for, and it is already on disk.

This compares three populations per session: the pre-takeoff window, the dataset's own
labelled ambient tail, and drone frames. Two statistics per 100 ms frame:

  band_db   total energy in the model band, 200-2200 Hz
  contrast  brightest beamformed direction over the map mean, in dB

A drone is a coherent point source, so it should show high contrast. Ambient wind is
diffuse and should show low contrast. If the pre-takeoff window looks like the labelled
ambient tail on both statistics, it is usable; if it looks like drone frames, something is
flying and it is not.

Run: .venv/bin/python utils/validate/probe_pretakeoff.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.beamform import Beamformer                                       # noqa: E402
from src.config import CHUNK, MIC_SUBSETS, MODEL_BAND, SR, segments       # noqa: E402
from src.stft_cache import SegmentAudio, bandpass, load_cache             # noqa: E402

MARGIN_S = 20.0        # discard this much before takeoff: motor spin-up, handling, chatter
WINDOW_S = 60.0        # how much pre-takeoff audio to characterize


def stats_from_spectra(X: np.ndarray, bf: Beamformer) -> tuple[np.ndarray, np.ndarray]:
    """(band energy dB, peak-to-mean contrast dB) per frame."""
    maps = bf.maps(torch.from_numpy(X).to(bf.device), log=False)[:, 0]
    flat = maps.flatten(1)
    peak = flat.max(1).values
    return (10 * torch.log10(flat.mean(1))).cpu().numpy(), \
           (10 * torch.log10(peak / flat.mean(1))).cpu().numpy()


def main() -> None:
    channels = MIC_SUBSETS["all24"]
    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    sel_cache = (freqs >= 200.0) & (freqs <= 4000.0)

    print(f"{'session':<22}{'population':<16}{'frames':>7}{'band dB':>10}{'contrast dB':>13}")
    for seg in segments():
        if not seg.has_drone:
            continue
        audio = SegmentAudio(seg)
        # The per-recorder corrections can be negative, so the true pre-roll is set by
        # whichever recorder started last, not by skip_seconds alone.
        available = min(audio.starts) / SR - MARGIN_S
        if available <= 5:
            print(f"{seg.session + ' f' + seg.flight_id:<22}"
                  f"{'pre-takeoff':<16}{'-':>7}   only {seg.skip_seconds:.0f}s before takeoff")
            continue
        span = min(WINDOW_S, available)
        first = -int(round((span + MARGIN_S) * SR))
        n = int(span * SR) // CHUNK
        blk = bandpass(audio.read(first, n * CHUNK)).reshape(n, CHUNK, -1)
        X = np.fft.rfft(blk, axis=1)[:, sel_cache].transpose(0, 2, 1).astype(np.complex64)

        stft, meta = load_cache(seg)
        bf = Beamformer(seg.align, channels, meta["freqs"], n_subbands=1, band=MODEL_BAND)
        pre_band, pre_con = stats_from_spectra(X[:, channels][:, :, bf.sel], bf)
        d_idx = np.arange(0, min(len(stft), 600))
        dr_band, dr_con = stats_from_spectra(
            np.asarray(stft[d_idx][:, channels][:, :, bf.sel]), bf)

        name = f"{seg.session} f{seg.flight_id}"
        print(f"{name:<22}{'pre-takeoff':<16}{n:>7}{np.median(pre_band):>10.1f}"
              f"{np.median(pre_con):>13.2f}")
        print(f"{'':<22}{'drone (first 60s)':<16}{len(d_idx):>7}{np.median(dr_band):>10.1f}"
              f"{np.median(dr_con):>13.2f}")

    for seg in segments():
        if seg.has_drone:
            continue
        stft, meta = load_cache(seg)
        bf = Beamformer(seg.align, channels, meta["freqs"], n_subbands=1, band=MODEL_BAND)
        idx = np.arange(0, min(len(stft), 600))
        band, con = stats_from_spectra(np.asarray(stft[idx][:, channels][:, :, bf.sel]), bf)
        print(f"{seg.session + ' f' + seg.flight_id:<22}{seg.name:<16}{len(idx):>7}"
              f"{np.median(band):>10.1f}{np.median(con):>13.2f}")

    print("\nAvailable pre-takeoff audio, after a "
          f"{MARGIN_S:.0f}s margin:")
    total = 0.0
    for seg in segments():
        if not seg.has_drone:
            continue
        usable = min(SegmentAudio(seg).starts) / SR - MARGIN_S
        if usable > 5:
            total += usable
            print(f"  {seg.id:<32}{usable:>7.1f}s")
    print(f"  {'total':<32}{total:>7.1f}s  ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
