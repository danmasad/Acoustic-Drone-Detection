"""ESC-50 confusers, mixed into our own audio in the spectral domain.

Clips are added on top of our real frames, so every example keeps our array's noise floor and
"no drone" cannot be predicted from a clip's own recording gain or microphone.

Our frames are non-overlapping rectangular 4800-sample windows, so adding spectra is exactly
adding waveforms, and confusers are mixed into the cached spectra during training.

The same clip lands on every channel, a spatial pattern no real source produces. The
per-channel detector reads one channel at a time and cannot see it; the map model sees across
channels, so these mixtures are for the detector only.

The result is a confuser-rejection rate (how often a chainsaw reads as a drone). The field
false-alarm rate comes from real ambient audio (`src.demand`).
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from .config import CACHE, CACHE_BAND, CHUNK, DATA, SR
from .stft_cache import bandpass

ESC_DIR = DATA / "esc50"
CACHE_PATH = CACHE / "esc50" / "spectra.npz"
ESC_SR = 44100
TEST_FOLD = 5          # held out from training mixes, used for the reported rejection number

# Classes an acoustic drone detector can plausibly confuse for a multirotor: broadband
# rotating machinery, or anything with a strong harmonic stack in 200-2200 Hz. Reported
# separately, because a mean over 50 classes hides exactly the cases that matter.
HARD_CLASSES = ("helicopter", "airplane", "chainsaw", "engine", "vacuum_cleaner",
                "washing_machine", "hand_saw", "wind")


def build_cache(overwrite: bool = False, verbose: bool = True) -> Path:
    """Resample every clip to 48 kHz and store its frames in our own STFT layout."""
    if CACHE_PATH.exists() and not overwrite:
        if verbose:
            print(f"ESC-50 spectral cache present: {CACHE_PATH}")
        return CACHE_PATH
    meta_csv = ESC_DIR / "meta" / "esc50.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(f"{meta_csv} missing; run utils/setup/download_esc50.sh first")
    rows = list(csv.DictReader(meta_csv.open()))

    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    sel = (freqs >= CACHE_BAND[0]) & (freqs <= CACHE_BAND[1])

    spectra, fold, category, clip_of_frame = [], [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        sr, x = wavfile.read(ESC_DIR / "audio" / row["filename"])
        if sr != ESC_SR:
            raise ValueError(f"{row['filename']}: {sr} Hz, expected {ESC_SR}")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(1)
        x = resample_poly(x, 160, 147)                       # 44100 -> 48000
        n = len(x) // CHUNK
        if n == 0:
            continue
        y = bandpass(x)[:n * CHUNK].reshape(n, CHUNK)
        spectra.append(np.fft.rfft(y, axis=1)[:, sel].astype(np.complex64))
        fold += [int(row["fold"])] * n
        category += [row["category"]] * n
        clip_of_frame += [i] * n
        if verbose and (i + 1) % 200 == 0:
            print(f"\r  {i + 1}/{len(rows)} clips, {time.time() - t0:.0f}s",
                  end="", flush=True)

    S = np.concatenate(spectra)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, spectra=S, fold=np.array(fold, np.int8),
             category=np.array(category), clip=np.array(clip_of_frame, np.int32),
             freqs=freqs[sel])
    if verbose:
        print(f"\rESC-50 cache: {len(S)} frames ({len(S) * CHUNK / SR / 60:.0f} min), "
              f"{CACHE_PATH.stat().st_size / 2**20:.0f} MiB, {time.time() - t0:.0f}s")
    return CACHE_PATH


class Confusers:
    """Frames of ESC-50, split by fold, ready to add onto our own spectra.

    `snr_db` is the confuser's in-band energy relative to the frame it lands on, so the
    scaling adapts to loud and quiet frames alike instead of being an absolute gain.
    """

    def __init__(self, folds=None, snr_db=(-5.0, 20.0), verbose: bool = True):
        d = np.load(build_cache(verbose=verbose), allow_pickle=False)
        keep = np.isin(d["fold"], list(folds)) if folds is not None else \
            np.ones(len(d["fold"]), bool)
        self.spectra = d["spectra"][keep]
        self.category = d["category"][keep]
        # Clip index per frame, so a K-frame stack can stay within one clip.
        self.clip = d["clip"][keep]
        self.snr_db = snr_db
        if verbose:
            print(f"confusers: {len(self.spectra)} frames from folds "
                  f"{sorted(set(d['fold'][keep].tolist()))}, "
                  f"{len(set(self.category))} classes")

    def __len__(self) -> int:
        return len(self.spectra)

    def stack(self, K: int, rng) -> tuple[np.ndarray, str]:
        """K temporally continuous frames from one clip, (K, F) complex64."""
        while True:
            i = int(rng.integers(len(self.spectra) - K)) if K > 1 else \
                int(rng.integers(len(self.spectra)))
            if K == 1 or self.clip[i] == self.clip[i + K - 1]:
                return self.spectra[i:i + K], str(self.category[i])

    def mix_into(self, X, rng, prob: float = 0.5, snr_db=None):
        """(B, K, M, F) -> the same tensor with confusers added to a random subset of items.

        Returns (mixed, mixed_flag, categories). The flag matters for reporting: a mixed
        *positive* frame is still a positive, and mixing into both classes is what stops
        "contains an added sound" from becoming a perfect cue for the negative class.
        """
        B, K, M, F = X.shape
        lo, hi = snr_db if snr_db is not None else self.snr_db
        Xn = X.numpy() if isinstance(X, torch.Tensor) else X
        out = Xn.copy()
        flag = np.zeros(B, bool)
        cats: list[str] = [""] * B
        for bi in range(B):
            if rng.random() >= prob:
                continue
            clip, cat = self.stack(K, rng)
            e_clip = float((np.abs(clip) ** 2).sum())
            e_frame = float((np.abs(Xn[bi]) ** 2).sum() / M)
            if e_clip <= 0 or e_frame <= 0:
                continue
            snr = float(rng.uniform(lo, hi))
            alpha = np.sqrt(e_frame * 10 ** (snr / 10) / e_clip)
            out[bi] += (alpha * clip)[:, None, :].astype(np.complex64)
            flag[bi], cats[bi] = True, cat
        return torch.from_numpy(out), flag, cats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the ESC-50 spectral cache.")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    build_cache(overwrite=a.overwrite)
    d = np.load(CACHE_PATH, allow_pickle=False)
    cats, counts = np.unique(d["category"], return_counts=True)
    print(f"{len(cats)} classes, {counts.min()}-{counts.max()} frames each")
    print("hard classes present: " + ", ".join(c for c in HARD_CLASSES if c in set(cats)))
