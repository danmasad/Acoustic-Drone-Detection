"""Real continuous ambient audio, which is what a false-alarm rate actually needs.

Our own no-drone audio is 5.6 minutes, so the finest non-zero false-alarm rate we can even
express is one alarm in 5.6 minutes, or 10.7 per hour. Any claim below that is arithmetic, not
measurement. DEMAND gives 90 minutes across 18 environments at our own 48 kHz, which takes the
resolvable floor to about 0.7 per hour and, more usefully, replaces a single number with a
number plus a spread over soundscapes.

What this can and cannot say:

- It **can** say how often a detector fires on real-world noise it has never heard, per
  environment, which is the closest thing to a deployment figure this project can produce.
- It **cannot** say what our array would do in those places. Different microphones, different
  gain, different wind. The rate is reported as indicative with the mismatch stated, next to
  the coarse-but-ours 10.7 per hour resolution from the array's own ambient.

Only single channels are used, which is why the geometry mismatch does not matter: the
per-channel detector reads one microphone at a time by construction. The map model cannot be
evaluated on this at all, and pretending otherwise by replicating one channel 24 times would
manufacture a spatial pattern no real source produces.

Framing is deliberately identical to `src.esc50` and to our own cache — the same 4800-sample
non-overlapping rectangular window and the same band selection — so a frame from DEMAND is
interchangeable with a frame of ours as far as the model is concerned.
"""
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

from .config import CACHE, CACHE_BAND, CHUNK, DATA, SR
from .stft_cache import FILTER_BAND, FILTER_ORDER

DEMAND_DIR = DATA / "demand"
CACHE_PATH = CACHE / "demand" / "spectra.npz"

# The 18 scenes, grouped as DEMAND groups them. The grouping is the point: a false-alarm rate
# in a park and one on a metro platform are different claims, and averaging them hides both.
GROUPS = {
    "domestic": ("DKITCHEN", "DLIVING", "DWASHING"),
    "nature": ("NFIELD", "NPARK", "NRIVER"),
    "office": ("OHALLWAY", "OMEETING", "OOFFICE"),
    "public": ("PCAFETER", "PRESTO", "PSTATION"),
    "street": ("SCAFE", "SPSQUARE", "STRAFFIC"),
    "transport": ("TBUS", "TCAR", "TMETRO"),
}
GROUP_OF = {s: g for g, ss in GROUPS.items() for s in ss}

# Outdoor scenes with wind, traffic and open sky are the honest comparison for an array that
# sits in a field. Indoor scenes are reported too, but they are an easier problem.
OUTDOOR = ("NFIELD", "NPARK", "NRIVER", "SPSQUARE", "STRAFFIC", "PSTATION")


def build_cache(overwrite: bool = False, verbose: bool = True) -> Path:
    """Frame every scene's first channel into our own STFT layout."""
    if CACHE_PATH.exists() and not overwrite:
        if verbose:
            print(f"DEMAND spectral cache present: {CACHE_PATH}")
        return CACHE_PATH
    wavs = sorted(DEMAND_DIR.glob("*_ch01.wav"))
    if not wavs:
        raise FileNotFoundError(f"no scenes in {DEMAND_DIR}; run utils/download_demand.sh")

    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    sel = (freqs >= CACHE_BAND[0]) & (freqs <= CACHE_BAND[1])
    b, a = butter(FILTER_ORDER, [FILTER_BAND[0] / (SR / 2), FILTER_BAND[1] / (SR / 2)],
                  btype="band")

    spectra, scene, t0 = [], [], time.time()
    for path in wavs:
        name = path.name.split("_")[0]
        sr, x = wavfile.read(path)
        if sr != SR:
            raise ValueError(f"{path.name}: {sr} Hz, expected {SR}; the 48k release is the "
                             f"one we want, precisely so there is no resampling to defend")
        x = np.asarray(x, np.float64)
        if x.ndim > 1:
            x = x[:, 0]
        n = len(x) // CHUNK
        y = filtfilt(b, a, x)[:n * CHUNK].reshape(n, CHUNK)
        spectra.append(np.fft.rfft(y, axis=1)[:, sel].astype(np.complex64))
        scene += [name] * n
        if verbose:
            print(f"  {name}: {n} frames ({n * CHUNK / SR / 60:.1f} min), "
                  f"{time.time() - t0:.0f}s")

    S = np.concatenate(spectra)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_PATH, spectra=S, scene=np.array(scene), freqs=freqs[sel])
    if verbose:
        print(f"DEMAND cache: {len(S)} frames ({len(S) * CHUNK / SR / 60:.0f} min), "
              f"{CACHE_PATH.stat().st_size / 2**20:.0f} MiB")
    return CACHE_PATH


class Ambient:
    """Framed spectra of every DEMAND scene, all of them labelled no-drone.

    A gain question has to be answered before any of this means anything: DEMAND was recorded
    with its own microphones and its own preamp settings, so its absolute level says nothing
    about how loud that place would have been through our array. Rescaling each scene so its
    median in-band frame energy matches our own ambient's median is the least-arbitrary choice
    available — it holds the recording level fixed and lets the *spectral shape* and the
    *variability* of the scene be what the detector responds to. The alternative, using DEMAND's
    raw level, would confound "this soundscape is confusing" with "this recording was louder".
    """

    def __init__(self, reference_energy: float | None = None, verbose: bool = True):
        d = np.load(build_cache(verbose=verbose), allow_pickle=False)
        self.spectra = d["spectra"]
        self.scene = d["scene"]
        self.freqs = d["freqs"]
        self.scenes = sorted(set(self.scene.tolist()))
        self.gain = {}
        if reference_energy is not None:
            for s in self.scenes:
                m = self.scene == s
                med = float(np.median((np.abs(self.spectra[m]) ** 2).sum(-1)))
                g = np.sqrt(reference_energy / max(med, 1e-30))
                self.gain[s] = g
                self.spectra[m] *= np.float32(g)
        if verbose:
            print(f"DEMAND: {len(self.spectra)} frames over {len(self.scenes)} scenes, "
                  f"{len(self.spectra) * CHUNK / SR / 60:.0f} min"
                  + (", level-matched to our own ambient" if self.gain else ""))

    def hours(self, scene: str | None = None) -> float:
        n = len(self.spectra) if scene is None else int((self.scene == scene).sum())
        return n * CHUNK / SR / 3600.0

    def frames(self, scene: str, K: int = 1, channels: int = 1) -> np.ndarray:
        """(n, K, channels, F) complex64: contiguous K-frame stacks from one scene.

        The same single channel is repeated when more than one is asked for, and that is only
        ever valid for a model that reads channels independently.
        """
        S = self.spectra[self.scene == scene]
        n = len(S) - K + 1
        idx = np.arange(n)[:, None] + np.arange(K)[None, :]
        return np.repeat(S[idx][:, :, None, :], channels, axis=2)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build the DEMAND spectral cache.")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    build_cache(overwrite=a.overwrite)
    amb = Ambient(verbose=False)
    print(f"{len(amb.scenes)} scenes, {amb.hours():.2f} hours total, "
          f"so one false alarm over the whole set is {1 / amb.hours():.2f}/hour")
    for g, ss in GROUPS.items():
        have = [s for s in ss if s in amb.scenes]
        print(f"  {g:<10}{', '.join(have)}")
