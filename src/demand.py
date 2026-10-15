"""DEMAND: real continuous ambient audio for false-alarm rates.

Test 2's no-drone audio is 5.6 minutes, so the finest non-zero false-alarm rate it can
measure is one alarm in 5.6 minutes, or 10.7 per hour. DEMAND gives 90 minutes across 18
environments at our 48 kHz, which lowers that floor to about 0.7 per hour and reports a spread
over soundscapes alongside the overall rate.

It measures how often a detector fires on real ambient it has never heard, per environment. It
does not say what our array would record there: the microphones, gain and wind differ.

Only single channels are used, so only the per-channel detector can be evaluated. Replicating
one channel 24 times for the map model would create a spatial pattern no real source produces.

Framing matches `src.esc50` and our own cache: the same band-pass, 4800-sample rectangular
frames and frequency bins, so a DEMAND frame is interchangeable with one of ours.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .config import CACHE, CACHE_BAND, CHUNK, DATA, SR
from .stft_cache import bandpass

DEMAND_DIR = DATA / "demand"
CACHE_PATH = CACHE / "demand" / "spectra.npz"

# The 18 scenes in DEMAND's own groups. Rates are reported per group, since a park and a metro
# platform are different conditions.
GROUPS = {
    "domestic": ("DKITCHEN", "DLIVING", "DWASHING"),
    "nature": ("NFIELD", "NPARK", "NRIVER"),
    "office": ("OHALLWAY", "OMEETING", "OOFFICE"),
    "public": ("PCAFETER", "PRESTO", "PSTATION"),
    "street": ("SCAFE", "SPSQUARE", "STRAFFIC"),
    "transport": ("TBUS", "TCAR", "TMETRO"),
}
GROUP_OF = {s: g for g, ss in GROUPS.items() for s in ss}

# Outdoor scenes, with wind, traffic and open sky, are the closest match to an array in a
# field. Indoor scenes are reported too.
OUTDOOR = ("NFIELD", "NPARK", "NRIVER", "SPSQUARE", "STRAFFIC", "PSTATION")


def build_cache(overwrite: bool = False, verbose: bool = True) -> Path:
    """Frame every scene's first channel into our own STFT layout."""
    if CACHE_PATH.exists() and not overwrite:
        if verbose:
            print(f"DEMAND spectral cache present: {CACHE_PATH}")
        return CACHE_PATH
    wavs = sorted(DEMAND_DIR.glob("*_ch01.wav"))
    if not wavs:
        raise FileNotFoundError(f"no scenes in {DEMAND_DIR}; run utils/setup/download_demand.sh")

    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    sel = (freqs >= CACHE_BAND[0]) & (freqs <= CACHE_BAND[1])

    spectra, scene, t0 = [], [], time.time()
    for path in wavs:
        name = path.name.split("_")[0]
        sr, x = wavfile.read(path)
        if sr != SR:
            raise ValueError(f"{path.name}: {sr} Hz, expected {SR} "
                             f"(the 48 kHz release, so nothing is resampled)")
        x = np.asarray(x, np.float64)
        if x.ndim > 1:
            x = x[:, 0]
        n = len(x) // CHUNK
        y = bandpass(x)[:n * CHUNK].reshape(n, CHUNK)
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

    DEMAND's absolute level reflects its own microphones and gain, not our array. With
    `reference_energy`, each scene is rescaled so its median in-band frame energy equals our
    ambient's median, so the detector responds to the scene's spectral shape and variability
    rather than to recording level.
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
        """Duration in hours of one scene, or of all scenes when `scene` is None."""
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
