"""Step 2 gate: is the cache complete, correctly sized, and still frame-aligned?

Three checks, in increasing strength:

1. Every segment's frame count matches its declared duration and its label arrays.
2. Cached spectra reproduce a recomputation straight from the WAVs, which is what proves
   the block-wise filtering introduced no edge artefacts and no off-by-one in indexing.
3. Steering the *cached* frames of test_1 reproduces the step 1 direction accuracy exactly
   (median 1.6 deg azimuth). This is the end-to-end check: a one-frame slip between audio
   and labels would show up here as degraded accuracy, and nothing else in the pipeline
   would notice.

Run: .venv/bin/python utils/validate/check_stft_cache.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import AZ, CHUNK, CSV_HZ, EL, MODEL_BAND, SR, segments      # noqa: E402
from src.geometry import mic_positions, steering                            # noqa: E402
from src.stft_cache import SegmentAudio, bandpass, load_cache               # noqa: E402

RECHECK_FRAMES = [0, 301, 705]        # 705 sits across a filtering block boundary
N_STEER = 150                         # the same slice step 1 used
EXPECTED_AZ_MEDIAN = 1.6


def recompute(seg, frame: int) -> np.ndarray:
    """(24, F) spectrum of one frame, computed from the WAVs with generous filter context."""
    audio = SegmentAudio(seg)
    pad = CHUNK
    lo = max(0, frame * CHUNK - pad)
    hi = min(audio.n_samples, (frame + 1) * CHUNK + pad)
    blk = bandpass(audio.read(lo, hi - lo))
    head = frame * CHUNK - lo
    return np.fft.rfft(blk[head:head + CHUNK], axis=0).T


def main() -> int:
    ok = True
    total = 0.0
    print(f"{'segment':<32}{'frames':>8}{'declared':>10}{'labels':>8}{'GiB':>7}")
    for seg in segments():
        stft, meta = load_cache(seg)
        n = len(stft)
        declared = (None if seg.skip_seconds_end is None else
                    int(round((seg.skip_seconds_end - seg.skip_seconds) * CSV_HZ)))
        size = (seg.cache_dir / "stft.npy").stat().st_size / 2**30
        total += size
        print(f"{seg.id:<32}{n:>8}{declared if declared else '-':>10}"
              f"{len(meta['has_drone']):>8}{size:>7.2f}")
        if len(meta["has_drone"]) != n:
            print("   FAIL: label length disagrees with frame count")
            ok = False
        if declared is not None and n != declared:
            print(f"   FAIL: {n} frames but the segment declares {declared}")
            ok = False
        if seg.has_drone != bool(meta["has_drone"].max()):
            print("   FAIL: drone flag disagrees with the segment role")
            ok = False
    print(f"{'total':<32}{'':>8}{'':>10}{'':>8}{total:>7.2f}")

    seg = next(s for s in segments("test_1") if s.has_drone)
    stft, meta = load_cache(seg)
    freqs = meta["freqs"]

    print(f"\nrecomputing frames {RECHECK_FRAMES} from the WAVs")
    for f in RECHECK_FRAMES:
        got, want = stft[f], recompute(seg, f)[:, np.isin(np.fft.rfftfreq(CHUNK, 1 / SR),
                                                          freqs)]
        rel = np.abs(got - want).max() / np.abs(want).max()
        print(f"  frame {f:>4}: max relative difference {rel:.2e}")
        if rel > 1e-3:
            print("   FAIL: cached spectrum does not match the audio")
            ok = False

    sel = (freqs >= MODEL_BAND[0]) & (freqs <= MODEL_BAND[1])
    W = steering(mic_positions(seg.align), freqs[sel])
    X = np.asarray(stft[:N_STEER])[:, :, sel]
    Y = np.einsum("fdm,nmf->nfd", W, X, optimize=True)
    energy = (np.abs(Y) ** 2).sum(1)
    j, i = np.unravel_index(energy.argmax(1), (len(EL), len(AZ)))
    az_err = np.abs(((AZ[i] - meta["az"][:N_STEER] + 180) % 360) - 180)
    el_err = np.abs(EL[j] - meta["el"][:N_STEER])
    med = float(np.median(az_err))
    print(f"\nsteering {N_STEER} cached frames of {seg.id} with all 24 microphones")
    print(f"  azimuth median {med:.1f} deg (step 1 measured {EXPECTED_AZ_MEDIAN}), "
          f"elevation median {np.median(el_err):.1f}, "
          f"within 10 deg {(az_err <= 10).mean() * 100:.0f}%")
    if abs(med - EXPECTED_AZ_MEDIAN) > 0.5:
        print("   FAIL: cached frames are not aligned with the labels")
        ok = False

    print("\nGATE PASS" if ok else "\nGATE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
