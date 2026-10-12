"""Step 1 gate: do the ported geometry and labels still agree with the audio?

Repeats the original feasibility check (utils/validate/validate_dataset.py, which streamed the
audio over HTTP) using the pipeline modules and local files, on the same 15 s slice of
test_1. Delay-and-sum on the raw audio has no free parameters, so if the peak direction
still lands on the GPS direction then the UTM projection, the azimuth offset, the
per-recorder sample corrections and the microphone positions are all still consistent.

Expected, from the original run: median azimuth error 1.6 deg, elevation 5.1 deg, 91% of
frames within 10 deg azimuth.

Run: .venv/bin/python utils/validate/check_geometry_labels.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import AZ, CHUNK, EL, MODEL_BAND, SR, segments          # noqa: E402
from src.geometry import mic_positions, steering                        # noqa: E402
from src.labels import frame_labels                                     # noqa: E402
from src.stft_cache import FILTER_BAND, FILTER_ORDER, SegmentAudio      # noqa: E402

N_FRAMES = 150            # 15 s, the same slice the original check used
EXPECTED = {"az_median": 1.6, "el_median": 5.1, "within10": 0.91}
TOLERANCE = 0.5           # degrees; block filtering differs slightly from per-frame


def main() -> int:
    seg = next(s for s in segments("test_1") if s.has_drone)
    print(f"segment {seg.id}  skip={seg.skip_seconds}s  "
          f"initial_azimuth={seg.initial_azimuth_deg} deg")

    audio = SegmentAudio(seg)
    labels = frame_labels(seg, n_frames=N_FRAMES)
    n = min(N_FRAMES, audio.n_frames, len(labels["az"]))
    print(f"audio available {audio.n_frames} frames, using {n}")

    b, a = butter(FILTER_ORDER, [FILTER_BAND[0] / (SR / 2), FILTER_BAND[1] / (SR / 2)],
                  btype="band")
    blk = filtfilt(b, a, audio.read(0, n * CHUNK), axis=0).reshape(n, CHUNK, -1)

    freqs = np.fft.rfftfreq(CHUNK, 1 / SR)
    sel = (freqs >= MODEL_BAND[0]) & (freqs <= MODEL_BAND[1])
    W = steering(mic_positions(seg.align), freqs[sel])
    print(f"steering {W.shape} = {W.nbytes / 2**20:.0f} MB, {sel.sum()} bins, "
          f"{len(EL)}x{len(AZ)} directions")

    t0 = time.time()
    X = np.fft.rfft(blk, axis=1)[:, sel].astype(np.complex64)            # (n, F, M)
    Y = np.einsum("fdm,nfm->nfd", W, X, optimize=True)
    energy = (np.abs(Y) ** 2).sum(1).reshape(n, len(EL), len(AZ))
    print(f"beamformed {n} frames in {time.time() - t0:.1f}s "
          f"({n / (time.time() - t0):.0f} fps, CPU)")

    j, i = np.unravel_index(energy.reshape(n, -1).argmax(1), (len(EL), len(AZ)))
    az_err = np.abs(((AZ[i] - labels["az"][:n] + 180) % 360) - 180)
    el_err = np.abs(EL[j] - labels["el"][:n])
    got = {"az_median": float(np.median(az_err)),
           "el_median": float(np.median(el_err)),
           "within10": float((az_err <= 10).mean())}

    print(f"\ndrone distance {labels['range_m'][:n].min():.0f}-"
          f"{labels['range_m'][:n].max():.0f} m, "
          f"speed {labels['speed_mps'][:n].mean():.1f} m/s mean")
    print(f"azimuth error    median {got['az_median']:.1f} deg  "
          f"mean {az_err.mean():.1f}  (expected {EXPECTED['az_median']})")
    print(f"elevation error  median {got['el_median']:.1f} deg  "
          f"mean {el_err.mean():.1f}  (expected {EXPECTED['el_median']})")
    print(f"within 10 deg    {got['within10'] * 100:.0f}% of frames  "
          f"(expected {EXPECTED['within10'] * 100:.0f}%)")

    ok = (abs(got["az_median"] - EXPECTED["az_median"]) <= TOLERANCE
          and abs(got["el_median"] - EXPECTED["el_median"]) <= TOLERANCE
          and got["within10"] >= EXPECTED["within10"] - 0.03)
    print("\nGATE PASS" if ok else "\nGATE FAIL: alignment or projection has drifted")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
