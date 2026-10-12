"""Do our numbers reproduce the reference paper's, or did we lose something in reimplementation?

Their results live in figures rather than tables, so the comparison has to be built rather than
looked up. Three things had to match before any number is comparable:

- **Their bins**, 0-50 / 50-100 / 100-200 m, not our 30 m bins.
- **Mean** angular error, not our median. On a heavy-tailed error distribution these differ by
  a lot, and quoting our median against their mean would flatter us by ~16 degrees.
- **Their false-positive rule.** They never threshold an energy; they call a frame a detection
  when the direction estimate is *stable*, defined as the centroid standard deviation over the
  last 3 observations staying under 10 degrees (section 4.1). On five minutes of no-drone audio
  DAS was stable 67.0% of the time and their U-Net 14.9%. Reproducing that means implementing
  their rule, not ours. The paper does not say how the standard deviation of three directions
  is reduced to one number, so we use the root-mean-square angular distance to the mean
  direction, which is the natural spherical reading.

Paper values below are read off Fig. 5 of arXiv:2508.00307v1 to the nearest degree; they are
bar heights, so treat them as +/- 1 degree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import AZ, EL, run_dir, run_id                        # noqa: E402
from src.geometry import angular_distance, unit_vectors               # noqa: E402
from src.metrics import angular_errors                                # noqa: E402
from src.predict import baseline_predictions, cnn_predictions         # noqa: E402

PAPER_BINS = (0.0, 50.0, 100.0, 200.0)
PAPER = {                       # Fig. 5, mean angular error in degrees
    ("beamforming", "test_2"): (38, 75, 33),
    ("unet", "test_2"): (10, 13, 14),
    ("beamforming", "test_1"): (8, 29, 92),
    ("unet", "test_1"): (12, 24, 54),
}
PAPER_FP = {"beamforming": 67.0, "unet": 14.9}      # section 4.1, five-minute no-drone audio
STABILITY_WINDOW = 3
STABILITY_DEG = 10.0


def binned_mean(err: np.ndarray, range_m: np.ndarray, bins=PAPER_BINS) -> list:
    """(mean error, frame count) per range bin; NaN where the bin is empty."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (range_m >= lo) & (range_m < hi) & np.isfinite(err)
        out.append((float(err[m].mean()) if m.any() else float("nan"), int(m.sum())))
    return out


def stability_flags(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Their detection rule: the last 3 direction estimates agree to within 10 degrees.

    Root-mean-square angular distance to the mean of the window, which is the spherical
    equivalent of a standard deviation. Frames before the window fills are False.
    """
    v = unit_vectors(az, el)                                   # (T, 3)
    stable = np.zeros(len(v), bool)
    for t in range(STABILITY_WINDOW - 1, len(v)):
        w = v[t - STABILITY_WINDOW + 1:t + 1]
        mean = w.mean(0)
        norm = np.linalg.norm(mean)
        if norm < 1e-9:
            continue
        d = np.degrees(np.arccos(np.clip(w @ (mean / norm), -1.0, 1.0)))
        stable[t] = float(np.sqrt((d ** 2).mean())) <= STABILITY_DEG
    return stable


def with_decoder(pred: dict, mode: str = "centroid") -> dict:
    """Point `pred_az`/`pred_el` at the chosen decoder's columns.

    A dump from `src.predict` carries both decoders, so switching between the paper's
    centroid and our brightest pixel is a column swap rather than another forward pass.
    """
    if mode == "argmax":
        return {**pred, "pred_az": pred["argmax_az"], "pred_el": pred["argmax_el"]}
    return pred


def report(name: str, paper_key: str, split: str, pred: dict) -> None:
    d = pred["has_drone"].astype(bool)
    if "pred_az" in pred:
        err = angular_distance(pred["pred_az"][d], pred["pred_el"][d],
                               pred["az"][d], pred["el"][d])
    else:
        err = angular_errors(pred["el_idx"][d], pred["az_idx"][d], pred["az"][d], pred["el"][d])
    ours = binned_mean(err, np.asarray(pred["range_m"])[d])
    theirs = PAPER[(paper_key, split)]

    print(f"\n{name} on {split}")
    print(f"  {'range':<12}{'paper':>9}{'ours':>9}{'diff':>9}{'frames':>9}")
    for (lo, hi), (mine, n), t in zip(zip(PAPER_BINS[:-1], PAPER_BINS[1:]), ours, theirs):
        diff = f"{mine - t:+.0f}" if np.isfinite(mine) else "—"
        val = f"{mine:.0f}d" if np.isfinite(mine) else "—"
        print(f"  {f'{lo:.0f}-{hi:.0f} m':<12}{t:>8}d{val:>9}{diff:>9}{n:>9}")

    # Their false-positive rule, on the no-drone frames of the same split.
    if (~d).any():
        p_az = pred["pred_az"] if "pred_az" in pred else np.asarray(AZ)[pred["az_idx"]]
        p_el = pred["pred_el"] if "pred_el" in pred else np.asarray(EL)[pred["el_idx"]]
        stable = stability_flags(p_az, p_el)
        fp = 100.0 * stable[~d].mean()
        print(f"  {'stable-direction false positives':<34}{PAPER_FP[paper_key]:>6.1f}% paper"
              f"   {fp:>5.1f}% ours   ({int((~d).sum())} no-drone frames)")


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    print(f"\nAll numbers are mean angular error in degrees, drone frames only, "
          f"all 24 microphones (their array).")

    report("DAS beamforming, brightest pixel", "beamforming", split,
           baseline_predictions(split, "all24", verbose=False))

    # Both decoders, because this is where we suspected the reimplementation gap lives: they
    # centroid an eroded mask, we were taking the brightest pixel.
    for mode in ("argmax", "centroid"):
        for seed in (0, 1, 2):
            run = run_id("all24", 1, seed=seed)
            if (run_dir(run) / "best.pt").exists():
                pred = cnn_predictions(run, split, verbose=False)
                report(f"our U-Net, {mode} decoding (seed {seed})", "unet", split,
                       with_decoder(pred, mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test_2"))
