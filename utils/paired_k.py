"""Does temporal context pay only where single-frame evidence is weak?

The headline grid compared two pooled averages and found +1.8 points on one array against a
+/-1.2 point seed spread: no reliable effect. But a pooled average is exactly where a
subgroup effect goes to die. The mechanism we measured — a real drone persists while sidelobe
phantoms flicker — should only pay when a single frame is ambiguous, which is the far field. If
context helps at 150 m and hurts near the array, the pooled number is zero and both facts are
invisible.

Two sharpenings over the grid, at no extra compute:

- **Pairing on frames.** K=1 and K=5 saw the same audio, so the same frame can be scored under
  both and the difference taken per frame. That removes all between-frame variance, which is
  the largest term by far, and it is what makes a subgroup analysis feasible at three seeds.
- **Conditioning on range**, which is the axis the mechanism predicts.

One statistical caution, stated because it would be easy to oversell: frames 100 ms apart are
strongly autocorrelated, so per-frame counts are not independent samples and a McNemar p-value
computed over them would be far too confident. The per-frame counts below are descriptive; the
error bar that means anything is the spread across seeds, and that is what the verdict uses.

K=5 drops the first four frames of every segment, so the two runs do not cover identical frame
sets; everything here is computed on the intersection, joined on (segment, frame).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS                                   # noqa: E402
from src.metrics import DISTANCE_BINS, angular_errors            # noqa: E402
from src.predict import load                                     # noqa: E402

SUBSETS = ("all24", "device3", "best6")
SEEDS = (0, 1, 2)
THRESHOLD = 10.0


def hits(pred: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """(seg, frame) -> row, plus the per-frame hit indicator and range, drone frames only."""
    d = pred["has_drone"].astype(bool)
    err = angular_errors(pred["el_idx"][d], pred["az_idx"][d], pred["az"][d], pred["el"][d])
    key = {(int(s), int(f)): i for i, (s, f) in enumerate(zip(pred["seg"][d],
                                                              pred["frame"][d]))}
    return key, (err <= THRESHOLD), np.asarray(pred["range_m"], float)[d]


def paired(subset: str, seed: int, split: str, bins=None) -> dict | None:
    try:
        k1 = load(f"{subset}_K1_w32_b16_wce_s{seed}_{split}_causal")
        k5 = load(f"{subset}_K5_w32_b16_wce_s{seed}_{split}_causal")
    except FileNotFoundError:
        return None
    key1, hit1, rng1 = hits(k1)
    key5, hit5, _ = hits(k5)
    common = sorted(set(key1) & set(key5))
    i1 = np.array([key1[k] for k in common])
    i5 = np.array([key5[k] for k in common])
    a, b, r = hit1[i1], hit5[i5], rng1[i1]
    out = {"n": len(common), "k1": float(a.mean()), "k5": float(b.mean()),
           "delta": float(b.mean() - a.mean()),
           "k5_only": int((b & ~a).sum()), "k1_only": int((a & ~b).sum()), "by_range": []}
    for lo, hi in (bins or list(zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]))):
        m = (r >= lo) & (r < hi)
        out["by_range"].append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                                "delta": float(b[m].mean() - a[m].mean()) if m.any()
                                else float("nan")})
    return out


def occupied_bins(split: str) -> list:
    """Range bins with drone frames in them, so empty bins never reach an average."""
    pred = load(f"baseline_all24_{split}")
    d = pred["has_drone"].astype(bool)
    r = np.asarray(pred["range_m"], float)
    return [(lo, hi) for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:])
            if (d & (r >= lo) & (r < hi)).any()]


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    results = {"split": split, "threshold_deg": THRESHOLD, "subsets": {}}
    bins = occupied_bins(split)
    names = [f"{int(lo)}-{'inf' if not np.isfinite(hi) else int(hi)}" for lo, hi in bins]

    print(f"\nFrame-paired K=5 minus K=1, frames within {THRESHOLD:.0f} deg, percentage "
          f"points. Mean +/- sd over seeds.\n")
    print(f"  {'array':<10}{'overall':>12}" + "".join(f"{n:>10}" for n in names))
    for subset in SUBSETS:
        runs = [paired(subset, s, split, bins) for s in SEEDS]
        runs = [r for r in runs if r]
        if not runs:
            continue
        results["subsets"][subset] = runs
        d = np.array([r["delta"] for r in runs]) * 100
        per_bin = np.array([[b["delta"] for b in r["by_range"]] for r in runs]) * 100
        sd = f"+/-{d.std(ddof=1):.1f}" if len(d) > 1 else ""
        print(f"  {subset:<10}{d.mean():>+8.1f} {sd:<4}"
              + "".join(f"{v:>+10.1f}" for v in np.nanmean(per_bin, 0)))

    print("\nPer-frame disagreement counts (descriptive only, frames are autocorrelated):")
    for subset, runs in results["subsets"].items():
        k5 = sum(r["k5_only"] for r in runs)
        k1 = sum(r["k1_only"] for r in runs)
        n = sum(r["n"] for r in runs)
        print(f"  {subset:<10}{k5:>7} frames K=5 alone gets right, {k1:>7} K=1 alone, "
              f"out of {n} paired ({100 * (k5 + k1) / n:.0f}% disagree at all)")

    # Verdict: is there a range bin where context helps by more than the seed spread?
    print("\nWhere context pays, if anywhere:")
    any_effect = False
    for subset, runs in results["subsets"].items():
        per_bin = np.array([[b["delta"] for b in r["by_range"]] for r in runs]) * 100
        counts = [b["n"] for b in runs[0]["by_range"]]
        for j, name in enumerate(names):
            col = per_bin[:, j]
            if np.isnan(col).any() or counts[j] < 200:
                continue
            m, sd = col.mean(), (col.std(ddof=1) if len(col) > 1 else np.inf)
            if abs(m) > 2 * sd:
                any_effect = True
                print(f"  {subset} at {name} m: {m:+.1f} pp +/- {sd:.1f} over seeds "
                      f"({counts[j]} frames)")
    if not any_effect:
        print("  Nowhere. No range bin shows an effect larger than twice its seed spread, so "
              "the pooled null is not hiding a far-field gain.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "paired_k.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS / 'paired_k.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test_2"))
