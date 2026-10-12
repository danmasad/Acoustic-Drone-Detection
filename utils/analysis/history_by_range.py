"""Where does temporal context actually pay? Broken out by range, from dumps already on disk.

Two facts about history are established and pooled, which is the problem. Freezing the history at
inference — feeding the same current frame K times, so the weights and the frame are untouched and
only the motion is removed — costs 4.3 pp of hit rate, while shuffling or reversing it costs 0.1
and 0.2. So the model uses the extra frames as evidence rather than as motion. And on the small
array the depth curve is monotonic, 69.3% at K=1 to 73.2% at K=9, while on all 24 microphones it
is flat.

Neither number says *where* the gain lands, and that decides what it means:

- If the gain concentrates beyond 100 m, history is buying signal-to-noise on faint targets, and
  it speaks directly to our one reproduction gap, which is 100-200 m azimuth.
- If it is uniform across range, history is doing something cheaper and more general — averaging
  away beamformer sidelobes at every distance — and the far-field gap stays unexplained.

This is the cleanest comparison in the project and it costs nothing to run. Depth is varied by
`reorder` at inference on one set of weights, so a depth-1 and a depth-9 prediction come from the
same model scoring the same frame in the same order. The pairing is exact: no seed variance, no
training variance, one frame against itself. The only caveat is the usual one — frames are 100 ms
apart and a flight crosses a range bin in one continuous stretch, so the quoted interval is a
floor rather than an honest confidence interval.

Reads `results/preds/` only, and never calls the GPU, so it is safe to run while training.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import predict                                            # noqa: E402
from src.config import ANGULAR_THRESHOLD, FIGURES, table_file      # noqa: E402
from src.geometry import angular_distance                          # noqa: E402

BINS = ((0, 30), (30, 60), (60, 90), (90, 120), (120, 180))
SEEDS = (0, 1, 2)
# device3 K=9 first: it is the only cell with the full depth range, and the only array where the
# depth curve moves at all, so it is where a range breakdown can say anything.
CELLS = {"device3 K=9": ("device3_K9_w32_b16_wce_s{s}", 9),
         "device3 K=5": ("device3_K5_w32_b16_wce_s{s}", 5),
         "best6 K=5": ("best6_K5_w32_b16_wce_s{s}", 5),
         "all24 K=5": ("all24_K5_w32_b16_wce_s{s}", 5)}


def order_of(j: int, K: int) -> str:
    """`freeze{j}` keeps j frames of real history; at j == K nothing is frozen."""
    return "causal" if j >= K else f"freeze{j}"


def frame_errors(name: str):
    """(hit indicator, azimuth error, range) over drone frames, or None if not dumped."""
    try:
        d = predict.load(name)
    except FileNotFoundError:
        return None
    m = d["has_drone"].astype(bool)
    err = angular_distance(d["pred_az"][m], d["pred_el"][m], d["az"][m], d["el"][m])
    az = np.abs((d["pred_az"][m] - d["az"][m] + 180) % 360 - 180) * np.cos(
        np.radians(d["el"][m]))
    return (err <= ANGULAR_THRESHOLD).astype(float), az, d["range_m"][m]


def cell_depths(template: str, K: int, split: str):
    """{depth: (hit, azimuth, range)} averaged over seeds, frames kept aligned."""
    out = {}
    for j in range(1, K + 1):
        per_seed = [frame_errors(f"{template.format(s=s)}_{split}_{order_of(j, K)}")
                    for s in SEEDS]
        got = [p for p in per_seed if p is not None]
        if not got:
            continue
        n = min(len(g[0]) for g in got)
        out[j] = (np.mean([g[0][:n] for g in got], 0), np.mean([g[1][:n] for g in got], 0),
                  got[0][2][:n], len(got))
    return out


def by_bin(v: np.ndarray, rng: np.ndarray, stat=np.mean) -> list:
    return [float(stat(v[(rng >= lo) & (rng < hi)]))
            if ((rng >= lo) & (rng < hi)).any() else float("nan") for lo, hi in BINS]


def counts(rng: np.ndarray) -> list:
    return [int(((rng >= lo) & (rng < hi)).sum()) for lo, hi in BINS]


def report(name: str, depths: dict) -> dict:
    if len(depths) < 2:
        print(f"\n{name}: only {len(depths)} depth(s) dumped, skipping")
        return {}
    lo_j, hi_j = min(depths), max(depths)
    hit_lo, az_lo, rng, n_seeds = depths[lo_j]
    hit_hi, az_hi, _, _ = depths[hi_j]

    head = "".join(f"{f'{a}-{b}m':>11}" for a, b in BINS)
    print(f"\n{name}, {n_seeds} seeds, hit rate within {ANGULAR_THRESHOLD:.0f} degrees")
    print(f"  {'frames':<12}" + "".join(f"{c:>11}" for c in counts(rng)) + f"{'all':>11}")
    for j in sorted(depths):
        h, _, r, _ = depths[j]
        print(f"  {f'depth {j}':<12}"
              + "".join(f"{100 * v:>10.1f}%" for v in by_bin(h, r))
              + f"{100 * h.mean():>10.1f}%")
    print(f"  {'':<12}{head}")

    d = hit_hi - hit_lo
    print(f"\n  frame-paired gain, depth {hi_j} against depth {lo_j} (percentage points):")
    rows = []
    for (a, b), n in zip(BINS, counts(rng)):
        m = (rng >= a) & (rng < b)
        g = d[m] if m.any() else np.array([np.nan])
        se = float(g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
        rows.append({"bin": [a, b], "n_frames": n, "gain_pp": float(100 * np.mean(g)),
                     "se_floor_pp": 100 * se,
                     "azimuth_gain_deg": float(np.mean(az_lo[m]) - np.mean(az_hi[m]))
                     if m.any() else float("nan")})
        print(f"    {f'{a}-{b}m':>10}  {100 * np.mean(g):>+6.1f} +- {100 * se:<5.1f}"
              f"  azimuth {rows[-1]['azimuth_gain_deg']:>+6.1f} deg")
    print(f"    {'pooled':>10}  {100 * d.mean():>+6.1f}")
    return {"n_seeds": n_seeds, "depth_low": lo_j, "depth_high": hi_j,
            "pooled_gain_pp": float(100 * d.mean()), "by_bin": rows,
            "hit_by_depth": {j: by_bin(depths[j][0], depths[j][2]) for j in sorted(depths)},
            "pooled_by_depth": {j: float(depths[j][0].mean()) for j in sorted(depths)}}


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    out = {"split": split, "bins": [list(b) for b in BINS], "cells": {}}
    plots = {}
    for name, (template, K) in CELLS.items():
        depths = cell_depths(template, K, split)
        r = report(name, depths)
        if r:
            out["cells"][name] = r
            plots[name] = depths

    main_cell = "device3 K=9"
    if main_cell in out["cells"]:
        rows = out["cells"][main_cell]["by_bin"]
        near = [r for r in rows if r["bin"][1] <= 90 and r["n_frames"] > 100]
        far = [r for r in rows if r["bin"][0] >= 90 and r["n_frames"] > 100]
        near_g = float(np.mean([r["gain_pp"] for r in near])) if near else float("nan")
        far_g = float(np.mean([r["gain_pp"] for r in far])) if far else float("nan")
        print(f"\nVerdict on {main_cell}: {near_g:+.1f} pp inside 90 m against {far_g:+.1f} pp "
              f"beyond it.")
        if np.isnan(far_g):
            verdict = ("The far bins hold too few frames to say anything, which is itself the "
                       "answer: this array does not reach far enough to test the idea.")
        elif far_g > near_g + 3:
            verdict = ("The gain is concentrated at long range, which is what a "
                       "signal-to-noise story predicts and it lands on exactly the bin our "
                       "reproduction gap lives in. Temporal integration is buying faint-target "
                       "sensitivity, not motion.")
        elif far_g < near_g - 3:
            verdict = ("The gain is in the near field, where the target is already loud. That "
                       "rules out the signal-to-noise story: history is sharpening a peak that "
                       "was already there rather than finding a faint one.")
        else:
            verdict = ("The gain is flat across range. History is not rescuing faint targets, it "
                       "is averaging away beamformer noise everywhere, which leaves the "
                       "far-field gap unexplained by anything temporal.")
        print(f"  {verdict}")
        out["verdict"] = verdict
        out["near_gain_pp"], out["far_gain_pp"] = near_g, far_g

    path = table_file("history_by_range.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    if plots:
        plot(plots, split)
    return 0


def plot(cells: dict, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres = [(a + b) / 2 for a, b in BINS]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    name, depths = next(iter(cells.items()))
    lo_j, hi_j = min(depths), max(depths)
    for j in sorted(depths):
        h, _, r, _ = depths[j]
        shade = 0.15 + 0.75 * (j - lo_j) / max(hi_j - lo_j, 1)
        ax.plot(centres, [100 * v for v in by_bin(h, r)], marker="o",
                color=plt.cm.viridis(shade),
                label=f"depth {j}" if j in (lo_j, hi_j) else None)
    ax.set_title(f"{name}: hit rate by range, dark to light is 1 to {hi_j} frames", fontsize=9)
    ax.set_ylabel(f"within {ANGULAR_THRESHOLD:.0f} deg (%)")

    for cname, d in cells.items():
        a, b = min(d), max(d)
        gain = 100 * (d[b][0] - d[a][0])
        ax2.plot(centres, by_bin(gain, d[a][2]), marker="s", label=f"{cname} ({a} to {b})")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_title("frame-paired gain from full history", fontsize=9)
    ax2.set_ylabel("gain (pp)")
    for a in (ax, ax2):
        a.set_xlabel("range (m)")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"history_by_range_{split}.png", dpi=160)
    print(f"wrote {FIGURES / f'history_by_range_{split}.png'}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    raise SystemExit(main(a[0] if a else "test_2"))
