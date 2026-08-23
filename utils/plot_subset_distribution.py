"""Where do our chosen microphone subsets sit among ALL possible subsets?

Selecting the best of 106,183 arrangements invites the objection that the comparison was
rigged by search. Plotting the whole distribution answers it in the open: it shows how much
spread is actually reachable, and how unusual the deployable choice (device 3) really is.

Reads results/subset_search.npz, produced by utils/search_best_dense.py, so nothing is
recomputed and every subset is on one scale: the 95th percentile of the false-peak level
across 250 source directions.

Run: .venv/bin/python utils/search_best_dense.py     # once, ~35 min
     .venv/bin/python utils/plot_subset_distribution.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
STORE = REPO / "results/subset_search.npz"

# label -> (colour, size of the subset it belongs to)
MARKS = {
    "all 24":           ("black",      24),
    "device 3":         ("tab:blue",    6),
    "two-stage best-6": ("tab:green",   6),
    "symmetric-6":      ("tab:orange",  6),
    "greedy-spread 6":  ("tab:red",     6),
}


def main():
    if not STORE.exists():
        sys.exit(f"missing {STORE}; run utils/search_best_dense.py first")
    z = np.load(STORE)
    n_dirs = int(z["n_dirs"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, n in zip(axes, (6, 4)):
        p95 = z[f"p95_{n}"]
        combos = z[f"combos_{n}"]
        ax.hist(p95, bins=80, color="0.75", edgecolor="none")
        ax.set(xlabel="false peak, 95th percentile over directions (dB, lower is better)",
               ylabel="number of subsets",
               title=f"All {len(p95):,} non-coplanar {n}-microphone subsets")

        winner = int(np.argmin(p95))
        lines = [(f"best of {len(p95):,}: {combos[winner].tolist()}",
                  float(p95[winner]), "tab:purple")]
        for label, (colour, size) in MARKS.items():
            key = f"named_{label}"
            if key in z and size == n:
                lines.append((label, float(z[key][0]), colour))

        # Coincident marks would overprint, so merge anything within 0.05 dB.
        merged = {}
        for label, v, colour in lines:
            hit = next((k for k in merged if abs(k - v) < 0.05), None)
            if hit is None:
                merged[v] = [[label], colour]
            else:
                merged[hit][0].append(label)

        for v, (labels, colour) in sorted(merged.items()):
            better = 100.0 * (p95 < v).mean()
            rank = f"top {better:.1f}%" if better < 50 else f"bottom {100 - better:.0f}%"
            ax.axvline(v, color=colour, lw=2,
                       label=f"{' = '.join(labels)}   {v:.1f} dB, {rank}")
            print(f"n={n}  {' = '.join(labels):<44}{v:>6.1f} dB   {rank}")

        if n == 6:
            ax.axvline(float(z["named_all 24"][0]), color="black", lw=2, ls="--",
                       label=f"all 24 (reference)   {float(z['named_all 24'][0]):.1f} dB")
        else:
            ax.annotate(f"all 24 sits at {float(z['named_all 24'][0]):.1f} dB,\n"
                        "far off this scale", xy=(0.02, 0.82), xycoords="axes fraction",
                        fontsize=8, color="0.35")
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        print(f"n={n}  population: best {p95.min():.1f}, median {np.median(p95):.1f}, "
              f"worst {p95.max():.1f} dB   (spread {p95.max() - p95.min():.1f} dB)\n")

    fig.suptitle(f"Every possible thinned array, scored on {n_dirs} source directions. "
                 "A narrow spread at n=4 means there is nothing to choose between them.")
    fig.tight_layout()
    out = REPO / "figures/subset_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
