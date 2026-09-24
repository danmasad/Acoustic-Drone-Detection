"""Reliability diagram of the map model's presence head on test 2.

The notes report presence as AUROC and specificity, which only use the ranking of the logits.
Whether sigmoid(logit) can be read as a probability is a separate question, and it decides
whether a fixed 0.5 threshold means anything and whether a probability-level loss change
(Phase 3 of the review) has something to fix.

For every headline cell the three seeds are pooled, the sigmoid of the presence logit is binned
into ten equal-width bins, and each bin's mean predicted probability is set against the drone
frequency in it. Expected calibration error (ECE, frame-weighted |confidence - frequency|) and
the Brier score summarize it, alongside the base rate a constant predictor would score.

Mask-pixel calibration needs the heatmaps, which are not dumped; it is `utils/mask_calibration.py`.

Gate: every cell's bin counts sum to its frame count and every bin's mean confidence lies
inside the bin.

Run: .venv/bin/python utils/calibration.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES, RESULTS                      # noqa: E402

PREDS = RESULTS / "preds"
SPLIT = "test_2"
ARRAYS = ("all24", "device3", "best6")
KS = (1, 5, 9)
EDGES = np.linspace(0.0, 1.0, 11)
MIN_PLOTTED = 50           # frames; sparser bins are counted in ECE but not drawn


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -x.astype(np.float64)))


def reliability(p: np.ndarray, y: np.ndarray) -> dict:
    idx = np.clip(np.digitize(p, EDGES[1:-1]), 0, len(EDGES) - 2)
    bins = []
    for b in range(len(EDGES) - 1):
        sel = idx == b
        n = int(sel.sum())
        bins.append({"lo": float(EDGES[b]), "hi": float(EDGES[b + 1]), "n": n,
                     "confidence": float(p[sel].mean()) if n else None,
                     "frequency": float(y[sel].mean()) if n else None})
    ece = sum(b["n"] * abs(b["confidence"] - b["frequency"]) for b in bins if b["n"]) / len(p)
    prior = float(y.mean())
    return {"n_frames": int(len(p)), "drone_rate": prior, "ece": float(ece),
            "brier": float(np.mean((p - y) ** 2)), "brier_constant_prior": prior * (1 - prior),
            "mean_confidence": float(p.mean()),
            "frac_above_0.99": float(np.mean(p > 0.99)),
            "frac_no_drone_above_0.5": float(np.mean(p[~y.astype(bool)] > 0.5)),
            "bins": bins}


def main() -> int:
    print(__doc__.split("\n\n")[0])
    out, failures = {}, []
    for arr in ARRAYS:
        for k in KS:
            cell = f"{arr}_K{k}_w32_b16_wce"
            files = sorted(PREDS.glob(f"{cell}_s?_{SPLIT}_causal.npz"))
            if not files:
                continue
            ds = [np.load(f) for f in files]
            p = np.concatenate([sigmoid(d["score"]) for d in ds])
            y = np.concatenate([d["has_drone"] for d in ds]).astype(np.float64)
            r = reliability(p, y) | {"seeds": len(files)}
            if sum(b["n"] for b in r["bins"]) != len(p):
                failures.append(f"{cell}: bin counts do not sum to {len(p)}")
            for b in r["bins"]:
                if b["n"] and not b["lo"] - 1e-12 <= b["confidence"] <= b["hi"] + 1e-12:
                    failures.append(f"{cell}: bin {b['lo']:.1f} confidence outside the bin")
            out[cell] = r
            print(f"  {cell:<24} ECE {r['ece']:.3f}  Brier {r['brier']:.3f} "
                  f"(constant {r['brier_constant_prior']:.3f})  "
                  f"p>0.99 on {100 * r['frac_above_0.99']:.1f}% of frames, "
                  f"p>0.5 on {100 * r['frac_no_drone_above_0.5']:.1f}% of no-drone frames")

    path = RESULTS / "calibration_presence.json"
    path.write_text(json.dumps({"split": SPLIT, "cells": out}, indent=2))
    print(f"wrote {path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(ARRAYS), figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": (3, 1)})
    centres = (EDGES[:-1] + EDGES[1:]) / 2
    for col, arr in enumerate(ARRAYS):
        top, bottom = axes[0, col], axes[1, col]
        top.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
        for k in KS:
            r = out.get(f"{arr}_K{k}_w32_b16_wce")
            if r is None:
                continue
            conf = [b["confidence"] for b in r["bins"] if b["n"] >= MIN_PLOTTED]
            freq = [b["frequency"] for b in r["bins"] if b["n"] >= MIN_PLOTTED]
            top.plot(conf, freq, marker="o", label=f"K={k}  ECE {r['ece']:.2f}")
            share = np.array([b["n"] for b in r["bins"]]) / r["n_frames"]
            bottom.bar(centres + (k - 5) * 0.012, share, width=0.024, label=f"K={k}")
        top.axhline(out[next(c for c in out if c.startswith(arr))]["drone_rate"],
                    color="0.3", lw=0.8, ls=":")
        top.set_title(arr)
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.legend(loc="upper left", fontsize=8)
        bottom.set_yscale("log")
        bottom.set_xlabel("predicted presence probability")
    axes[0, 0].set_ylabel("drone frequency")
    axes[1, 0].set_ylabel("share of frames")
    fig.suptitle(f"Presence head reliability, {SPLIT} (3 seeds pooled; bins under {MIN_PLOTTED} "
                 "frames not drawn; dotted line = drone rate)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig_path = FIGURES / f"calibration_presence_{SPLIT}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")

    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: bins are consistent" if not failures else "GATE FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
