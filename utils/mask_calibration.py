"""Reliability of the map model's per-pixel mask probabilities on test 2 (completes A2).

The centroid decoder thresholds the sigmoid at exactly 0.5 before it erodes and averages, so
whether a pixel at 0.5 is lit half the time is not a nicety: if the mask is over-confident the
activated region is too wide and the centroid is dragged by the fringe, and if it is
under-confident the threshold empties the map and the decoder falls back to the peak.

Targets are rebuilt from each dump's labelled direction with `labels.target_mask` at the
training radius, so they are the dataset's labels, not a lagged copy. Probabilities come from
the full mask logits `python -m src.predict --heatmaps` writes to `cache/heatmaps/`. Two views
per cell: the three seeds pooled pixel by pixel, and the seed ensemble (their mean
probability, which is what `utils/ensemble.py` decodes). Ten equal-width bins as in the
presence diagram, with ECE and Brier, on every frame and on drone frames alone.

Gate: every view's bin counts sum to frames x pixels and every bin's mean confidence lies
inside the bin. `--self-test` also checks that labels drawn from the probabilities themselves
score an ECE under 0.01.

Run: .venv/bin/python utils/mask_calibration.py [--cells ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import predict                                       # noqa: E402
from src.config import ANGULAR_THRESHOLD, FIGURES, RESULTS    # noqa: E402
from src.labels import target_mask                            # noqa: E402

CELLS = ("all24_K1_w32_b16_wce", "device3_K1_w32_b16_wce", "device3_K9_w32_b16_wce")
SEEDS = (0, 1, 2)
SPLIT = "test_2"
EDGES = np.linspace(0.0, 1.0, 11)
CHUNK = 1024
MIN_PLOTTED = 1000          # pixels


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -x.astype(np.float64)))


class Bins:
    """Streaming reliability: per-bin pixel count, summed confidence, summed target."""

    def __init__(self):
        n = len(EDGES) - 1
        self.n, self.p, self.y = np.zeros(n), np.zeros(n), np.zeros(n)
        self.sq = 0.0

    def add(self, p: np.ndarray, y: np.ndarray) -> None:
        p, y = p.ravel(), y.ravel().astype(np.float64)
        idx = np.clip(np.digitize(p, EDGES[1:-1]), 0, len(EDGES) - 2)
        m = len(EDGES) - 1
        self.n += np.bincount(idx, minlength=m)
        self.p += np.bincount(idx, weights=p, minlength=m)
        self.y += np.bincount(idx, weights=y, minlength=m)
        self.sq += float(np.sum((p - y) ** 2))

    def summary(self) -> dict:
        total = self.n.sum()
        bins = [{"lo": float(EDGES[b]), "hi": float(EDGES[b + 1]), "n": int(self.n[b]),
                 "confidence": float(self.p[b] / self.n[b]) if self.n[b] else None,
                 "frequency": float(self.y[b] / self.n[b]) if self.n[b] else None}
                for b in range(len(EDGES) - 1)]
        ece = sum(b["n"] * abs(b["confidence"] - b["frequency"]) for b in bins if b["n"]) / total
        rate = float(self.y.sum() / total)
        return {"n_pixels": int(total), "positive_rate": rate, "ece": float(ece),
                "brier": self.sq / total, "brier_constant_prior": rate * (1 - rate),
                "mean_confidence": float(self.p.sum() / total), "bins": bins}


def check(name: str, r: dict, expected: int) -> list[str]:
    bad = []
    if r["n_pixels"] != expected:
        bad.append(f"{name}: {r['n_pixels']} pixels binned, expected {expected}")
    for b in r["bins"]:
        if b["n"] and not b["lo"] - 1e-12 <= b["confidence"] <= b["hi"] + 1e-12:
            bad.append(f"{name}: bin {b['lo']:.1f} confidence outside the bin")
    return bad


def calibrate(cell: str, seeds, split: str) -> tuple[dict, list[str]]:
    members = [f"{cell}_s{s}_{split}_causal" for s in seeds]
    dumps = [predict.load(m) for m in members]
    for m in members:
        if not predict.heatmap_path(m).exists():
            raise SystemExit(f"{predict.heatmap_path(m)} missing: run python -m src.predict "
                             f"--runs {m.rsplit('_' + split, 1)[0]} --split {split} --heatmaps")
    maps = [np.load(predict.heatmap_path(m), mmap_mode="r") for m in members]
    ref = dumps[0]
    for m, d in zip(members[1:], dumps[1:]):
        if not (np.array_equal(d["seg"], ref["seg"]) and np.array_equal(d["frame"], ref["frame"])):
            raise SystemExit(f"{m}: frames are not in the same order as {members[0]}")

    views = {v: Bins() for v in ("pooled", "pooled_drone", "ensemble", "ensemble_drone")}
    n = len(ref["score"])
    drone = ref["has_drone"].astype(bool)
    for lo in range(0, n, CHUNK):
        sl = slice(lo, min(lo + CHUNK, n))
        y = target_mask(ref["az"][sl], ref["el"][sl], ANGULAR_THRESHOLD)
        ps = [sigmoid(np.asarray(h[sl])) for h in maps]
        d = drone[sl]
        for p in ps:
            views["pooled"].add(p, y)
            views["pooled_drone"].add(p[d], y[d])
        mean = np.mean(ps, axis=0)
        views["ensemble"].add(mean, y)
        views["ensemble_drone"].add(mean[d], y[d])

    pixels = int(np.prod(maps[0].shape[1:]))
    expected = {"pooled": n * pixels * len(maps), "pooled_drone": int(drone.sum()) * pixels
                * len(maps), "ensemble": n * pixels, "ensemble_drone": int(drone.sum()) * pixels}
    out, bad = {"members": members}, []
    for v, b in views.items():
        out[v] = b.summary()
        bad += check(f"{cell} {v}", out[v], expected[v])
    return out, bad


def self_test() -> int:
    rng = np.random.default_rng(0)
    b = Bins()
    p = rng.beta(0.3, 3.0, 2_000_000)
    b.add(p, rng.random(p.shape) < p)
    r = b.summary()
    bad = check("self-test", r, len(p))
    if r["ece"] >= 0.01:
        bad.append(f"self-test: calibrated labels score ECE {r['ece']:.4f}")
    print("GATE " + ("FAIL" if bad else "PASS") + f" (self-test, ECE {r['ece']:.4f})")
    for x in bad:
        print(f"  {x}")
    return 1 if bad else 0


def plot(report: dict, split: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = list(report)
    fig, axes = plt.subplots(1, len(cells), figsize=(3.8 * len(cells), 3.8), squeeze=False)
    for ax, cell in zip(axes[0], cells):
        ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1)
        for v, style in (("pooled", "-o"), ("ensemble", "-s"), ("pooled_drone", ":o")):
            r = report[cell][v]
            pts = [(b["confidence"], b["frequency"]) for b in r["bins"] if b["n"] >= MIN_PLOTTED]
            if pts:
                ax.plot(*zip(*pts), style, ms=4, label=f"{v.replace('_', ', ')}  "
                                                       f"ECE {r['ece']:.3f}")
        ax.axvline(0.5, color="0.3", lw=0.8, ls=":")
        ax.set_title(cell.replace("_w32_b16_wce", ""), fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted pixel probability")
        ax.legend(loc="upper left", fontsize=7)
    axes[0, 0].set_ylabel("fraction of pixels inside the target")
    fig.suptitle(f"Mask-pixel reliability, {split} (dotted line: the centroid's 0.5 threshold)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"calibration_mask_{split}.png"
    fig.savefig(path, dpi=150)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cells", nargs="*", default=list(CELLS))
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--split", default=SPLIT)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()

    report, failures = {}, []
    for cell in a.cells:
        r, bad = calibrate(cell, a.seeds, a.split)
        report[cell], failures = r, failures + bad
        for v in ("pooled", "ensemble", "pooled_drone"):
            s = r[v]
            print(f"  {cell:<24} {v:<13} ECE {s['ece']:.4f}  Brier {s['brier']:.4f} "
                  f"(constant {s['brier_constant_prior']:.4f})  positives "
                  f"{100 * s['positive_rate']:.2f}%, mean confidence "
                  f"{100 * s['mean_confidence']:.2f}%")

    path = RESULTS / "calibration_mask.json"
    path.write_text(json.dumps({"split": a.split, "radius_deg": ANGULAR_THRESHOLD,
                                "cells": report}, indent=2) + "\n")
    print(f"wrote {path}\nwrote {plot(report, a.split)}")
    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: bins are consistent" if not failures else "GATE FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
