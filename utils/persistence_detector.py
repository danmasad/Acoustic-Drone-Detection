"""Localizer persistence as a detection score: does a steady direction mean a drone?

Every learned detector here separates drone from no-drone by what the frame sounds like, and under
leave-one-scene-out that is worth nothing: the per-channel detector's fold-mean AUROC is 0.49
(notes 28), because it learns the training scenes' ambient rather than the drone (notes 38).
A drone has a property that ambient mostly does not: it is one point source that moves smoothly,
so a localizer pointed at it gives nearly the same direction frame after frame, while on diffuse
noise the peak wanders. That is geometry, not timbre, and it is the paper's own 3-frame stability
rule (`check_vs_paper.stability_flags`) made continuous.

Score: minus the RMS great-circle distance of the last n direction estimates from their mean
direction, causal, within one recording segment, for n = 3, 5, 9. No parameters, nothing fit, so
there is nothing to overfit to a scene. Frames before a segment's window fills are dropped.

Scored on test 2 for delay-and-sum (three arrays) and for the U-Net (four headline cells, three
seeds), as AUROC of drone frames against each no-drone scene separately and pooled, with the
presence logit (or DAS peak level) on the same frames as the learned-score reference. Drone
frames are also split by range, against all no-drone frames, since persistence should fail
exactly where the peak is too weak to hold still. Test 1 stays sealed: a new statistic on its
dumps would be a new look at it.

Gate: at n = 3 and 10 degrees the score reproduces `stability_flags` frame for frame.

Run: .venv/bin/python utils/persistence_detector.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import metrics, predict                              # noqa: E402
from src.config import RESULTS                                # noqa: E402
from check_vs_paper import STABILITY_DEG, STABILITY_WINDOW, stability_flags  # noqa: E402
from significance import time_order                          # noqa: E402

SPLIT = "test_2"
WINDOWS = (3, 5, 9)
SEEDS = (0, 1, 2)
RANGE_BINS = ((0, 50), (50, 100), (100, 200))
DAS = ("all24", "device3", "best6")
UNET = ("all24_K1", "device3_K1", "device3_K9", "best6_K1")
DETECTOR_LOSO = "clean_linear"


def unit(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], 1)


def spread(az: np.ndarray, el: np.ndarray, seg: np.ndarray, n: int) -> np.ndarray:
    """Causal RMS angular deviation (deg) of the last n directions; NaN until the window fills."""
    out = np.full(len(az), np.nan)
    v = unit(az, el)
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        if len(idx) < n:
            continue
        w = sliding_window_view(v[idx], n, axis=0)              # (m, 3, n)
        mean = w.mean(2)
        norm = np.linalg.norm(mean, axis=1, keepdims=True)
        mean = np.where(norm > 1e-9, mean / np.maximum(norm, 1e-12), 0.0)
        cos = np.clip(np.einsum("mc,mcn->mn", mean, w), -1.0, 1.0)
        rms = np.sqrt((np.degrees(np.arccos(cos)) ** 2).mean(1))
        # A window whose directions cancel exactly has no mean direction; stability_flags
        # counts it as valid and unstable, and so does this.
        rms[norm[:, 0] <= 1e-9] = np.inf
        out[idx[n - 1:]] = rms
    return out


def ordered(name: str) -> dict:
    d = predict.load(name)
    o = time_order(d)
    return {k: v[o] for k, v in d.items() if isinstance(v, np.ndarray) and v.shape[:1] == o.shape}


def aurocs(score: np.ndarray, d: dict, valid: np.ndarray, seg_names: list[str]) -> dict:
    y = d["has_drone"].astype(bool)
    out = {"pooled": metrics.auroc(score[valid], y[valid])}
    for s, name in enumerate(seg_names):
        neg = valid & ~y & (d["seg"] == s)
        if neg.any():
            sel = (valid & y) | neg
            out[f"vs {name}"] = metrics.auroc(score[sel], y[sel])
    for lo, hi in RANGE_BINS:
        pos = valid & y & (d["range_m"] >= lo) & (d["range_m"] < hi)
        sel = pos | (valid & ~y)
        out[f"drone {lo}-{hi} m"] = metrics.auroc(score[sel], y[sel]) if pos.any() else None
    return out


def score_dump(name: str) -> dict:
    d = ordered(name)
    seg_names = [str(s).replace("Mar_18_2025_f2_", "") for s in predict.load(name)["seg_ids"]]
    res = {}
    for n in WINDOWS:
        sp = spread(d["pred_az"], d["pred_el"], d["seg"], n)
        valid = ~np.isnan(sp)
        res[f"persistence_n{n}"] = aurocs(-sp, d, valid, seg_names)
    # The learned (or, for DAS, level-based) score on exactly the frames the widest window keeps.
    valid = ~np.isnan(spread(d["pred_az"], d["pred_el"], d["seg"], max(WINDOWS)))
    res["reference_score"] = aurocs(d["score"].astype(float), d, valid, seg_names)
    return res


def gate() -> list[str]:
    d = ordered(f"baseline_all24_{SPLIT}")
    failures = []
    for s in np.unique(d["seg"]):
        m = d["seg"] == s
        ref = stability_flags(d["pred_az"][m], d["pred_el"][m])
        mine = spread(d["pred_az"][m], d["pred_el"][m], d["seg"][m], STABILITY_WINDOW)
        got = np.nan_to_num(mine, nan=np.inf) <= STABILITY_DEG
        if not np.array_equal(ref, got):
            failures.append(f"segment {s}: {int((ref != got).sum())} frames disagree")
    return failures


def mean_sd(runs: list[dict]) -> dict:
    out = {}
    for key in runs[0]:
        out[key] = {}
        for k in runs[0][key]:
            vals = [r[key][k] for r in runs if r[key][k] is not None]
            out[key][k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                           "per_seed": vals} if vals else None
    return out


def show(label: str, res: dict, cols: list[str]) -> None:
    for key in [f"persistence_n{n}" for n in WINDOWS] + ["reference_score"]:
        cells = []
        for c in cols:
            v = res[key][c]
            if isinstance(v, dict):
                v = v["mean"] if v else None
            cells.append("   -  " if v is None else f"{v:6.3f}")
        print(f"  {label:<14}{key:<17}" + "".join(f"{c:>10}" for c in cells))


def main() -> int:
    print(__doc__.split("\n\n")[0])
    failures = gate()

    out = {"split": SPLIT, "windows": list(WINDOWS), "das": {}, "unet": {}}
    for arr in DAS:
        out["das"][arr] = score_dump(f"baseline_{arr}_{SPLIT}")
    for cell in UNET:
        runs = [score_dump(f"{cell}_w32_b16_wce_s{s}_{SPLIT}_causal") for s in SEEDS
                if (predict.PREDS / f"{cell}_w32_b16_wce_s{s}_{SPLIT}_causal.npz").exists()]
        out["unet"][cell] = mean_sd(runs) | {"seeds": len(runs)}

    loso = json.loads((RESULTS / f"detector_loso_{DETECTOR_LOSO}.json").read_text())
    out["detector_loso_reference"] = {
        "variant": DETECTOR_LOSO,
        "folds": {f: v["metrics"]["auroc"] for f, v in loso["folds"].items()},
        "mean": float(np.mean([v["metrics"]["auroc"] for v in loso["folds"].values()]))}

    cols = ["pooled", "vs no_drone_2", "vs pre_takeoff",
            "drone 0-50 m", "drone 50-100 m", "drone 100-200 m"]
    print("\nAUROC on test 2 (U-Net: mean of seeds). Reference = presence logit / DAS peak level.")
    print(f"  {'':<31}" + "".join(f"{c.replace('drone ', ''):>10}" for c in cols))
    for arr, res in out["das"].items():
        show(f"DAS {arr}", res, cols)
    for cell, res in out["unet"].items():
        show(f"U-Net {cell}", res, cols)
    print(f"\n  per-channel detector, leave-one-scene-out ({DETECTOR_LOSO}): fold mean "
          f"{out['detector_loso_reference']['mean']:.3f}")

    path = RESULTS / "persistence_detector.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: n=3 at 10 deg reproduces stability_flags" if not failures else "GATE FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
