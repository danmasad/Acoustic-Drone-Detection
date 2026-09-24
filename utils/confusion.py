"""Confusion matrices at the 95%-recall operating point, for every detector we report.

AUROC and specificity are what the report quotes, and neither says in plain counts what an
operator would see. This fixes the threshold where 95% of drone frames fire — the operating
point every detection table in the notes already uses — and lays out the four cells.

Two sources, because only one of them has per-frame scores:

- **Map model presence head**: per-frame logits are in `results/preds/`, so the table is
  computed directly, with the threshold rule of `metrics.specificity_at_recall`.
- **Per-channel detector**: only summaries were committed. At a 95%-recall threshold the drone
  row is fixed by construction, specificity fixes the no-drone row, and the class counts come
  from known drone-frame counts on test 2, test 1 and fold C (which holds test 2's drone
  segment). Folds A and B have no committed counts, so theirs is estimated from every best-F1
  point that sits at recall 1.0 (see `prior_count`); checked against test 2's known count,
  that estimate is off by 0 to 2 frames, so the A and B tables are good to ~3 frames.

Gate: every direct table reproduces its dump's stored specificity; every reconstructed table
reproduces its stored specificity to one frame (plus the estimate's tolerance on folds A and
B); the prior estimate lands within tolerance of test 2's known count.

Test 1 rows come from dumps and summaries that already exist; nothing is re-scored.

Run: .venv/bin/python utils/confusion.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS, RUNS                         # noqa: E402
from src.metrics import specificity_at_recall                # noqa: E402

PREDS = RESULTS / "preds"
RECALL = 0.95
CELLS = [f"{a}_K{k}_w32_b16_wce" for a in ("all24", "device3", "best6") for k in (1, 5, 9)]
LOSO = ("clean_linear", "clean_log64", "confusers_linear", "confusers_log64")


def table(tp: int, fn: int, fp: int, tn: int) -> dict:
    n_pos, n_neg = tp + fn, fp + tn
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / n_pos, "specificity": tn / n_neg,
            "precision": tp / max(tp + fp, 1),
            "fire_rate_on_no_drone": fp / n_neg}


def from_scores(score: np.ndarray, has_drone: np.ndarray) -> dict:
    y = has_drone.astype(bool)
    thr = np.quantile(score[y], 1.0 - RECALL)
    tp, fn = int((score[y] >= thr).sum()), int((score[y] < thr).sum())
    tn, fp = int((score[~y] < thr).sum()), int((score[~y] >= thr).sum())
    return table(tp, fn, fp, tn) | {"threshold": float(thr)}


PRIOR_TOLERANCE = 3        # frames; measured against test 2's known count (0 and +2 frames)


def prior_count(*summaries: dict) -> int | None:
    """Drone-frame count estimated from summaries whose best-F1 point is at recall 1.0.

    Precision there is n_pos / (n_pos + negatives above the threshold), which can only exceed
    the prior, so precision x n is an upper bound; the smallest bound over the summaries is
    taken. None if no summary sits at recall 1.0.
    """
    bounds = [int(np.floor(m["f1_precision"] * m["n_frames"])) for m in summaries
              if abs(m["f1_recall"] - 1.0) < 1e-9]
    return min(bounds) if bounds else None


def from_summary(m: dict, n_pos: int) -> dict:
    """Rebuild the table from a detector summary and the split's drone-frame count."""
    n_neg = int(m["n_frames"]) - n_pos
    tp = int(np.ceil(RECALL * n_pos))
    tn = int(round(m["specificity_at_95_recall"] * n_neg))
    return table(tp, n_pos - tp, n_neg - tn, tn)


def mean_tables(ts: list[dict]) -> dict:
    keys = ("tp", "fn", "fp", "tn", "recall", "specificity", "precision",
            "fire_rate_on_no_drone")
    out = {k: float(np.mean([t[k] for t in ts])) for k in keys}
    out["specificity_sd"] = float(np.std([t["specificity"] for t in ts], ddof=1)) \
        if len(ts) > 1 else 0.0
    out["n_seeds"] = len(ts)
    return out


def map_model(split: str, failures: list) -> dict:
    by_cell = defaultdict(list)
    for f in sorted(PREDS.glob(f"*_s?_{split}_causal.npz")):
        cell = re.sub(r"_s\d_" + split + "_causal$", "", f.stem)
        if cell not in CELLS:
            continue
        d = np.load(f)
        t = from_scores(d["score"], d["has_drone"])
        stored = specificity_at_recall(d["score"], d["has_drone"])
        if abs(stored - t["specificity"]) > 1e-12:
            failures.append(f"{f.stem}: specificity {t['specificity']} vs {stored}")
        by_cell[cell].append(t)
    return {c: mean_tables(ts) for c, ts in by_cell.items()}


def detector(failures: list) -> dict:
    out = {}

    def check(name, m, t, frames=1):
        if abs(t["specificity"] - m["specificity_at_95_recall"]) > frames / (t["fp"] + t["tn"]):
            failures.append(f"{name}: rebuilt specificity {t['specificity']} "
                            f"vs {m['specificity_at_95_recall']}")

    # The K=1 baseline dump covers exactly the frames the K=1 detector was scored on.
    n_pos_t2 = int(np.load(PREDS / "baseline_all24_test_2.npz")["has_drone"].sum())
    full = []
    for s in range(3):
        p = RUNS / f"detector_K1_w32_s{s}_full" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())["test_2"]
            est = prior_count(m)
            if est is not None and not 0 <= est - n_pos_t2 <= PRIOR_TOLERANCE:
                failures.append(f"{p.parent.name}: prior estimate {est} vs known {n_pos_t2}")
            t = from_summary(m, n_pos_t2)
            check(p.parent.name, m, t)
            full.append(t)
    out["test_2_within_scene"] = mean_tables(full)

    # Fold C is the Mar 18 flight 2 session, whose drone segment is test 2's; A and B have no
    # committed counts, so they are estimated from every summary of that fold.
    fold_counts, estimated = {"C_Mar18_f2": n_pos_t2}, {}
    for fold in ("A_Nov25", "B_Mar18_f1"):
        sums = [v for variant in LOSO
                for v in (json.loads((RESULTS / f"detector_loso_{variant}.json").read_text())
                          ["folds"][fold][k] for k in ("metrics", "last_epoch"))]
        fold_counts[fold] = prior_count(*sums)
        estimated[fold] = True
    out["loso_fold_drone_frames"] = fold_counts | {
        "estimated": sorted(estimated), "tolerance_frames": PRIOR_TOLERANCE}

    for variant in LOSO:
        d = json.loads((RESULTS / f"detector_loso_{variant}.json").read_text())
        folds = {}
        for fold, v in d["folds"].items():
            t = from_summary(v["metrics"], fold_counts[fold])
            check(f"loso {variant} {fold}", v["metrics"], t,
                  frames=1 + PRIOR_TOLERANCE * (fold in estimated))
            folds[fold] = t
        out[f"loso_{variant}"] = folds | {"mean": mean_tables(list(folds.values()))}

    t1 = json.loads((RESULTS / "test_1.json").read_text())
    n_pos = int(t1["baseline"]["all24"]["n_drone"])
    t = from_summary(t1["detector"], n_pos)
    check("test_1 detector", t1["detector"], t)
    out["test_1"] = t
    return out


def show(name: str, t: dict) -> None:
    print(f"  {name:<37} TP {t['tp']:>7.0f}  FN {t['fn']:>5.0f}  FP {t['fp']:>6.0f}  "
          f"TN {t['tn']:>5.0f}   fires on {100 * t['fire_rate_on_no_drone']:5.1f}% of "
          f"no-drone frames   precision {t['precision']:.3f}")


def main() -> int:
    failures: list[str] = []
    out = {"recall": RECALL,
           "map_presence_head": {s: map_model(s, failures) for s in ("test_2", "test_1")},
           "per_channel_detector": detector(failures)}

    print(f"Confusion matrices at {RECALL:.0%} recall (mean over seeds where there are seeds)\n")
    for split, cells in out["map_presence_head"].items():
        print(f"map model presence head, {split}")
        for c, t in cells.items():
            show(c, t)
        print()
    det = out["per_channel_detector"]
    print("per-channel detector")
    show("test 2 (within-scene)", det["test_2_within_scene"])
    for variant in LOSO:
        show(f"LOSO {variant}, mean of folds", det[f"loso_{variant}"]["mean"])
    show("test 1", det["test_1"])

    (RESULTS / "confusion.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'confusion.json'}")
    if failures:
        print("GATE FAIL\n  " + "\n  ".join(failures))
        return 1
    print("GATE PASS: every table reproduces its stored specificity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
