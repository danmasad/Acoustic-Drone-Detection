"""Percent of drone frames inside 10°, by distance, on the frames each model was fit on.

The results-table models are the original recipe: they trained on the whole site-1
training split and stopped on test 2. This scores those checkpoints on that split.
Same-site (test 2) numbers are read from the dumps already on disk.

    .venv/bin/python utils/analysis/score_train_ranges.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import AZ, EL, pred_file, run_id, table_file  # noqa: E402
from src.geometry import angular_distance       # noqa: E402
from src.predict import baseline_predictions, cnn_predictions  # noqa: E402

BINS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180)]
MODELS = [
    ("Delay-and-sum, 24 mics", None, ["baseline_all24"]),
    ("U-Net, 24 mics, K = 1", [run_id("all24", 1, seed=s) for s in range(3)], None),
    ("U-Net, device 3, K = 1", [run_id("device3", 1, seed=s) for s in range(3)], None),
    ("U-Net, device 3, K = 5", [run_id("device3", 5, seed=s) for s in range(3)], None),
    ("U-Net, device 3, K = 9", [run_id("device3", 9, seed=s) for s in range(3)], None),
    ("U-Net, searched 6, K = 1", [run_id("best6", 1, seed=s) for s in range(3)], None),
]
OUT = table_file("train_range_hit.json")


def by_range(pred):
    m = pred["has_drone"].astype(bool) & np.isfinite(pred["az"])
    err = angular_distance(pred["az"][m], pred["el"][m],
                           AZ[pred["az_idx"][m]], EL[pred["el_idx"][m]])
    r = pred["range_m"][m]
    hits, counts = [], []
    for lo, hi in BINS:
        x = err[(r >= lo) & (r < hi)]
        counts.append(int(len(x)))
        hits.append(float(100 * np.mean(x <= 10)) if len(x) else float("nan"))
    counts.append(int(m.sum()))
    hits.append(float(100 * np.mean(err <= 10)))
    return hits, counts


def from_disk(name):
    d = np.load(pred_file(name), allow_pickle=False)
    return {k: d[k] for k in d.files}


def cell(runs, split):
    per, counts = [], None
    for run in runs:
        if split == "test_2":
            pred = from_disk(f"{run}_test_2" if run.startswith("baseline") else f"{run}_test_2_causal")
        elif run.startswith("baseline"):
            pred = baseline_predictions("train", subset="all24", verbose=True)
        else:
            # CPU: the K=9 upsample on MPS wedged in waitUntilCompleted after one seed.
            pred = cnn_predictions(run, "train", device="cpu", verbose=True)
        hits, counts = by_range(pred)
        per.append(hits)
        print(f"    {run}: " + "  ".join(f"{h:.1f}" for h in hits), flush=True)
    per = np.array(per)
    mu, sd = per.mean(0), (per.std(0, ddof=1) if len(per) > 1 else np.zeros(6))
    return {"mean": mu.round(1).tolist(), "sd": sd.round(1).tolist(), "n": counts}


def main() -> int:
    out = {"bins": [f"{lo}–{hi}" for lo, hi in BINS] + ["all"], "splits": {}}
    for split in ("test_2", "train"):
        print(f"\n== {split}", flush=True)
        rows = {}
        for name, nets, base in MODELS:
            print(name, flush=True)
            rows[name] = cell(base or nets, split)
        out["splits"][split] = rows
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2))
        print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
