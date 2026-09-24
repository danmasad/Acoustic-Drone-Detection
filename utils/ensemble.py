"""Seed ensembles: average the members' mask probabilities, then decode once.

Every cell of the grid is trained with three seeds and the report quotes their mean, but a
mean of three decoded answers is not what an ensemble would do. Averaging the per-pixel
sigmoids before the decoder lets the members vote on the map itself: a phantom that one seed
paints bright and the other two leave dark drops below the centroid's 0.5 threshold, where
averaging the decoded directions would have dragged the answer a third of the way towards it.

Presence is combined as the mean *logit*, not the mean probability. float32 sigmoid saturates
to exactly 1.0 above a logit of about 17, and ties at 1.0 once turned a working head into an
AUROC of 0.5000; the mean logit ranks frames the same way a geometric mean of odds would.

Needs each member's full mask logits, which only `python -m src.predict --heatmaps` writes (to
`cache/heatmaps/`, 128 MB a run on test 2), so this runs wherever the STFT cache does. Only the
rectangular grid is supported; the polar disc has its own decoder inputs.

Gate: an "ensemble" of one member reproduces that member's own dump — the centroid direction
to float32 precision, the presence score exactly, and the grid-snapped argmax on at least
99.9% of frames. The grid index is the argmax of float64 probabilities here and of logits in
the dump, which can only disagree on a tie that float64 sigmoid saturation creates.

Run: .venv/bin/python -m src.predict --runs <members> --heatmaps
     .venv/bin/python utils/ensemble.py [--cells all24_K1_w32_b16_wce ...]
     .venv/bin/python utils/ensemble.py --self-test      # no data needed
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics, predict                              # noqa: E402
from src.config import RESULTS                                # noqa: E402
from src.models import decode_mask, decode_probs              # noqa: E402

CELLS = ("all24_K1_w32_b16_wce", "device3_K1_w32_b16_wce", "device3_K9_w32_b16_wce")
SEEDS = (0, 1, 2)
SPLIT = "test_2"
ORDER = "causal"
BATCH = 512
GRID_AGREEMENT = 0.999
REPORTED = ("auroc", "specificity_at_95_recall", "hit_10deg", "median_angular_deg")


def combine(heatmaps: list[np.ndarray], scores: list[np.ndarray]) -> dict:
    """Members' mask logits (N, n_el, n_az) and presence logits (N,) -> one set of dump
    columns. With a single member this is the member's own decode."""
    n, n_el, n_az = heatmaps[0].shape
    out = {"score": np.mean(np.stack(scores), axis=0, dtype=np.float64).astype(np.float32),
           "el_idx": np.empty(n, np.int64), "az_idx": np.empty(n, np.int64),
           "pred_az": np.empty(n, np.float32), "pred_el": np.empty(n, np.float32)}
    for lo in range(0, n, BATCH):
        sl = slice(lo, min(lo + BATCH, n))
        # float32 sigmoid of float32 logits, so that one member goes through exactly the
        # operations `decode_mask` applied when the dump was written.
        p = torch.stack([torch.sigmoid(torch.from_numpy(h[sl])) for h in heatmaps]).mean(0)
        az, el = decode_probs(p, mode="centroid")
        out["pred_az"][sl], out["pred_el"][sl] = az, el
        p64 = torch.stack([torch.sigmoid(torch.from_numpy(h[sl]).double())
                           for h in heatmaps]).mean(0)
        k = p64.flatten(1).argmax(1).numpy()
        out["el_idx"][sl], out["az_idx"][sl] = k // n_az, k % n_az
    return out


def load_member(run_id: str, split: str) -> tuple[dict, np.ndarray]:
    name = f"{run_id}_{split}_{ORDER}"
    d = predict.load(name)
    if d["meta"].get("polar"):
        raise SystemExit(f"{run_id}: polar runs are not supported")
    path = predict.heatmap_path(name)
    if not path.exists():
        raise SystemExit(f"{path} missing: run python -m src.predict --runs {run_id} "
                         f"--split {split} --heatmaps")
    h = np.load(path, mmap_mode="r")
    if len(h) != len(d["score"]):
        raise SystemExit(f"{name}: {len(h)} heatmaps for {len(d['score'])} frames")
    return d, h


def scores_of(cols: dict, ref: dict) -> dict:
    grid = metrics.summarize(cols["score"], ref["has_drone"], cols["el_idx"], cols["az_idx"],
                             ref["az"], ref["el"], ref["range_m"])
    cent = metrics.summarize(cols["score"], ref["has_drone"], cols["el_idx"], cols["az_idx"],
                             ref["az"], ref["el"], ref["range_m"],
                             pred_az=cols["pred_az"], pred_el=cols["pred_el"])
    out = {k: grid[k] for k in REPORTED}
    out["centroid_hit_10deg"] = cent["hit_10deg"]
    out["centroid_median_angular_deg"] = cent["median_angular_deg"]
    return out


def reproduces(cols: dict, d: dict) -> list[str]:
    bad = []
    if not np.array_equal(cols["score"], d["score"].astype(np.float32)):
        bad.append("presence score differs")
    for key in ("pred_az", "pred_el"):
        diff = np.abs(((cols[key] - d[key] + 180) % 360) - 180) if key == "pred_az" \
            else np.abs(cols[key] - d[key])
        if diff.max() > 1e-4:
            bad.append(f"{key} differs by up to {diff.max():.2e} deg")
    agree = np.mean((cols["el_idx"] == d["el_idx"]) & (cols["az_idx"] == d["az_idx"]))
    if agree < GRID_AGREEMENT:
        bad.append(f"grid index agrees on {agree:.5f} of frames")
    return bad


def self_test() -> int:
    """combine() on synthetic maps: one member must match decode_mask, and three members
    that agree must match any one of them."""
    rng = np.random.default_rng(0)
    n, n_el, n_az = 300, 23, 91
    logits = rng.normal(-4.0, 1.0, (n, n_el, n_az)).astype(np.float32)
    e, a = rng.integers(2, n_el - 2, n), rng.integers(0, n_az, n)
    for i in range(n):
        for de in (-1, 0, 1):
            for da in (-1, 0, 1):
                logits[i, e[i] + de, (a[i] + da) % n_az] += 8.0 + rng.normal()
    scores = rng.normal(size=n).astype(np.float32)
    t = torch.from_numpy(logits)
    ref_az, ref_el = decode_mask(t, mode="centroid")
    k = t.flatten(1).argmax(1).numpy()
    ref = {"score": scores, "pred_az": ref_az, "pred_el": ref_el,
           "el_idx": k // n_az, "az_idx": k % n_az}
    bad = reproduces(combine([logits], [scores]), ref)
    bad += [f"three identical members: {b}"
            for b in reproduces(combine([logits] * 3, [scores] * 3), ref)]
    print("GATE " + ("FAIL" if bad else "PASS") + " (self-test on synthetic maps)")
    for b in bad:
        print(f"  {b}")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cells", nargs="*", default=list(CELLS),
                   help="run ids without the _s<seed> suffix")
    p.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    p.add_argument("--split", default=SPLIT)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()

    report, failures = {}, []
    for cell in a.cells:
        members = [f"{cell}_s{s}" for s in a.seeds]
        loaded = [load_member(r, a.split) for r in members]
        ref = loaded[0][0]
        for r, (d, _) in zip(members[1:], loaded[1:]):
            if not (np.array_equal(d["seg"], ref["seg"])
                    and np.array_equal(d["frame"], ref["frame"])):
                raise SystemExit(f"{r}: frames are not in the same order as {members[0]}")

        singles = {}
        for r, (d, h) in zip(members, loaded):
            one = combine([h], [d["score"]])
            failures += [f"{r}: {b}" for b in reproduces(one, d)]
            singles[r] = scores_of(one, ref)
        cols = combine([h for _, h in loaded], [d["score"] for d, _ in loaded])
        ens = scores_of(cols, ref)
        mean = {k: float(np.mean([s[k] for s in singles.values()])) for k in ens}
        report[cell] = {"members": members, "ensemble": ens, "member_mean": mean,
                        "members_scored": singles}

        name = f"{cell}_ens{len(members)}_{a.split}_{ORDER}"
        dump = {k: ref[k] for k in predict.COMMON if k not in cols} | cols
        predict._save(name, dump, list(ref["seg_ids"]),
                      {"kind": "ensemble", "members": members, "split": a.split,
                       "order": ORDER, "K": ref["meta"]["K"], "subset": ref["meta"]["subset"]})

        print(f"\n{cell}, {len(members)} seeds on {a.split}")
        print(f"  {'':28s} {'member mean':>12s} {'ensemble':>10s}")
        for k in ens:
            print(f"  {k:28s} {mean[k]:12.4f} {ens[k]:10.4f}")

    (RESULTS / "ensemble.json").write_text(json.dumps(
        {"split": a.split, "order": ORDER, "cells": report}, indent=2) + "\n")
    print("\nGATE " + ("FAIL" if failures else "PASS"))
    for f in failures:
        print(f"  {f}")
    if not failures:
        print("  every member, ensembled alone, reproduces its own dump")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
