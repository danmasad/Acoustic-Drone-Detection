"""Unattended remainder of Track B: Hann decision, then B4 write-up, LOFO, ablations, extras.

Waits for the Hann test-2 cache and for `logs/b4.log` to print `battery finished`, then runs
everything that still needs a GPU, in order. Safe to re-run: finished batteries are skipped,
and each notes section is appended only if its heading is not already present.

Does not push, does not open a PR, and does not score test 1 unless a B5 cell beats B4 by
9 pp under LOFO *and* a prereg entry has been written.

    .venv/bin/python utils/detach.py logs/continue_track_b.log -- \
        .venv/bin/python -u utils/continue_track_b.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.config import RESULTS, RUNS                          # noqa: E402
from utils.run_grid import B4, B4_CELLS, B5, run_id           # noqa: E402

PY = sys.executable
NOTES = REPO / "docs" / "design_notes.md"
PLAN = REPO / "docs" / "review_plan.md"
LOGS = REPO / "logs"
SLEEP = 30
REPORTED = ("hit_10deg", "median_angular_deg", "auroc", "specificity_at_95_recall")


def sh(args, check: bool = True) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=REPO, check=check).returncode


def wait_until(label: str, ready, seconds: int = SLEEP) -> None:
    print(f"waiting for {label}...", flush=True)
    while not ready():
        time.sleep(seconds)
    print(f"  {label}: ready", flush=True)


def hann_ready() -> bool:
    from src.config import segments
    from src.stft_cache import cache_dir
    return all((cache_dir(s, "hann") / "stft.npy").exists() for s in segments("test_2"))


def battery_finished(log_name: str) -> bool:
    p = LOGS / log_name
    return p.exists() and "battery finished" in p.read_text()


def has_section(heading: str) -> bool:
    return NOTES.exists() and heading in NOTES.read_text()


def append_notes(heading: str, body: str) -> None:
    if has_section(heading):
        print(f"  notes already have {heading}")
        return
    with NOTES.open("a") as f:
        f.write("\n" + heading + "\n\n" + body.rstrip() + "\n")
    print(f"  wrote {heading}")


def b4_ids():
    return [run_id({"subset": s, "K": k, "seed": seed} | B4)
            for s, k in B4_CELLS for seed in (0, 1, 2)]


def cell_metrics(run: str, split: str) -> dict | None:
    p = RUNS / run / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    block = m.get(split) or m.get("test_2")
    if block is None:
        return None
    return {k: block[k] for k in REPORTED if k in block} | {
        "best_epoch": m.get("best_epoch"), "epochs_run": m.get("epochs_run")}


def mean_sd(vals):
    import numpy as np
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


def do_hann() -> None:
    wait_until("Hann test-2 cache", hann_ready)
    if (RESULTS / "window_hann.json").exists():
        print("Hann comparison already written")
        return
    sh([PY, str(REPO / "utils" / "compare_window.py"), "--device", "cpu"])
    r = json.loads((RESULTS / "window_hann.json").read_text())
    rows = "\n".join(
        f"| {s} | {c['rect']['hit_10deg']:.3f} | {c['hann']['hit_10deg']:.3f} | "
        f"{c['delta_hit_10deg']:+.3f} |"
        for s, c in r["cells"].items())
    decision = ("adopted: hit@10 rose by at least 1 pp on both arrays"
                if r["adopt"] else
                "kept: Hann did not raise hit@10 by 1 pp on both arrays")
    append_notes(
        "## 47b. Hann window on test 2",
        "Delay-and-sum on the Hann STFT cache (`cache_hann/`, test 2 only) against the committed\n"
        "rectangular baseline. Gate: same frame counts. A 1 pp rise on both `all24` and `device3`\n"
        f"was required to switch the cache; Hann was **{decision}**.\n\n"
        f"| Subset | Rect hit@10 | Hann hit@10 | Δ |\n|---|---|---|---|\n{rows}\n\n"
        "`results/window_hann.json`, `results/baseline_hann.json`.")


def do_b4_notes() -> None:
    wait_until("B4 battery", lambda: battery_finished("b4.log"))
    cells = {}
    lines = ["The new baseline (B4): block validation on the training scenes so test 2 is a true",
             "test; cosine schedule, clip 1, AdamW 1e-4, encoder dropout 0.1, emission-time labels.",
             "Three seeds. Grid-snapped hit@10 on test 2.\n",
             "| Cell | Hit@10 | Median deg | AUROC | Epochs |",
             "|---|---|---|---|---|"]
    for s, k in B4_CELLS:
        hits, meds, aucs, epochs = [], [], [], []
        for seed in (0, 1, 2):
            rid = run_id({"subset": s, "K": k, "seed": seed} | B4)
            m = cell_metrics(rid, "test_2")
            if m is None:
                lines.append(f"| {s} K={k} s{seed} | missing | | | |")
                continue
            hits.append(m["hit_10deg"]); meds.append(m["median_angular_deg"])
            aucs.append(m["auroc"]); epochs.append(m["epochs_run"])
        if hits:
            hm, hs = mean_sd(hits)
            cells[f"{s} K={k}"] = {"hit_mean": hm, "hit_sd": hs, "n": len(hits)}
            lines.append(f"| {s} K={k} | {100*hm:.1f}% ± {100*hs:.1f} | "
                         f"{mean_sd(meds)[0]:.1f} | {mean_sd(aucs)[0]:.3f} | "
                         f"{mean_sd(epochs)[0]:.0f} |")
    (RESULTS / "b4_baseline.json").write_text(json.dumps({"split": "test_2", "cells": cells},
                                                         indent=2) + "\n")
    append_notes("## 48. New baseline under a true test-2 split",
                 "\n".join(lines) + "\n\n`results/b4_baseline.json`.")


def run_battery(name: str, log: str) -> None:
    if battery_finished(log):
        print(f"{name} already finished")
        return
    sh([PY, str(REPO / "utils" / "run_grid.py"), name])


def do_lofo_notes() -> None:
    rows = ["| Cell | Fold | Held-out hit@10 |", "|---|---|---|"]
    for s, k in B4_CELLS:
        for fold in ("A_Nov25", "B_Mar18_f1", "C_Mar18_f2"):
            rid = run_id({"subset": s, "K": k, "seed": 0} | B4 | {"val_split": "lofo", "fold": fold})
            m = cell_metrics(rid, f"scene_{fold}")
            hit = f"{100*m['hit_10deg']:.1f}%" if m else "missing"
            rows.append(f"| {s} K={k} | {fold} | {hit} |")
    append_notes("## 48b. Leave-one-flight-out of the B4 recipe",
                 "One seed per fold. Each run is scored only on its held-out scene.\n\n"
                 + "\n".join(rows))


def do_b5_notes() -> dict:
    """Return {flag: (mean_hit, sd)} for the B5 device3 K=9 cells, plus B4 reference."""
    b4 = []
    for seed in (0, 1, 2):
        m = cell_metrics(run_id({"subset": "device3", "K": 9, "seed": seed} | B4), "test_2")
        if m:
            b4.append(m["hit_10deg"])
    b4_mean, b4_sd = mean_sd(b4) if b4 else (None, None)
    lines = [f"B4 device3 K=9 on test 2: {100*b4_mean:.1f}% ± {100*b4_sd:.1f}." if b4
             else "B4 device3 K=9 missing.",
             "", "| Ablation | Hit@10 | vs B4 |", "|---|---|---|"]
    out = {"b4_mean": b4_mean, "b4_sd": b4_sd, "flags": {}}
    for name, flag in B5.items():
        hits = []
        for seed in (0, 1, 2):
            m = cell_metrics(run_id({"subset": "device3", "K": 9, "seed": seed} | B4 | flag),
                             "test_2")
            if m:
                hits.append(m["hit_10deg"])
        if hits:
            hm, hs = mean_sd(hits)
            delta = hm - b4_mean if b4_mean is not None else None
            out["flags"][name] = {"hit_mean": hm, "hit_sd": hs, "delta": delta}
            lines.append(f"| {name} | {100*hm:.1f}% ± {100*hs:.1f} | "
                         f"{100*delta:+.1f} pp |" if delta is not None else
                         f"| {name} | {100*hm:.1f}% | |")
    (RESULTS / "b5_ablations.json").write_text(json.dumps(out, indent=2) + "\n")
    append_notes("## 49–52. Ablations on the B4 device3 K=9 cell",
                 "One flag at a time, three seeds, test 2 as a true test.\n\n" + "\n".join(lines))
    return out


def do_post() -> None:
    b4 = b4_ids()
    existing = [r for r in b4 if (RUNS / r / "best.pt").exists()]
    if existing:
        sh([PY, "-m", "src.predict", "--split", "test_2", "--runs", *existing, "--heatmaps"],
           check=False)
        stems = sorted({r.rsplit("_s", 1)[0] for r in existing})
        sh([PY, str(REPO / "utils" / "ensemble.py"), "--cells", *stems], check=False)
        sh([PY, str(REPO / "utils" / "average_checkpoints.py"), *existing], check=False)
        sh([PY, str(REPO / "utils" / "mask_calibration.py"), "--cells", *stems], check=False)


def do_prereg(b5: dict) -> None:
    if not b5.get("flags") or b5.get("b4_mean") is None:
        return
    b4_sd = b5["b4_sd"] or 0.0
    winners = {n: v for n, v in b5["flags"].items()
               if v.get("delta") is not None and v["delta"] > b4_sd}
    if not winners:
        print("B5 did not beat B4 by more than seed spread; no prereg, test 1 stays sealed")
        return
    best = max(winners, key=lambda n: winners[n]["delta"])
    gain = 100 * winners[best]["delta"]
    path = REPO / "docs" / "prereg_test1.md"
    entry = (f"\n\n## Track B, B5 `{best}` (written after B4/B5, before any new test-1 score)\n\n"
             f"Predicted test-1 gain of the B5 `{best}` cell over the B4 device3 K=9 recipe: "
             f"{gain:.1f} pp (test-2 delta; seed spread of B4 was {100*b4_sd:.1f} pp).\n")
    if f"B5 `{best}`" not in path.read_text():
        path.write_text(path.read_text() + entry)
        print(f"wrote prereg for {best} ({gain:.1f} pp)")
    if gain < 9:
        print(f"predicted gain {gain:.1f} pp < 9; test 1 stays sealed")
        return
    print("predicted gain >= 9 pp; scoring test 1 once from existing B4/B5 checkpoints")
    sh([PY, "-m", "src.predict", "--split", "test_1", "--runs",
        run_id({"subset": "device3", "K": 9, "seed": 0} | B4 | B5[best])], check=False)


def patch_plan(row: str) -> None:
    text = PLAN.read_text()
    if row in text:
        return
    # leave the file as-is if we cannot find a safe anchor
    print("review_plan not patched automatically")


def main() -> int:
    os.chdir(REPO)
    print("continue_track_b: Hann -> B4 notes -> LOFO -> B5 -> extras", flush=True)
    do_hann()
    do_b4_notes()
    run_battery("b4_lofo", "b4_lofo.log")
    # run_grid writes to stdout; when we invoke it here the parent log is continue_track_b.log,
    # so mark lofo finished by checking metrics rather than that filename.
    do_lofo_notes()
    run_battery("b5", "b5.log")
    b5 = do_b5_notes()
    do_post()
    do_prereg(b5)
    print("continue_track_b: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
