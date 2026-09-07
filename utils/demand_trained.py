"""Does training on real ambient fix the false-alarm rate on ambient it has not heard?

The detector's problem is not subtle. At the threshold that catches 95% of drones it fires on
69% of DEMAND frames, 24,881 alarms per hour, and backing off to 50% recall still leaves 6,733.
We only ever *measured* on DEMAND, never trained on it, and the obvious objection is that a
model whose entire negative class is 5.6 minutes of one field's ambient has never been given the
chance to learn what "not a drone" sounds like.

This is the honest version of giving it that chance. DEMAND's 18 scenes come in 6 groups of 3
(domestic, nature, office, public, street, transport). Each fold trains on 5 groups and reports
the false-alarm rate on the 3 scenes of the group held out, so the question is never "can it fit
ambient it has seen" but "does hearing 75 minutes of real ambient generalize to a soundscape it
has not". Group-level holdout rather than scene-level, because scenes inside a group share a
recording environment and a scene-level split would answer the easier question.

Two numbers per fold, and the pair is the result:

- **held-out** false alarms per hour, the number that matters
- **trained-on** false alarms per hour, which should collapse toward zero. If it does not, the
  model cannot even fit real ambient, which is a different and worse finding than failing to
  generalize.

The threshold is never chosen here. Each fold's own 95%-recall threshold comes from its drone
frames, as everywhere else in this project, and `utils/demand_false_alarms.py` does the scoring
unchanged, so a fold's rate is directly comparable with the 24,881 baseline.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import demand                                              # noqa: E402
from src.config import RESULTS, RUNS                                # noqa: E402
from src.detector import train                                      # noqa: E402

PY = str(Path(__file__).resolve().parent.parent / ".venv/bin/python")
BASELINE_RUN = "detector_K1_w32_s0_full"     # ambient-free, the 24,881/h model

# Gate 2, declared before the runs: an improvement has to be an order of magnitude on the
# held-out groups and must not cost the drone side more than 0.02 AUROC on test 2.
GATE_PER_HOUR = 5000.0
GATE_AUROC_DROP = 0.02


def fold_runs(frac: float, epochs: int, seed: int, verbose: bool = True) -> list[dict]:
    """Train one detector per held-out group, skipping folds already on disk."""
    out = []
    for held in demand.GROUPS:
        groups = [g for g in demand.GROUPS if g != held]
        run_id = f"detector_K1_w32_amb_s{seed}_full_no{held}"
        if (RUNS / run_id / "metrics.json").exists():
            print(f"skip {run_id}, already trained", flush=True)
        else:
            print(f"\n=== holding out {held} ({', '.join(demand.GROUPS[held])}) ===",
                  flush=True)
            train(K=1, seed=seed, epochs=epochs, tag=f"full_no{held}",
                  ambient_groups=groups, ambient_frac=frac, verbose=verbose)
        out.append({"held_out": held, "run_id": run_id})
    return out


def false_alarms(run_id: str) -> dict:
    """Per-scene false alarms for one run, via the unchanged evaluation script."""
    path = RESULTS / f"demand_false_alarms_{run_id}.json"
    if not path.exists():
        subprocess.run([PY, "-u", str(Path(__file__).with_name("demand_false_alarms.py")),
                        run_id], check=False)
    return json.loads(path.read_text()) if path.exists() else {}


def main(frac: float = 0.3, epochs: int = 12, seed: int = 0) -> int:
    print(__doc__.split("\n\n")[0])
    folds = fold_runs(frac, epochs, seed)

    base = json.loads((RUNS / BASELINE_RUN / "metrics.json").read_text())
    base_auroc = base["test_2"]["auroc"]
    base_fa = false_alarms(BASELINE_RUN)

    rows = []
    for f in folds:
        if not (RUNS / f["run_id"] / "metrics.json").exists():
            print(f"!!! {f['run_id']} never finished, skipping")
            continue
        m = json.loads((RUNS / f["run_id"] / "metrics.json").read_text())
        fa = false_alarms(f["run_id"])
        if not fa:
            print(f"!!! no false-alarm output for {f['run_id']}")
            continue
        held_scenes = set(demand.GROUPS[f["held_out"]])
        per_hour = {s: r["per_hour"] for s, r in fa["scenes"].items()}
        held = [v for s, v in per_hour.items() if s in held_scenes]
        seen = [v for s, v in per_hour.items() if s not in held_scenes]
        rows.append({"held_out": f["held_out"], "run_id": f["run_id"],
                     "auroc": m["test_2"]["auroc"],
                     # Runs from before the monitor-split change have no last_epoch section.
                     "auroc_last_epoch": m.get("last_epoch", {}).get("auroc"),
                     "held_out_per_hour": float(np.mean(held)),
                     "held_out_scenes": {s: per_hour[s] for s in sorted(held_scenes)
                                         if s in per_hour},
                     "trained_on_per_hour": float(np.mean(seen)),
                     "pooled_per_hour": fa["pooled_per_hour"]})

    print(f"\nbaseline {BASELINE_RUN}: test-2 AUROC {base_auroc:.3f}, "
          f"{base_fa.get('pooled_per_hour', float('nan')):.0f} false alarms/h pooled over all "
          f"18 scenes, none of them seen in training")
    print(f"\n{'held-out group':<16}{'AUROC':>8}{'held-out /h':>13}{'trained-on /h':>15}"
          f"{'pooled /h':>11}")
    for r in rows:
        print(f"{r['held_out']:<16}{r['auroc']:>8.3f}{r['held_out_per_hour']:>13.0f}"
              f"{r['trained_on_per_hour']:>15.0f}{r['pooled_per_hour']:>11.0f}")

    if not rows:
        print("\nno folds completed; nothing to conclude")
        return 1
    held = np.array([r["held_out_per_hour"] for r in rows])
    seen = np.array([r["trained_on_per_hour"] for r in rows])
    aurocs = np.array([r["auroc"] for r in rows])
    median_held = float(np.median(held))
    worst_drop = float(base_auroc - aurocs.min())
    passed = median_held < GATE_PER_HOUR and worst_drop <= GATE_AUROC_DROP

    print(f"\nmedian over folds: {median_held:.0f}/h on the held-out group against "
          f"{np.median(seen):.0f}/h on the groups it trained on")
    print(f"drone side: test-2 AUROC {aurocs.mean():.3f} +- {aurocs.std(ddof=1):.3f} against "
          f"the ambient-free {base_auroc:.3f}, worst fold is {worst_drop:+.3f}")
    median_seen = float(np.median(seen))
    base_rate = base_fa.get("pooled_per_hour") or float("nan")
    # Read as ratios against the ambient-free baseline, not against the gate alone. A fold that
    # cuts the trained-on rate 4.5x has plainly learned something about real ambient even if it
    # is still above an absolute bar, and calling that "cannot fit it" would be wrong.
    cut_seen, cut_held = base_rate / max(median_seen, 1), base_rate / max(median_held, 1)
    if passed:
        verdict = ("Real ambient transfers: exposure to five groups cuts the rate on the sixth, "
                   f"so the {base_rate:,.0f}/h was a training-data problem rather than a limit "
                   "of the model.")
    elif cut_seen >= 2.0 and cut_held < 0.6 * cut_seen:
        verdict = (f"Partial, and the shape of it is the finding: hearing real ambient cuts the "
                   f"rate {cut_seen:.1f}x on the groups it heard but only {cut_held:.1f}x on the "
                   f"group it did not, and the held-out rate stays above the bar. It learns "
                   f"specific soundscapes rather than what a drone is not, which is the same "
                   f"across-site failure leave-one-scene-out found.")
    elif cut_seen >= 2.0:
        verdict = (f"It improves everywhere by about {cut_held:.1f}x but not far enough to reach "
                   f"{GATE_PER_HOUR:,.0f}/h, so more ambient of this kind would narrow the gap "
                   f"without closing it.")
    else:
        verdict = ("It does not even fit the ambient it trained on, so the ceiling is the "
                   "per-channel spectrum itself, not the negative class.")
    print(f"\nGATE 2 {'PASS' if passed else 'FAIL'}: needs median held-out below "
          f"{GATE_PER_HOUR:.0f}/h and no more than {GATE_AUROC_DROP:.2f} AUROC lost")
    print(f"  {verdict}")

    summary = {"baseline_run": BASELINE_RUN, "baseline_auroc": base_auroc,
               "baseline_pooled_per_hour": base_fa.get("pooled_per_hour"),
               "ambient_frac": frac, "epochs": epochs, "seed": seed,
               "gate_per_hour": GATE_PER_HOUR, "gate_auroc_drop": GATE_AUROC_DROP,
               "median_held_out_per_hour": median_held,
               "median_trained_on_per_hour": median_seen,
               "reduction_trained_on": cut_seen, "reduction_held_out": cut_held,
               "worst_auroc_drop": worst_drop, "gate_passed": bool(passed),
               "verdict": verdict, "folds": rows}
    path = RESULTS / "demand_trained.json"
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Leave-one-group-out training on DEMAND ambient.")
    p.add_argument("--frac", type=float, default=0.3,
                   help="ambient negatives per batch, relative to our own channel-frames")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    raise SystemExit(main(frac=a.frac, epochs=a.epochs, seed=a.seed))
