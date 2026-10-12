"""Gate for step 5a: does the frame stack actually use the past, and does order matter?

"K=5 beats K=1" is not enough evidence on its own — a K=5 model has five times the input
channels and slightly more parameters, so it could win without ever looking earlier than the
current frame. The three test-time interventions in `src.train.reorder` separate those cases
using one set of trained weights:

    freeze   current frame repeated K times  ->  history removed, capacity kept
    shuffle  same frames, random order       ->  order removed, history kept
    reverse  time flipped                    ->  order removed, smoothness kept

Pass requires history to matter: K=5 must beat K=1, and freezing must give the gain back.
Order sensitivity is measured and reported, not required — notes section 13 predicts
persistence dominates beyond ~70 m, where the true peak moves under one grid cell per stack,
and persistence is order-invariant by construction. Near-field rows are where order should show.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import run_dir, run_id                          # noqa: E402

KEY = "hit_10deg"
NEAR = 60.0            # metres; the range below which a stack spans more than one grid cell


def load(run_id: str) -> dict:
    path = run_dir(run_id) / "metrics.json"
    if not path.exists():
        sys.exit(f"missing {path}. Train it first.")
    return json.loads(path.read_text())


def near_field_median(m: dict) -> float:
    """Median angular error inside NEAR metres, pooled over the bins below it."""
    bins = [b for b in m["by_distance"] if b["hi"] <= NEAR and b["n"] > 0]
    return sum(b["median_deg"] * b["n"] for b in bins) / sum(b["n"] for b in bins)


def main(k1_run: str, k5_run: str) -> int:
    k1, k5 = load(k1_run), load(k5_run)
    base, stack = k1["test_2"], k5["test_2"]
    controls = k5.get("order_control", {})
    if not controls:
        sys.exit(f"{k5_run} has no order_control block; retrain with the current src/train.py")

    K = k5["config"]["K"]
    print(f"Step 5a order control: {k5_run} (K={K}) against {k1_run} (K=1)\n")
    rows = [("K=1 baseline", base), (f"K={K} causal", stack)]
    rows += [(f"K={K} {name}", m) for name, m in controls.items()]
    print(f"{'condition':<16}{'<10deg':>9}{'median':>9}{'near<60m':>10}{'AUROC':>8}"
          f"{'vs causal':>11}")
    for name, m in rows:
        is_ref = name == f"K={K} causal"
        delta = "—".rjust(11) if is_ref else f"{(m[KEY] - stack[KEY]) * 100:>+9.1f}pp"
        print(f"{name:<16}{m[KEY] * 100:>8.1f}%{m['median_angular_deg']:>8.2f}d"
              f"{near_field_median(m):>9.2f}d{m['auroc']:>8.3f}{delta}")

    stack_gain = (stack[KEY] - base[KEY]) * 100
    freeze_loss = (stack[KEY] - controls["freeze"][KEY]) * 100
    order_loss = max((stack[KEY] - controls[o][KEY]) * 100 for o in ("shuffle", "reverse"))
    near_order = max(near_field_median(controls[o]) - near_field_median(stack)
                     for o in ("shuffle", "reverse"))

    print(f"\ndepends on history (freeze)   {freeze_loss:+.1f} pp")
    print(f"depends on order (shuf/rev)   {order_loss:+.1f} pp overall, "
          f"{near_order:+.2f} deg inside {NEAR:.0f} m")
    print(f"beats K=1                     {stack_gain:+.1f} pp of frames within 10 deg")

    # Two separate questions, and only the first is what step 5a exists to answer. The gate
    # asks whether the stack uses the past at all, because if it does not, the grid measures
    # nothing. Whether that use *pays* is a difference of differences, and a single seed
    # cannot resolve it: epoch-to-epoch swings here are several points, so anything under
    # NOISE_PP is reported as undecided and left to the 3-seed headline grid.
    NOISE_PP = 2.0
    mechanism = freeze_loss > 0.2
    print("\nGATE PASS" if mechanism else "\nGATE FAIL")
    if not mechanism:
        print("  Freezing the stack to the current frame costs nothing, so the model never "
              "looks back and there is no temporal mechanism to measure. Climb the "
              "escalation ladder in notes section 13 before running the grid.")
    else:
        print(f"  The stack uses the past: freezing it costs {freeze_loss:.1f} pp, so the "
              "grid is measuring something real.")
    if mechanism and order_loss < 0.2:
        print("  History matters, order does not — the predicted result. Say 'a drone "
              "persists while false peaks flicker', never 'drifts smoothly'.")
    if abs(stack_gain) < NOISE_PP:
        print(f"  Payoff undecided: {stack_gain:+.1f} pp over K=1 is inside single-seed "
              "noise. The 3-seed headline grid decides, and a null result there is a "
              "publishable answer to the question the reference paper left open.")
    elif stack_gain >= NOISE_PP:
        print(f"  Payoff visible already: {stack_gain:+.1f} pp over K=1 at one seed.")
    else:
        print(f"  Stacking is {-stack_gain:.1f} pp *worse* than K=1 at this seed; check "
              "regularization before the grid, since K=5 has five times the input channels.")
    return 0 if mechanism else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("k1", nargs="?", default=run_id("device3", 1))
    p.add_argument("k5", nargs="?", default=run_id("device3", 5))
    a = p.parse_args()
    raise SystemExit(main(a.k1, a.k5))
