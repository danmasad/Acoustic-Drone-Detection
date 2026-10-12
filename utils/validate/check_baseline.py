"""Step 3 gate: does the classical baseline fail the way the literature says it fails?

Delay-and-sum has no parameters, so its behaviour is a property of physics and of our
pipeline, not of any fitting. That makes it the right place to validate the metrics harness:
if the numbers here are sane, a later surprise from the network is the network's doing.

The predicted failure is that angular error grows with range, badly, past roughly 90 m. If
it did not, either the labels or the beamformer would be wrong.

Run (unsandboxed, uses the GPU): .venv/bin/python utils/validate/check_baseline.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baseline import evaluate                                  # noqa: E402
from src.metrics import format_summary                             # noqa: E402

GROWTH = 3.0        # error beyond 60 m must be at least this many times the near-field error


def main() -> int:
    ok = True
    res = evaluate("test_2", "all24")
    for score, m in res.items():
        print(format_summary(score, m))
        print()

    m = res["peak_db"]
    bins = {int(b["lo"]): b for b in m["by_distance"]}
    near, far = bins[0]["median_deg"], bins[60]["median_deg"]
    print(f"range degradation: {near:.1f} deg within 30 m -> {far:.1f} deg at 60-90 m "
          f"({far / near:.1f}x)")
    if not far / near >= GROWTH:
        print(f"   FAIL: expected at least {GROWTH}x degradation with range")
        ok = False

    if not np.isfinite(m["auprc"]) or not 0.0 <= m["auprc"] <= 1.0:
        print("   FAIL: AUPRC out of range")
        ok = False
    prior = m["n_drone"] / m["n_frames"]
    print(f"detection: AUPRC {m['auprc']:.3f} against a positive-class prior of "
          f"{prior:.3f}. Brightest-pixel energy barely separates drone from wind, which is "
          f"the paper's own finding (67% false positives) restated under real metrics, and "
          f"the bar the CNN has to clear.")

    print("\nGATE PASS" if ok else "\nGATE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
