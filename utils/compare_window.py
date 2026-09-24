"""Delay-and-sum on the Hann STFT cache against the committed rectangular baseline.

The inherited analysis window is rectangular. A Hann taper cuts the sidelobes of each 100 ms
frame, which can only help if those sidelobes are what is pulling the peak off the drone.
Scored on test 2 only, same subsets as `results/baseline.json`. Hann is adopted only if
`peak_db` hit@10 rises by at least 1 pp on both `all24` and `device3`; one subset is not
enough, because the window is a global cache choice.

Gate: Hann and rect have the same frame counts.

Run: .venv/bin/python utils/compare_window.py [--device cpu]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import baseline                                      # noqa: E402
from src.config import RESULTS, segments                      # noqa: E402
from src.stft_cache import cache_dir                          # noqa: E402

SPLIT = "test_2"
SUBSETS = ("all24", "device3")
ADOPT_PP = 0.01
KEYS = ("hit_10deg", "median_angular_deg", "auroc")


def hann_ready() -> list[str]:
    return [seg.id for seg in segments(SPLIT)
            if not (cache_dir(seg, "hann") / "stft.npy").exists()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    missing = hann_ready()
    if missing:
        print("GATE FAIL: Hann cache missing for " + ", ".join(missing))
        return 1

    rect = json.loads((RESULTS / "baseline.json").read_text())
    hann_all, failures, better = {}, [], []
    print(f"{'subset':<10} {'rect hit@10':>12} {'hann hit@10':>12} {'delta':>8}")
    for subset in SUBSETS:
        stored = rect[f"{subset}/{SPLIT}"]["peak_db"]
        print(f"  running DAS {subset} on Hann...", flush=True)
        hann_all[f"{subset}/{SPLIT}"] = baseline.evaluate(
            SPLIT, subset, device=a.device, window="hann", verbose=False)
        h = hann_all[f"{subset}/{SPLIT}"]["peak_db"]
        if h["n_frames"] != stored["n_frames"] or h["n_drone"] != stored["n_drone"]:
            failures.append(f"{subset}: {h['n_frames']}/{h['n_drone']} vs "
                            f"{stored['n_frames']}/{stored['n_drone']}")
        delta = h["hit_10deg"] - stored["hit_10deg"]
        better.append(delta >= ADOPT_PP)
        print(f"{subset:<10} {stored['hit_10deg']:12.4f} {h['hit_10deg']:12.4f} {delta:+8.4f}")

    report = {"split": SPLIT, "adopt": bool(all(better)), "cells": {
        s: {"rect": {k: rect[f"{s}/{SPLIT}"]["peak_db"][k] for k in KEYS},
            "hann": {k: hann_all[f"{s}/{SPLIT}"]["peak_db"][k] for k in KEYS},
            "delta_hit_10deg": float(hann_all[f"{s}/{SPLIT}"]["peak_db"]["hit_10deg"]
                                     - rect[f"{s}/{SPLIT}"]["peak_db"]["hit_10deg"])}
        for s in SUBSETS}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline_hann.json").write_text(json.dumps(hann_all, indent=2) + "\n")
    (RESULTS / "window_hann.json").write_text(json.dumps(report, indent=2) + "\n")
    print(("ADOPT Hann: hit@10 rose by at least "
           f"{100 * ADOPT_PP:.0f} pp on both arrays" if report["adopt"]
           else "KEEP rect: Hann did not improve hit@10 on both arrays"))
    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: Hann frame counts match rect" if not failures else "GATE FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
