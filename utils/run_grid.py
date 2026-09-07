"""Run a battery of training configurations one after another.

Sequential on purpose: MPS is one compute unit and the model is compute-bound, so two runs in
parallel each take twice as long and buy nothing. Runs whose metrics.json already exists are
skipped, which makes the whole battery resumable after a crash, a reboot, or a deliberate stop.

    .venv/bin/python utils/detach.py logs/headline.log -- \
        .venv/bin/python -u utils/run_grid.py headline

Set `RUN_GRID_STOP_HOUR` to give an unattended queue a curfew: no cell starts once that hour has
passed. Resuming later picks up exactly where it stopped.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RUNS                                    # noqa: E402
from src.train import train                                    # noqa: E402

WIDTH = 32   # notes section 18: 2.3x faster than their 64 and it overfits less

BATTERIES = {
    # Step 5a: the order control needs one K=1 and one K=5 run at identical width and seed.
    "step5a": [{"subset": "device3", "K": K} for K in (1, 5)],

    # 5b headline: the claim is a difference of differences, so every cell gets 3 seeds.
    # device3 first, and seed 0 first within it, because those two runs are also what the
    # step 5a order control compares — so the gate can be re-checked an hour in rather than
    # after the whole battery.
    "headline": [{"subset": s, "K": K, "seed": seed}
                 for s in ("device3", "all24", "best6")
                 for K in (1, 5)
                 for seed in (0, 1, 2)],

    # Why did best6 lose by 18 points? It is 3 microphones from device 3 plus 3 from device 4,
    # while device3 is one recorder. The four recorders were started by hand and corrected by a
    # single integer sample offset each, so cross-device phase is only as good as that one
    # correction, while within-device phase comes from one clock and one converter. Hypothesis:
    # usable spatial information lives inside a device, and cross-device baselines contribute
    # little no matter how good the beampattern looks. Predictions, all at K=1 to keep it cheap:
    # device1 and device4 (6 mics, one recorder) land near device3, symmetric6 (3+3 across two
    # recorders) lands near best6, and device1_3 (6+6) beats both. If that holds, classical
    # sidelobe optimization is actively misleading on real hardware - best6 has the *best*
    # beampattern of every 6-subset and the worst accuracy of the three we ran.
    "coherence": [{"subset": s, "K": 1} for s in
                  ("device1", "device4", "symmetric6", "device1_3", "quad4")],

    # 5b descriptive: single seed, cut from the bottom if the clock runs out.
    "descriptive": ([{"subset": s, "K": K} for s in ("all24", "best6") for K in (3, 9)]
                    + [{"subset": s, "K": 5} for s in ("device1", "device4", "symmetric6")]
                    + [{"subset": s, "K": 5} for s in ("symmetric12", "device1_3")]
                    + [{"subset": "quad4", "K": 5}]),

    # Stage E. The disc projection, on the one cell our paper comparison is made in, so the
    # comparison that decides it is mean azimuth error in the 100-200 m bin against the
    # existing rectangular all24 K=1 runs. Three seeds, because a one-seed win here would be
    # indistinguishable from the +/-1.2 point spread the headline grid already showed.
    "polar": [{"subset": "all24", "K": 1, "seed": s, "polar": True} for s in (0, 1, 2)],

    # Stage F. Sixteen sub-bands was our reconstruction of an under-specified number in the
    # paper (notes section 16). Halving and doubling it turns a documented guess into a
    # documented insensitivity — or, if it moves, into a finding. 16 is already run.
    "bins": [{"subset": "device3", "K": 1, "subbands": b} for b in (8, 32)],

    # Stage G. Only runs if utils/check_history_depth.py says accuracy was still climbing at
    # five frames; if it had saturated, this battery is the thing we deliberately do not run.
    "k9": [{"subset": "device3", "K": 9, "seed": s} for s in (0, 1, 2)],

    # --- second pass: the 2x2 of layout against loss -------------------------------------
    # Four cells at all24 K=1, two of which are already on disk (rect+wce, resampled disc+wce),
    # so that every comparison below changes exactly one thing:
    #   tversky        rect + Tversky vs rect + wce          isolates the loss
    #   polarnative    native disc + wce vs resampled + wce  isolates the interpolation
    #   polarnative_tversky                                  the interaction
    # Tversky is the paper's own loss (alpha 0.3, beta 0.7) and had never been run here at all.
    # Judged on mean azimuth error in the 100-200 m bin, which is where the disc was supposed to
    # help and where it lost by 25 degrees, not on the pooled hit rate that hid the effect.
    "tversky": [{"subset": "all24", "K": 1, "seed": s, "loss": "tversky"} for s in (0, 1, 2)],
    "polarnative": [{"subset": "all24", "K": 1, "seed": s, "polar": True, "disc": "native"}
                    for s in (0, 1, 2)],
    "polarnative_tversky": [{"subset": "all24", "K": 1, "seed": s, "polar": True,
                             "disc": "native", "loss": "tversky"} for s in (0, 1, 2)],

    # The cell the headline claim admits is missing. device3 at K=9 reaches 73.2% and is
    # currently compared against all24 at K=1, so the report has to hedge: "not a statement that
    # six microphones beat twenty-four, since we never trained all 24 at K=9". This trains it.
    # Unconditional, unlike everything above it — no gate, it completes a table.
    "all24k9": [{"subset": "all24", "K": 9, "seed": s} for s in (0, 1, 2)],

    # --- the last untested combination ---------------------------------------------------
    # Two separate things each buy accuracy in the far field, and for reasons that sound
    # independent: the natively steered disc because it stops smearing a low-contrast peak
    # across cells (notes 36), and nine frames of history because persistence averages while
    # phantoms flicker (notes 41). Nothing says whether they compound, cancel, or overlap.
    #
    # `device3`, not `all24`, because history only pays where microphones are scarce (notes 37)
    # and an arm on the array that gains nothing from depth could not answer the question.
    # Weighted CE, not Tversky, even though Tversky owns the best far-bin number we have: the
    # comparison is against `device3_K9_w32_b16_wce_s?` and one variable moves at a time. If the
    # disc survives here, Tversky on top of it is the natural follow-up.
    "polarnative_k9": [{"subset": "device3", "K": 9, "seed": s, "polar": True, "disc": "native"}
                       for s in (0, 1, 2)],
}


def run_id(cfg: dict) -> str:
    from src.config import N_SUBBANDS
    polar = ""
    if cfg.get("polar"):
        polar = "_polarn" if cfg.get("disc", "resample") == "native" else "_polar"
    return (f"{cfg['subset']}_K{cfg['K']}_w{cfg.get('base', WIDTH)}"
            f"_b{cfg.get('subbands', N_SUBBANDS)}_{cfg.get('loss', 'wce')}"
            + polar + f"_s{cfg.get('seed', 0)}")


def too_late() -> bool:
    """True once RUN_GRID_STOP_HOUR has passed, if it is set.

    Checked before every cell rather than once per battery, so an unattended queue stops at a
    known hour instead of in the middle of whatever it happened to reach. Nothing is lost: a
    battery skips the cells already on disk, so resuming costs no repeated work.
    """
    stop = os.environ.get("RUN_GRID_STOP_HOUR", "")
    if not stop.strip():
        return False
    from datetime import datetime

    if datetime.now().hour >= int(stop):
        print(f"past {int(stop):02d}:00, stopping before the next cell", flush=True)
        return True
    return False


def main(names: list[str]) -> int:
    configs, seen = [], set()
    for n in names:
        for c in BATTERIES[n]:
            if run_id(c) not in seen:      # batteries overlap: step5a's cells are in headline
                seen.add(run_id(c))
                configs.append(c)
    pending = [c for c in configs if not (RUNS / run_id(c) / "metrics.json").exists()]
    print(f"{len(configs)} configurations, {len(configs) - len(pending)} already done, "
          f"{len(pending)} to run\n", flush=True)
    t0 = time.time()
    for i, cfg in enumerate(pending, 1):
        if too_late():
            print(f"{len(pending) - i + 1} cells left unrun: "
                  f"{', '.join(run_id(c) for c in pending[i - 1:])}", flush=True)
            break
        print(f"=== [{i}/{len(pending)}] {run_id(cfg)}   "
              f"{(time.time() - t0) / 60:.0f} min elapsed", flush=True)
        try:
            train(base=WIDTH, **cfg)
        except Exception as exc:                    # one bad cell must not sink the battery
            print(f"!!! {run_id(cfg)} failed: {type(exc).__name__}: {exc}", flush=True)
    print(f"\nbattery finished in {(time.time() - t0) / 60:.0f} min", flush=True)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["step5a"]
    unknown = [a for a in args if a not in BATTERIES]
    if unknown:
        sys.exit(f"unknown battery {unknown}; choose from {list(BATTERIES)}")
    raise SystemExit(main(args))
