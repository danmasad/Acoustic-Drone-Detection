"""Run a battery of training configurations one after another.

Sequential on purpose: MPS is one compute unit and the model is compute-bound, so two runs in
parallel each take twice as long and buy nothing. Runs whose metrics.json already exists are
skipped, which makes the whole battery resumable after a crash, a reboot, or a deliberate stop.

    .venv/bin/python utils/setup/detach.py logs/headline.log -- \
        .venv/bin/python -u utils/train/run_grid.py headline

Set `RUN_GRID_STOP_HOUR` to give an unattended queue a curfew: no cell starts once that hour has
passed. Resuming later picks up exactly where it stopped.
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import run_dir                                 # noqa: E402
from src.config import run_id as config_run_id                 # noqa: E402
from src.train import train                                    # noqa: E402

WIDTH = 32   # 2.3x faster than their 64 and it overfits less

# One battery per question.
BATTERIES = {
    # Order control: one K=1 and one K=5 at identical width and seed.
    "step5a": [{"subset": "device3", "K": K} for K in (1, 5)],
    # Headline 2x3 grid, three seeds per cell; device3 seed 0 first so step 5a can be
    # re-checked an hour in.
    "headline": [{"subset": s, "K": K, "seed": seed}
                 for s in ("device3", "all24", "best6")
                 for K in (1, 5)
                 for seed in (0, 1, 2)],
    # Does usable phase live inside one recorder? Single-recorder and cross-recorder
    # six-microphone subsets at K=1.
    "coherence": [{"subset": s, "K": 1} for s in
                  ("device1", "device4", "symmetric6", "device1_3", "quad4")],
    # Single-seed descriptive cells; cut from the bottom if the clock runs out.
    "descriptive": ([{"subset": s, "K": K} for s in ("all24", "best6") for K in (3, 9)]
                    + [{"subset": s, "K": 5} for s in ("device1", "device4", "symmetric6")]
                    + [{"subset": s, "K": 5} for s in ("symmetric12", "device1_3")]
                    + [{"subset": "quad4", "K": 5}]),
    # The resampled disc, judged on far-bin azimuth error.
    "polar": [{"subset": "all24", "K": 1, "seed": s, "polar": True} for s in (0, 1, 2)],
    # Sub-band count: 16 was a reconstruction, so halve and double it.
    "bins": [{"subset": "device3", "K": 1, "subbands": b} for b in (8, 32)],
    # K=9, run only because check_history_depth said accuracy was still climbing.
    "k9": [{"subset": "device3", "K": 9, "seed": s} for s in (0, 1, 2)],
    # Layout x loss 2x2 at all24 K=1; rect+wce and resampled+wce already exist.
    "tversky": [{"subset": "all24", "K": 1, "seed": s, "loss": "tversky"} for s in (0, 1, 2)],
    "polarnative": [{"subset": "all24", "K": 1, "seed": s, "polar": True, "disc": "native"}
                    for s in (0, 1, 2)],
    "polarnative_tversky": [{"subset": "all24", "K": 1, "seed": s, "polar": True,
                             "disc": "native", "loss": "tversky"} for s in (0, 1, 2)],
    # Completes the K=9 row so six-vs-twenty-four is compared at equal depth.
    "all24k9": [{"subset": "all24", "K": 9, "seed": s} for s in (0, 1, 2)],
    # Native disc and nine frames together, on the array where depth pays.
    "polarnative_k9": [{"subset": "device3", "K": 9, "seed": s, "polar": True, "disc": "native"}
                       for s in (0, 1, 2)],
}


def run_id(cfg: dict) -> str:
    return config_run_id(base=WIDTH, **cfg)


def too_late() -> bool:
    """True once RUN_GRID_STOP_HOUR has passed, if it is set.

    Checked before every cell rather than once per battery, so an unattended queue stops at a
    known hour instead of in the middle of whatever it happened to reach. Nothing is lost: a
    battery skips the cells already on disk, so resuming costs no repeated work.
    """
    stop = os.environ.get("RUN_GRID_STOP_HOUR", "")
    if not stop.strip():
        return False
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
    pending = [c for c in configs if not (run_dir(run_id(c)) / "metrics.json").exists()]
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
