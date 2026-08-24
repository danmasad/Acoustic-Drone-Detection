"""Leave-one-scene-out: the only protocol this dataset's negatives can honestly support.

Detection has been reported by training on the `train` split and testing on `test_2`. Listing
the actual segments shows why that number is optimistic:

    train      Nov_25_2024  pre-takeoff          2.1 min ambient
               Mar_18_2025  f1 pre-takeoff       1.3 min
               Mar_18_2025  f2 no_drone_1        4.8 min
    test_2     Mar_18_2025  f2 no_drone_2        4.8 min
               Mar_18_2025  f2 pre-takeoff       0.8 min

The training and test ambient are **two halves of one continuous recording** of the same place
on the same afternoon with the same array. Whatever the detector learns about that soundscape
transfers to its own second half for free, so the reported AUROC measures within-scene
performance and calls it generalization.

The fix is to make the fold boundary a scene boundary. Three scenes exist, defined by session
and flight, and each is held out in turn while the other two train. Every fold's test audio is
then from a place and time the model never saw.

Early stopping cannot use the held-out scene, and this is not a formality. The first run of this
protocol showed AUROC on the unseen scene bouncing between 0.36 and 0.80 from one epoch to the
next, so "best epoch on the held-out scene" would have reported something close to 0.80 that a
rerun with a fixed epoch budget could not reproduce. Convergence is therefore monitored on 20% of
the *training* scenes, held out as interleaved 30-second blocks, and the held-out scene is scored
exactly once with the restored checkpoint. Because that checkpoint is still the argmax of a noisy
sequence, the score at the final epoch is reported next to it: when the two disagree, the gap is
the honest error bar on the protocol.

Three caveats that stay attached to these numbers:

- Scenes B and C share a session, so they share a site and an array calibration. Only holding
  out A is a fully independent location.
- Scene C holds three quarters of all the ambient audio, so the fold that holds it out trains
  on 2,042 negative frames. That fold is expected to be the worst, and it is the most honest
  estimate of what happens when a detector meets a new place having barely seen any.
- The monitor split is inside the training scenes, so it measures convergence and not
  generalization. That is all it is for.

test_1 stays sealed. This analysis pools `train` and `test_2` segments only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import detector                                         # noqa: E402
from src.config import RESULTS, segments                         # noqa: E402

# Scene = one recording session and flight. Segment ids, so a typo fails loudly.
MONITOR_FRACTION = 0.2       # of the training scenes, used only for early stopping

SCENES = {
    "A_Nov25": ["Nov_25_2024_f11_drone", "Nov_25_2024_f11_pre_takeoff"],
    "B_Mar18_f1": ["Mar_18_2025_f1_drone", "Mar_18_2025_f1_pre_takeoff"],
    "C_Mar18_f2": ["Mar_18_2025_f2_drone", "Mar_18_2025_f2_no_drone_1",
                   "Mar_18_2025_f2_no_drone_2", "Mar_18_2025_f2_pre_takeoff"],
}


def by_id() -> dict:
    out = {}
    for split in ("train", "test_2"):
        for seg in segments(split):
            out[seg.id] = seg
    return out


def scene_segments(names) -> list:
    lookup = by_id()
    segs = []
    for name in names:
        for sid in SCENES[name]:
            if sid not in lookup:
                sys.exit(f"unknown segment id {sid}")
            segs.append(lookup[sid])
    return segs


def variant_name(confusers: bool, log_bins: int) -> str:
    parts = ["confusers" if confusers else "clean",
             f"log{log_bins}" if log_bins else "linear"]
    return "+".join(parts)


def main(seed: int = 0, epochs: int = 12, confusers: bool = False,
         log_bins: int = 0) -> int:
    print(__doc__.split("\n\n")[0])
    variant = variant_name(confusers, log_bins)
    print(f"\nvariant: {variant}")
    folds, rows = {}, []
    for held in SCENES:
        rest = [s for s in SCENES if s != held]
        print(f"\n=== holding out {held}, training on {', '.join(rest)} ===")
        r = detector.train(seed=seed, epochs=epochs, tag=f"loso_{held}",
                           train_segs=scene_segments(rest),
                           val_segs=scene_segments([held]),
                           confusers=confusers, log_bins=log_bins,
                           monitor_fraction=MONITOR_FRACTION)
        # The module names its validation block "test_2"; here it holds the held-out scene.
        folds[held] = {"metrics": r["test_2"], "last_epoch": r.get("last_epoch"),
                       "best_epoch": r.get("best_epoch"),
                       "confuser_rejection": r.get("confuser_rejection")}
        rows.append((held, r["test_2"], r.get("confuser_rejection"), r.get("last_epoch")))

    au = np.array([m["auroc"] for _, m, _, _ in rows])
    rej = np.array([m["specificity_at_95_recall"] for _, m, _, _ in rows])
    last = np.array([(l or m)["auroc"] for _, m, _, l in rows])
    print(f"\nLeave-one-scene-out ({variant}), per fold:")
    print(f"  {'held-out scene':<14}{'AUROC':>8}{'last epoch':>12}"
          f"{'reject @95% recall':>21}{'frames':>9}")
    for held, m, _, l in rows:
        print(f"  {held:<14}{m['auroc']:>8.3f}{(l or m)['auroc']:>12.3f}"
              f"{m['specificity_at_95_recall'] * 100:>19.0f}%{m['n_frames']:>9}")
    print(f"  {'mean':<14}{au.mean():>8.3f}{last.mean():>12.3f}{rej.mean() * 100:>19.0f}%")
    print(f"  {'spread (sd)':<14}{au.std(ddof=1):>8.3f}{last.std(ddof=1):>12.3f}"
          f"{rej.std(ddof=1) * 100:>19.0f}%")
    print("  'last epoch' is the same held-out scene scored with the weights training ended on "
          "rather than\n  the checkpoint early stopping picked. A wide gap means the selection "
          "did the work.")

    if any(c for _, _, c, _ in rows):
        print("\nESC-50 fold 5, added on top of the held-out scene's own ambient, at the "
              "threshold that catches 95% of drone frames:")
        print(f"  {'held-out scene':<14}{'clean ambient':>15}{'all classes':>14}"
              f"{'drone-like':>13}{'worst class':>22}")
        for held, _, c, _ in rows:
            if not c:
                continue
            print(f"  {held:<14}{c['clean_fire_rate'] * 100:>14.0f}%"
                  f"{c['mean_fire_rate'] * 100:>13.0f}%"
                  f"{c['hard_class_fire_rate'] * 100:>12.0f}%{c['worst_class']:>22}")
        rates = {}
        for _, _, c, _ in rows:
            for cls, v in (c or {}).get("per_class", {}).items():
                rates.setdefault(cls, []).append(v["fire_rate"])

        # The eight classes named in advance, not the eight that happened to score highest:
        # picking the worst classes after the fact would report the tail of fifty draws.
        print("\n  the drone-like classes, chosen before running this (src.detector.HARD_"
              "CLASSES), mean fire rate over folds:")
        for cls in detector.HARD_CLASSES:
            if cls in rates:
                print(f"    {cls:<22}{100 * np.mean(rates[cls]):>5.0f}%")
        other = [np.mean(v) for c, v in rates.items() if c not in detector.HARD_CLASSES]
        print(f"    {'all 42 others (mean)':<22}{100 * np.mean(other):>5.0f}%")
        top = sorted(rates.items(), key=lambda kv: -np.mean(kv[1]))[:5]
        print("  worst five whatever they are: "
              + ", ".join(f"{c} {100 * np.mean(v):.0f}%" for c, v in top))
        print("  Read these against the clean-ambient column, not against zero: a detector that "
              "already fires\n  on quiet ambient will fire on anything added to it, and the "
              "difference is what the sound contributed.")

    print(f"\nThe within-scene protocol we had been reporting gave AUROC 0.662 while training "
          f"on ambient from the same recording as the test.")
    print(f"Leave-one-scene-out says {au.mean():.3f} +/- {au.std(ddof=1):.3f}. The spread "
          f"across scenes is the number that belongs in the report.")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"detector_loso_{variant.replace('+', '_')}.json"
    path.write_text(json.dumps({"seed": seed, "variant": variant, "scenes": dict(SCENES),
                                "confusers": confusers, "log_bins": log_bins,
                                "folds": folds, "mean_auroc": float(au.mean()),
                                "sd_auroc": float(au.std(ddof=1))}, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Leave-one-scene-out for the per-channel detector.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--confusers", action="store_true")
    p.add_argument("--log-bins", type=int, default=0)
    a = p.parse_args()
    raise SystemExit(main(a.seed, a.epochs, a.confusers, a.log_bins))
