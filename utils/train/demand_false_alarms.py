"""False alarms per hour on 90 minutes of real ambient audio the detector has never heard.

Why this exists. Our own no-drone audio is 5.6 minutes of test 2, so the finest non-zero
false-alarm rate the dataset can express is one alarm in 5.6 minutes = 10.7 per hour. Every
detection number in this project has been quoted against that floor, which means "0 false alarms"
and "10 per hour" are the same measurement. DEMAND supplies 90 minutes across 18 environments at
our own 48 kHz, taking the resolvable floor to about 0.7 per hour and — more useful than the
extra resolution — replacing one number with a distribution over soundscapes.

The threshold is **not** chosen here. It is the one already fixed on our own data at 95% drone
recall and stored in the run's checkpoint, so nothing about DEMAND informs where the operating
point sits. That is the whole point: this measures what an operating point chosen on our data
does when it meets somewhere else.

What the number is and is not:

- It **is** the rate at which this detector fires on real continuous noise from environments
  absent from its training set, per environment.
- It **is not** what our array would do in those places. DEMAND used its own microphones, gain
  and windscreens. Each scene is level-matched to the median in-band energy of our own ambient
  so that spectral shape and variability drive the answer rather than recording level, but a
  microphone mismatch cannot be normalized away. Reported as indicative, with the mismatch
  stated, next to the coarse-but-ours 10.7 per hour.

Only single channels are used, so the array-geometry mismatch is irrelevant by construction: the
per-channel detector reads one microphone at a time. The map model cannot be scored on this at
all, and faking it by replicating one channel 24 times would manufacture a spatial pattern that
no real source produces.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import demand, detector                                   # noqa: E402
from src.config import CACHE_BAND, CHUNK, SR, run_dir, segments, table_file  # noqa: E402
from src.data import SplitData                                      # noqa: E402

BATCH = 512


RECALLS = (0.95, 0.90, 0.75, 0.50)


@torch.no_grad()
def our_own(model, channels, K: int, stats, dev, bank, band=None,
            verbose: bool = False) -> dict:
    """Our own test-2 frames: per-frame scores, labels, and the ambient energy reference.

    One pass gives everything the comparison needs. The thresholds come from the *drone* frames,
    so that "95% recall" means the same thing here as it does anywhere else in the report, and
    our own ambient is scored at those same thresholds so that the DEMAND column is read against
    a like-for-like number rather than against zero.
    """
    data = SplitData("test_2", channels, K, band=band or CACHE_BAND, verbose=verbose,
                     segs=segments("test_2"))
    scores, labels, energies = [], [], []
    for _, X, _, present, _ in data.batches(256, shuffle=False):
        x = detector.spectra(X.to(dev), *stats, bank=bank)
        scores.append(model(x).reshape(len(present), -1).mean(1).cpu().numpy())
        labels.append(present.numpy().astype(bool))
        keep = ~present.bool()
        if keep.any():
            energies.append((X[keep].abs() ** 2).sum(-1).reshape(-1).numpy())
    s, y = np.concatenate(scores), np.concatenate(labels)
    return {"score": s, "drone": y,
            "reference_energy": float(np.median(np.concatenate(energies))),
            "n_negative": int((~y).sum()),
            "thresholds": {r: float(np.quantile(s[y], 1.0 - r)) for r in RECALLS}}


@torch.no_grad()
def scene_scores(model, amb: demand.Ambient, scene: str, cfg: dict, stats, dev,
                 bank) -> np.ndarray:
    """Per-frame averaged logit for one scene, in the model's own units."""
    X = amb.frames(scene, K=cfg["K"], channels=1)
    out = []
    for i in range(0, len(X), BATCH):
        x = torch.from_numpy(X[i:i + BATCH])
        out.append(model(detector.spectra(x.to(dev), *stats, bank=bank))
                   .reshape(len(x), -1).mean(1).cpu().numpy())
    return np.concatenate(out)


def main(run_id: str = "detector_K1_w32_s0_full", level_match: bool = True) -> int:
    print(__doc__.split("\n\n")[0])
    if not (run_dir(run_id) / "best.pt").exists():
        sys.exit(f"no checkpoint for {run_id}; train the full-data detector first")
    model, saved, stats, bank, dev = detector.load_run(run_id)
    cfg = saved["config"]
    ours = our_own(model, cfg["channels"], cfg["K"], stats, dev, bank,
                   band=tuple(cfg["band"]))
    threshold = ours["thresholds"][0.95]
    n_ours = ours["n_negative"]
    print(f"\n{run_id}: threshold {threshold:.3f}, fixed on our own drone frames at 95% recall.")
    print(f"  our own ambient is {n_ours} frames = {n_ours * CHUNK / SR / 60:.1f} min, resolving "
          f"{3600 / (n_ours * CHUNK / SR):.1f} false alarms/h at best")
    ref = ours["reference_energy"] if level_match else None
    if ref is not None:
        print(f"  level-matching every scene to our ambient median frame energy {ref:.3e}")
    amb = demand.Ambient(reference_energy=ref)

    rows, all_scores = {}, {}
    for scene in amb.scenes:
        s = scene_scores(model, amb, scene, cfg, stats, dev, bank)
        all_scores[scene] = s
        hours = len(s) * CHUNK / SR / 3600.0
        fired = int((s >= threshold).sum())
        rows[scene] = {"group": demand.GROUP_OF.get(scene, "?"),
                       "outdoor": scene in demand.OUTDOOR,
                       "frames": len(s), "hours": hours, "alarms": fired,
                       "per_hour": fired / hours,
                       "fire_rate": fired / len(s),
                       "median_logit": float(np.median(s)),
                       "gain_db": 20 * np.log10(amb.gain.get(scene, 1.0))}

    # One operating point cannot describe a detector, and at 95% recall this one is saturated:
    # it fires on nearly every frame of our own ambient too. Sweeping the recall the threshold
    # is set for shows whether the false-alarm rate is a tunable trade or a wall.
    sweep = []
    everything = np.concatenate([all_scores[s] for s in amb.scenes])
    for r in RECALLS:
        t = ours["thresholds"][r]
        ours_fa = float((ours["score"][~ours["drone"]] >= t).mean())
        dem_fa = float((everything >= t).mean())
        sweep.append({"recall": r, "threshold": t, "ours_fire_rate": ours_fa,
                      "ours_per_hour": ours_fa * 3600 / (CHUNK / SR),
                      "demand_fire_rate": dem_fa,
                      "demand_per_hour": dem_fa * 3600 / (CHUNK / SR)})
    print(f"\n{'drone recall':>13}{'threshold':>11}{'our ambient fires':>19}"
          f"{'DEMAND fires':>15}{'DEMAND alarms/h':>17}")
    for w in sweep:
        print(f"{w['recall'] * 100:>12.0f}%{w['threshold']:>11.2f}"
              f"{w['ours_fire_rate'] * 100:>18.0f}%{w['demand_fire_rate'] * 100:>14.0f}%"
              f"{w['demand_per_hour']:>17.0f}")

    print(f"\n{'scene':<12}{'group':<11}{'min':>6}{'alarms':>8}{'per hour':>10}"
          f"{'level adj':>11}")
    for scene, r in sorted(rows.items(), key=lambda kv: -kv[1]["per_hour"]):
        mark = " *" if r["outdoor"] else "  "
        print(f"{scene:<12}{r['group']:<11}{r['hours'] * 60:>6.1f}{r['alarms']:>8}"
              f"{r['per_hour']:>10.1f}{r['gain_db']:>+10.1f} dB{mark}")
    print("  * outdoor scene: wind, traffic and open sky, the honest comparison for an array "
          "that sits in a field")

    per_hour = np.array([r["per_hour"] for r in rows.values()])
    out_mask = np.array([r["outdoor"] for r in rows.values()])
    total_alarms = sum(r["alarms"] for r in rows.values())
    total_hours = sum(r["hours"] for r in rows.values())
    summary = {
        "run_id": run_id, "threshold": threshold, "level_matched": level_match,
        "reference_energy": ref, "scenes": rows, "recall_sweep": sweep,
        "pooled_per_hour": total_alarms / total_hours,
        "median_scene_per_hour": float(np.median(per_hour)),
        "iqr_scene_per_hour": [float(np.percentile(per_hour, 25)),
                               float(np.percentile(per_hour, 75))],
        "worst_scene": max(rows, key=lambda s: rows[s]["per_hour"]),
        "outdoor_median_per_hour": float(np.median(per_hour[out_mask])),
        "indoor_median_per_hour": float(np.median(per_hour[~out_mask])),
        "silent_scenes": [s for s, r in rows.items() if r["alarms"] == 0],
    }
    print(f"\npooled over {total_hours:.2f} h: {summary['pooled_per_hour']:.1f} false alarms/h")
    print(f"across scenes: median {summary['median_scene_per_hour']:.1f}/h, "
          f"IQR {summary['iqr_scene_per_hour'][0]:.1f}-{summary['iqr_scene_per_hour'][1]:.1f}, "
          f"worst {summary['worst_scene']} at {rows[summary['worst_scene']]['per_hour']:.1f}/h")
    print(f"outdoor median {summary['outdoor_median_per_hour']:.1f}/h against indoor "
          f"{summary['indoor_median_per_hour']:.1f}/h")
    if summary["silent_scenes"]:
        print(f"never fired in {len(summary['silent_scenes'])} scenes: "
              f"{', '.join(summary['silent_scenes'])}")

    # The spread is the result. A single pooled rate would let one loud scene stand in for
    # eighteen environments, and the report needs to say which environments are the problem.
    print("\nThe spread across scenes, not the pooled rate, is what this measures: an operating "
          "point picked on one field's ambient does not transfer uniformly to a metro platform.")

    path = table_file(f"demand_false_alarms_{run_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_id", nargs="?", default="detector_K1_w32_s0_full")
    p.add_argument("--raw-level", action="store_true", help="skip level matching")
    a = p.parse_args()
    raise SystemExit(main(run_id=a.run_id, level_match=not a.raw_level))
