"""Stage H: score everything on test 1, once, and never tune anything afterwards.

Test 1 is three minutes from a different site under conditions the model never saw. It has been
sealed since the first commit: every architecture choice, hyperparameter, array subset, decoder
and early-stopping rule in this repository was decided on test 2. This script exists so that the
final numbers are produced by one command, from checkpoints that already exist, with nothing
selected afterwards.

The discipline only means something if it is visible, so:

- Every model scored here is a checkpoint whose selection is already recorded in its own
  `metrics.json`, chosen on test 2.
- The output is written once to `results/test_1.json`. If that file exists the script refuses to
  overwrite it without `--force`, because "run it again with a tweak" is exactly the failure this
  seal was protecting against.
- Test 1 is small — 1,592 drone frames and 618 no-drone frames — so a single percentage point is
  about 16 frames. Differences that would be meaningful on test 2 are noise here, and the script
  prints the frame counts next to every rate to make that unavoidable.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import detector, predict                                 # noqa: E402
from src.config import FRAME_SECONDS, run_dir, segments, table_file  # noqa: E402
from src.metrics import format_summary, summarize                  # noqa: E402

SPLIT = "test_1"
SUBSETS = ("all24", "device3", "best6")
SEEDS = (0, 1, 2)
CELLS = [(s, k) for s in ("all24", "device3") for k in (1, 5)]


def polar_won() -> bool:
    """Did the disc projection beat the rectangular map on test 2? It did not, and that decision
    has to be read from the recorded comparison rather than from whether checkpoints happen to
    exist, or the seal turns into 'score everything and keep the best'."""
    path = table_file("polar_vs_rect.json")
    if not path.exists():
        return False
    hit = json.loads(path.read_text())["hit_10deg"]
    polar, rect = np.array(hit["polar"], float), np.array(hit["rect"], float)
    if len(polar) < 2 or len(rect) < 2:
        return False
    return polar.mean() - rect.mean() > max(polar.std(ddof=1), rect.std(ddof=1))


def k9_won() -> bool:
    """Same gate for K=9: it enters test 1 only if the extra history beat K=5 on test 2 by more
    than the seed spread."""
    hits: dict[int, list[float]] = {}
    for K in (5, 9):
        for seed in SEEDS:
            path = run_dir(f"device3_K{K}_w32_b16_wce_s{seed}") / "metrics.json"
            if path.exists():
                hits.setdefault(K, []).append(
                    json.loads(path.read_text())["test_2"]["hit_10deg"])
    if len(hits.get(9, [])) < 2 or len(hits.get(5, [])) < 2:
        return False
    nine, five = np.array(hits[9]), np.array(hits[5])
    return 100 * (nine.mean() - five.mean()) > 100 * max(nine.std(ddof=1), five.std(ddof=1))


def cnn_rows() -> list[dict]:
    rows = []
    cells = list(CELLS)
    if k9_won():
        cells.append(("device3", 9))
        print("  K=9 beat K=5 on test 2, so it is scored here too")
    for subset, K in cells:
        for seed in SEEDS:
            run_id = f"{subset}_K{K}_w32_b16_wce_s{seed}"
            if not (run_dir(run_id) / "best.pt").exists():
                continue
            pred = predict.cnn_predictions(run_id, SPLIT, verbose=False)
            m = summarize(pred["score"], pred["has_drone"], pred["el_idx"], pred["az_idx"],
                          pred["az"], pred["el"], pred["range_m"])
            rows.append({"model": f"{subset} K={K}", "run_id": run_id, "seed": seed,
                         "polar": False, **m})
    if not polar_won():
        print("  polar did not beat rectangular on test 2, so it is not scored on test 1")
        return rows
    for seed in SEEDS:
        run_id = f"all24_K1_w32_b16_wce_polar_s{seed}"
        if not (run_dir(run_id) / "best.pt").exists():
            continue
        pred = predict.cnn_predictions(run_id, SPLIT, verbose=False)
        m = summarize(pred["score"], pred["has_drone"], pred["el_idx"], pred["az_idx"],
                      pred["az"], pred["el"], pred["range_m"],
                      pred_az=pred["argmax_az"], pred_el=pred["argmax_el"])
        rows.append({"model": "all24 K=1 polar", "run_id": run_id, "seed": seed,
                     "polar": True, **m})
    return rows


def detector_row(run_id: str) -> dict | None:
    """The per-channel detector on test 1, using the threshold fixed on test 2."""
    from src.data import SplitData

    if not (run_dir(run_id) / "best.pt").exists():
        return None
    model, saved, stats, bank, dev = detector.load_run(run_id)
    cfg = saved["config"]
    data = SplitData(SPLIT, cfg["channels"], cfg["K"], band=tuple(cfg["band"]), verbose=False,
                     segs=segments(SPLIT))
    m = detector.evaluate(model, data, stats, dev, bank=bank)
    m["threshold_from_test_2"] = saved["test_2"]["threshold_at_95_recall"]
    m["run_id"] = run_id
    return m


def mean_sd(rows: list[dict], key: str) -> str:
    v = np.array([r[key] for r in rows], float)
    return f"{v.mean():.1f}" + (f" +/- {v.std(ddof=1):.1f}" if len(v) > 1 else "")


def main(force: bool = False, detector_run: str = "detector_K1_w32_s0_full") -> int:
    out_path = table_file(f"{SPLIT}.json")
    if out_path.exists() and not force:
        sys.exit(f"{out_path} already exists. Test 1 is scored once; pass --force only if you "
                 f"are adding a model that did not exist then, and say so in the report.")
    print(__doc__.split("\n\n")[0])

    print(f"\n=== classical baseline on {SPLIT} ===")
    base = {}
    for subset in SUBSETS:
        pred = predict.baseline_predictions(SPLIT, subset, verbose=False)
        m = summarize(pred["peak_db"], pred["has_drone"], pred["el_idx"], pred["az_idx"],
                      pred["az"], pred["el"], pred["range_m"])
        base[subset] = m
        print(format_summary(f"DAS {subset:<9}", m))
    n_drone = base[SUBSETS[0]]["n_drone"]
    n_neg = base[SUBSETS[0]]["n_frames"] - n_drone
    print(f"\n{SPLIT}: {n_drone} drone frames, {n_neg} no-drone frames "
          f"({n_neg * FRAME_SECONDS / 60:.1f} min, so one false alarm is "
          f"{3600 / max(n_neg * FRAME_SECONDS, 1e-9):.0f}/h)")

    print(f"\n=== U-Net cells on {SPLIT} ===")
    rows = cnn_rows()
    print(f"  {'model':<18}{'within 10 deg':>15}{'median':>12}{'AUROC':>16}")
    grouped = {}
    for r in rows:
        grouped.setdefault(r["model"], []).append(r)
    for model, rs in grouped.items():
        hits = np.array([r["hit_10deg"] for r in rs]) * 100
        med = np.array([r["median_angular_deg"] for r in rs])
        au = np.array([r["auroc"] for r in rs])
        sd = f" +/- {hits.std(ddof=1):.1f}" if len(rs) > 1 else ""
        print(f"  {model:<18}{hits.mean():>10.1f}%{sd:<6}{med.mean():>7.1f}d"
              f"{au.mean():>13.3f}")

    print(f"\n=== per-channel detector on {SPLIT} ===")
    det = detector_row(detector_run)
    if det:
        print(f"  {detector_run}: AUROC {det['auroc']:.3f}, "
              f"rejects {det['specificity_at_95_recall'] * 100:.0f}% of negatives at 95% "
              f"recall, {det['n_frames']} frames")
    else:
        print(f"  {detector_run} not found; train it before unsealing")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"split": SPLIT, "baseline": base, "cnn": rows,
                                    "detector": det}, indent=2))
    print(f"\nwrote {out_path}. Test 1 is now spent: no number here may inform a choice.")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    raise SystemExit(main(force="--force" in a,
                          detector_run=next((x for x in a if not x.startswith("--")),
                                            "detector_K1_w32_s0_full")))
