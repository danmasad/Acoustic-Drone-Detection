"""A hybrid decoder: take delay-and-sum's direction when its peak is unambiguous, the U-Net's
otherwise.

Delay-and-sum is worse than the U-Net overall (32.6% within 10 degrees on test 2 against about
70%), but that average mixes two very different kinds of frame. When the beamformed map has one
peak standing well clear of everything else, its location is physics, not inference, and there
is no reason to believe a network trained on two scenes can improve on it; when the peak barely
clears the floor, delay-and-sum is guessing among phantoms, which is the U-Net's job. The dumps
carry delay-and-sum's peak contrast (`contrast_db`, peak over map median), so the choice can be
made per frame with one threshold and no retraining.

The threshold is swept over quantiles of contrast on test 2's drone frames and the best one is
kept. That is selection on the scoring split, so the pooled gain is optimistic by construction;
the honest number is **cross-fitted**: test 2's drone segment is cut into 30-second blocks,
alternate blocks form two folds, the threshold is chosen on one fold and scored on the other,
and the two halves are pooled. Test 1 is not scored.

Scored grid-snapped, as the headline table is, so the U-Net-only end of the sweep is the
headline number.

Gate: with the threshold at +inf the hybrid reproduces every U-Net run's `metrics.json` hit rate,
and at -inf it reproduces delay-and-sum's in `results/baseline.json`.

Run: .venv/bin/python utils/hybrid_decoder.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics, predict                              # noqa: E402
from src.config import ANGULAR_THRESHOLD, RESULTS, RUNS       # noqa: E402

SPLIT = "test_2"
SEEDS = (0, 1, 2)
PAIRS = {"all24 K=1": ("all24_K1_w32_b16_wce", "all24"),
         "device3 K=1": ("device3_K1_w32_b16_wce", "device3"),
         "device3 K=9": ("device3_K9_w32_b16_wce", "device3")}
QUANTILES = np.concatenate([np.linspace(0.0, 0.95, 20), [0.97, 0.98, 0.99, 0.995, 0.999]])
FOLD_BLOCK = 300                       # frames: 30 s
RANGE_BINS = tuple(zip(metrics.DISTANCE_BINS[:-1], metrics.DISTANCE_BINS[1:]))


def drone_frames(name: str) -> dict:
    d = predict.load(name)
    m = d["has_drone"].astype(bool)
    err = metrics.angular_errors(d["el_idx"][m], d["az_idx"][m], d["az"][m], d["el"][m])
    out = {"hit": (err <= ANGULAR_THRESHOLD).astype(float),
           "key": d["seg"][m].astype(np.int64) * 10 ** 7 + d["frame"][m].astype(np.int64),
           "frame": d["frame"][m], "range_m": d["range_m"][m]}
    if "contrast_db" in d:
        out["contrast"] = d["contrast_db"][m]
    return out


def align(das: dict, unet: dict) -> dict:
    """Delay-and-sum restricted to the U-Net's frames (K>1 drops each segment's first frames)."""
    pos = {k: i for i, k in enumerate(das["key"])}
    idx = np.array([pos[k] for k in unet["key"]])
    return {k: v[idx] for k, v in das.items()}


def hybrid_hit(das: dict, unet: dict, thr: float) -> np.ndarray:
    return np.where(das["contrast"] > thr, das["hit"], unet["hit"])


def best_threshold(das: dict, units: list[dict], thresholds: np.ndarray, sel: np.ndarray) -> float:
    gains = [np.mean([hybrid_hit(das, u, t)[sel].mean() for u in units]) for t in thresholds]
    return float(thresholds[int(np.argmax(gains))])


def main() -> int:
    print(__doc__.split("\n\n")[0])
    base = json.loads((RESULTS / "baseline.json").read_text())
    failures, out = [], {"split": SPLIT, "fold_block_frames": FOLD_BLOCK}

    for pair, (stem, arr) in PAIRS.items():
        das_all = drone_frames(f"baseline_{arr}_{SPLIT}")
        stored_das = base[f"{arr}/{SPLIT}"]["peak_db"]["hit_10deg"]
        if abs(das_all["hit"].mean() - stored_das) > 1e-9:
            failures.append(f"{arr} DAS: {das_all['hit'].mean():.6f} vs stored {stored_das:.6f}")

        units = []
        for s in SEEDS:
            name = f"{stem}_s{s}_{SPLIT}_causal"
            if not (predict.PREDS / f"{name}.npz").exists():
                continue
            u = drone_frames(name)
            stored = json.loads((RUNS / f"{stem}_s{s}" / "metrics.json").read_text())[SPLIT]
            if abs(u["hit"].mean() - stored["hit_10deg"]) > 1e-6:
                failures.append(f"{stem}_s{s}: {u['hit'].mean():.6f} vs {stored['hit_10deg']:.6f}")
            units.append(u)
        das = align(das_all, units[0])
        if any(not np.array_equal(u["key"], units[0]["key"]) for u in units):
            failures.append(f"{pair}: seeds do not share frames")

        thresholds = np.concatenate([np.quantile(das["contrast"], QUANTILES), [np.inf]])
        everything = np.ones(len(das["hit"]), bool)
        sweep = [{"threshold_db": float(t), "das_share": float(np.mean(das["contrast"] > t)),
                  "hit_10deg": float(np.mean([hybrid_hit(das, u, t).mean() for u in units]))}
                 for t in thresholds]
        unet_only = sweep[-1]["hit_10deg"]
        t_in = best_threshold(das, units, thresholds, everything)
        in_sample = next(r for r in sweep if r["threshold_db"] == t_in)

        fold = (das["frame"] // FOLD_BLOCK) % 2 == 0
        cross = np.zeros(len(das["hit"]))
        t_fold = {}
        for f, (fit, score) in {"even": (fold, ~fold), "odd": (~fold, fold)}.items():
            t = best_threshold(das, units, thresholds, fit)
            t_fold[f] = t
            cross[score] = np.mean([hybrid_hit(das, u, t)[score] for u in units], 0)
        seed_mean_unet = np.mean([u["hit"] for u in units], 0)

        by_range = []
        for lo, hi in RANGE_BINS:
            m = (das["range_m"] >= lo) & (das["range_m"] < hi)
            if not m.any():
                continue
            by_range.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                             "das": float(das["hit"][m].mean()),
                             "unet": float(seed_mean_unet[m].mean()),
                             "hybrid_cross_fitted": float(cross[m].mean()),
                             "das_share_cross_fitted": float(np.mean(
                                 das["contrast"][m] > np.where(fold[m], t_fold["odd"],
                                                               t_fold["even"])))})

        # Is delay-and-sum ever better than the U-Net, contrast bin by contrast bin?
        edges = np.quantile(das["contrast"], np.linspace(0, 1, 11))
        by_contrast = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (das["contrast"] >= lo) & (das["contrast"] <= hi)
            by_contrast.append({"lo_db": float(lo), "hi_db": float(hi), "n": int(m.sum()),
                                "das": float(das["hit"][m].mean()),
                                "unet": float(seed_mean_unet[m].mean()),
                                "median_range_m": float(np.median(das["range_m"][m]))})
        # The ceiling for any per-frame gate: whichever of the two is right on each frame.
        oracle = float(np.mean([np.maximum(das["hit"], u["hit"]).mean() for u in units]))

        out[pair] = {"seeds": len(units), "n_frames": int(len(das["hit"])),
                     "das_only": float(das["hit"].mean()), "unet_only": unet_only,
                     "in_sample": in_sample, "oracle_gate": oracle,
                     "cross_fitted": {"hit_10deg": float(cross.mean()),
                                      "thresholds_db": t_fold,
                                      "gain_pp": 100 * float(cross.mean() - unet_only)},
                     "by_range": by_range, "by_contrast_decile": by_contrast, "sweep": sweep}

        print(f"\n{pair} ({len(units)} seeds, {len(das['hit'])} drone frames)")
        print(f"  DAS only {100 * das['hit'].mean():.1f}%   U-Net only {100 * unet_only:.1f}%")
        print(f"  best threshold in sample {in_sample['threshold_db']:.2f} dB "
              f"(DAS on {100 * in_sample['das_share']:.1f}% of frames): "
              f"{100 * in_sample['hit_10deg']:.1f}%")
        print(f"  cross-fitted: {100 * cross.mean():.1f}% "
              f"({100 * (cross.mean() - unet_only):+.2f} pp), thresholds "
              f"{t_fold['even']:.2f} / {t_fold['odd']:.2f} dB")
        print(f"  oracle gate (right answer of the two on every frame): {100 * oracle:.1f}%")
        print("  by contrast decile, DAS vs U-Net within 10 deg:")
        for r in by_contrast:
            flag = "  <- DAS better" if r["das"] > r["unet"] else ""
            print(f"    {r['lo_db']:5.2f}-{r['hi_db']:5.2f} dB  {100 * r['das']:5.1f}% vs "
                  f"{100 * r['unet']:5.1f}%   median range {r['median_range_m']:4.0f} m{flag}")
        print("  by range (cross-fitted hybrid):")
        for r in by_range:
            print(f"    {r['lo']:>4.0f}-{r['hi']:<4.0f} m  DAS {100 * r['das']:5.1f}%  "
                  f"U-Net {100 * r['unet']:5.1f}%  hybrid {100 * r['hybrid_cross_fitted']:5.1f}%"
                  f"  (DAS used on {100 * r['das_share_cross_fitted']:.1f}%)  n {r['n']}")

    path = RESULTS / "hybrid_decoder.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {path}")
    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: both ends of the sweep reproduce the stored hit rates"
          if not failures else "GATE FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
