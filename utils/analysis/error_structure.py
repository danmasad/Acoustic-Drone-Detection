"""Where the direction errors are, and what shape they have: range, elevation, and sign.

The notes report direction error as a mean or median per range bin. Two questions those numbers
cannot answer, and both change what should be fixed:

1. **Is the far-field error random or systematic?** A model that is lost scatters its answers
   around the circle; a model with a geometric mistake (a mirror ambiguity, a steering bias, a
   label offset) puts them in a structured place. Predicted-vs-true azimuth per range bin shows
   which, and the signed error summarizes it: its circular mean is the bias, its mean resultant
   length R is how concentrated it is (1 = one fixed offset, 0 = uniform on the circle), and the
   share within 30 degrees of a 180-degree flip tests the front-back ambiguity specifically.
2. **Is the disc's gain an elevation effect?** The native disc spends its pixels differently from
   the rectangle — finer in azimuth at the rim (the horizon), coarser near the pole — so if it
   helps anywhere it should help at low elevation. Error by elevation bin, rectangle vs disc,
   on the same frames and seeds, says whether it does.

Every number uses the continuous centroid decode, as `utils/analysis/compare_polar.py` does, so the
tables line up with the layout comparisons (notes 42's provenance note). Delay-and-sum is shown
alongside as the model-free reference.

Gate: the pooled hit rate within 10 degrees and the per-range-bin mean azimuth error of every
model cell reproduce `results/polar_vs_rect.json` and `results/polar_k9.json`.

Run: .venv/bin/python utils/analysis/error_structure.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import FIGURES, pred_file, table_file        # noqa: E402
from src.geometry import angular_distance                    # noqa: E402

SPLIT = "test_2"
SEEDS = (0, 1, 2)
RANGE_BINS = ((0, 50), (50, 100), (100, 200))
# Test 2 has only 307 drone frames above 75 degrees, so the top two 15-degree bins are merged.
ELEVATION_BINS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90))
FLIP_TOLERANCE = 30.0

PAIRS = {
    "all24 K=1": ("all24_K1_w32_b16_wce", "all24_K1_w32_b16_wce_polarn",
                  "polar_vs_rect.json", "rect + wce", "disc native + wce"),
    "device3 K=9": ("device3_K9_w32_b16_wce", "device3_K9_w32_b16_wce_polarn",
                    "polar_k9.json", "device3 K=9 rect", "device3 K=9 disc"),
}
BASELINES = {"all24 K=1": "baseline_all24", "device3 K=9": "baseline_device3"}


def wrap180(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def frames(path: Path) -> dict:
    d = np.load(path)
    m = d["has_drone"].astype(bool)
    az, el, taz, tel = d["pred_az"][m], d["pred_el"][m], d["az"][m], d["el"][m]
    signed = wrap180(az - taz)
    return {"pred_az": az, "true_az": taz, "true_el": tel, "range_m": d["range_m"][m],
            "signed_az": signed,
            "azimuth": np.abs(signed) * np.cos(np.radians(tel)),
            "angular": angular_distance(az, el, taz, tel)}


def load_cell(stem: str) -> list[dict]:
    return [frames(p) for s in SEEDS
            if (p := pred_file(f"{stem}_s{s}_{SPLIT}_causal")).exists()]


def select(f: dict, key: str, lo: float, hi: float) -> np.ndarray:
    return (f[key] >= lo) & (f[key] < hi)


def shape(signed: np.ndarray) -> dict:
    """Circular summary of signed azimuth errors, in degrees."""
    z = np.exp(1j * np.radians(signed)).mean()
    return {"n": int(len(signed)),
            "bias_deg": float(np.degrees(np.angle(z))),
            "resultant_length": float(np.abs(z)),
            "within_10": float(np.mean(np.abs(signed) <= 10)),
            "near_flip": float(np.mean(np.abs(signed) >= 180 - FLIP_TOLERANCE)),
            "uniform_near_flip": FLIP_TOLERANCE / 180.0}


def by_elevation(runs: list[dict]) -> list[dict]:
    rows = []
    for lo, hi in ELEVATION_BINS:
        hit = [float(np.mean(r["angular"][select(r, "true_el", lo, hi)] <= 10)) for r in runs]
        med = [float(np.median(r["angular"][select(r, "true_el", lo, hi)])) for r in runs]
        azm = [float(np.mean(r["azimuth"][select(r, "true_el", lo, hi)])) for r in runs]
        rows.append({"lo": lo, "hi": hi, "n": int(select(runs[0], "true_el", lo, hi).sum()),
                     "hit_10deg": float(np.mean(hit)), "hit_10deg_sd": float(np.std(hit)),
                     "median_angular": float(np.mean(med)), "mean_azimuth": float(np.mean(azm))})
    return rows


def by_range(runs: list[dict]) -> list[dict]:
    rows = []
    for lo, hi in RANGE_BINS:
        signed = np.concatenate([r["signed_az"][select(r, "range_m", lo, hi)] for r in runs])
        rows.append({"lo": lo, "hi": hi, **shape(signed),
                     "per_seed_bias_deg": [shape(r["signed_az"][select(r, "range_m", lo, hi)])
                                           ["bias_deg"] for r in runs]})
    return rows


def gate(runs: list[dict], ref: dict, name: str, failures: list) -> None:
    # polar_vs_rect.json nests per-bin statistics as {"mean": [...], ...}; polar_k9.json does not.
    def stored(key, inner):
        v = ref[key]
        return v[inner] if isinstance(v, dict) else v

    hit = [float(np.mean(r["angular"] <= 10)) for r in runs]
    if not np.allclose(hit, stored("hit_10deg", "per_seed"), atol=1e-6):
        failures.append(f"{name}: hit_10deg {hit} vs stored {ref['hit_10deg']}")
    azm = [float(np.mean([np.mean(r["azimuth"][select(r, "range_m", lo, hi)]) for r in runs]))
           for lo, hi in RANGE_BINS]
    if not np.allclose(azm, stored("mean_azimuth", "mean"), atol=1e-4):
        failures.append(f"{name}: mean_azimuth {azm} vs stored {ref['mean_azimuth']}")


def main() -> int:
    print(__doc__.split("\n\n")[0])
    out, failures, loaded = {"split": SPLIT}, [], {}
    for pair, (rect, disc, ref_file, ref_rect, ref_disc) in PAIRS.items():
        ref = json.loads(table_file(ref_file).read_text())["cells"]
        arms = {"rect": load_cell(rect), "disc": load_cell(disc),
                "das": [frames(pred_file(f"{BASELINES[pair]}_{SPLIT}"))]}
        gate(arms["rect"], ref[ref_rect], f"{pair} rect", failures)
        gate(arms["disc"], ref[ref_disc], f"{pair} disc", failures)
        loaded[pair] = arms
        out[pair] = {arm: {"seeds": len(runs), "by_elevation": by_elevation(runs),
                           "azimuth_error_shape_by_range": by_range(runs)}
                     for arm, runs in arms.items()}
        e_rect, e_disc = out[pair]["rect"]["by_elevation"], out[pair]["disc"]["by_elevation"]
        out[pair]["disc_minus_rect_by_elevation"] = [
            {"lo": a["lo"], "hi": a["hi"], "n": a["n"],
             "hit_10deg_pp": 100 * (b["hit_10deg"] - a["hit_10deg"]),
             "mean_azimuth_deg": b["mean_azimuth"] - a["mean_azimuth"]}
            for a, b in zip(e_rect, e_disc)]

        print(f"\n{pair}: within 10 deg by elevation (rect / disc / DAS), frames")
        for a, b, c in zip(e_rect, e_disc, out[pair]["das"]["by_elevation"]):
            print(f"  {a['lo']:>2}-{a['hi']:<2} deg  {100 * a['hit_10deg']:5.1f}% "
                  f"/ {100 * b['hit_10deg']:5.1f}% / {100 * c['hit_10deg']:5.1f}%   "
                  f"n {a['n']}")
        print(f"{pair}: signed azimuth error by range (bias, R, within 10, near 180 flip)")
        for arm in ("rect", "disc", "das"):
            for row in out[pair][arm]["azimuth_error_shape_by_range"]:
                print(f"  {arm:<4} {row['lo']:>3}-{row['hi']:<3} m  bias {row['bias_deg']:+7.1f}  "
                      f"R {row['resultant_length']:.2f}  within10 {100 * row['within_10']:5.1f}%  "
                      f"flip {100 * row['near_flip']:5.1f}% (uniform "
                      f"{100 * row['uniform_near_flip']:.0f}%)  n {row['n']}")

    path = table_file("error_by_elevation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    plot(loaded, out)

    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: hit rates and range-bin azimuth errors reproduce the polar JSONs"
          if not failures else "GATE FAIL")
    return 1 if failures else 0


def plot(loaded: dict, out: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for row, (pair, arms) in enumerate(loaded.items()):
        seed0 = arms["rect"][0]
        for col, (lo, hi) in enumerate(RANGE_BINS):
            ax = axes[row, col]
            m = select(seed0, "range_m", lo, hi)
            ax.scatter(seed0["true_az"][m], seed0["pred_az"][m], s=2, alpha=0.25, lw=0)
            ax.plot([-180, 180], [-180, 180], color="0.4", lw=0.8)
            for off in (-180, 180):
                ax.plot([-180, 180], [-180 + off, 180 + off], color="0.4", lw=0.8, ls=":")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-180, 180)
            ax.set_title(f"{pair} rect s0, {lo}-{hi} m (n {int(m.sum())})", fontsize=9)
            ax.set_xlabel("true azimuth (deg)")
            if col == 0:
                ax.set_ylabel("predicted azimuth (deg)")
        ax = axes[row, 3]
        centres = [(lo + hi) / 2 for lo, hi in ELEVATION_BINS]
        for arm, style in (("rect", "-o"), ("disc", "-s"), ("das", "--^")):
            e = out[pair][arm]["by_elevation"]
            ax.errorbar(centres, [100 * r["hit_10deg"] for r in e],
                        yerr=[100 * r["hit_10deg_sd"] for r in e], fmt=style, capsize=3,
                        label=arm)
        ax.set_xlabel("true elevation (deg)")
        ax.set_ylabel("within 10 deg (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"{pair}: hit rate by elevation (seed sd)", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(f"Direction error structure, {SPLIT} (continuous decode; dotted = 180 deg flip)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig_path = FIGURES / f"error_structure_{SPLIT}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    sys.exit(main())
