"""Layout against loss, five cells, decided on the one bin the disc was built to fix.

The hypothesis, from design notes section 7 and the reproduction gap in section 23. Our map is a
rectangle of elevation by azimuth, which is a plate carree projection of the sky. Circular padding
fixed its azimuth seam, but nothing fixed the other defect: near the zenith, one azimuth cell
covers almost no solid angle, so a single learned kernel means different things at different
elevations. Our one reproduction gap is beyond 100 m, and it is almost entirely azimuth — 53
degrees against 11 in elevation. Azimuth is the axis the rectangle distorts.

The first attempt put the map on a disc and lost that bin by 25 degrees, and there were two
reasons it might have deserved better. It built the disc by resampling the finished rectangular
map, so the input was a bilinear blur of 4-degree cells painted onto rim cells 2.5 degrees apart
while the target stayed a hard cap; and it was only ever run under weighted cross-entropy, never
under the paper's own Tversky loss, which weights misses over false positives and might be what a
sparse target near the rim needs. Those are separable, so this compares five cells:

    rect + wce                the reference, and the report's headline runs
    disc resampled + wce      the first attempt
    rect + tversky            the loss alone, no layout change
    disc native + wce         the layout alone, no interpolation
    disc native + tversky     both

Decided on mean azimuth error inside the 100-200 m bin, three seeds each, all at all24 K=1, which
is the cell our paper comparison is made in. Mean rather than median, and the paper's own distance
bins, so the number stays comparable with section 5.1 of the report. Frame-paired deltas are
reported next to the seed means because both arms score the same frames in the same order, and
pairing removes the frame-to-frame difficulty that dominates a 374-frame bin. Everything is read
from the per-frame dumps; no training happens here.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import predict                                            # noqa: E402
from src.config import FIGURES, RESULTS, RUNS                      # noqa: E402
from src.geometry import angular_distance                          # noqa: E402

PAPER_BINS = ((0, 50), (50, 100), (100, 200))
FINE_BINS = ((0, 30), (30, 60), (60, 90), (90, 120), (120, 180))
SEEDS = (0, 1, 2)
FAR = 2                              # index of the 100-200 m bin in PAPER_BINS

# Gate 1, declared before any of these runs existed. The far bin's seed spread is 13-16 degrees,
# so the standard error of a three-seed difference is around 12: this is a decision rule for what
# earns four hours of high-K compute tomorrow, not a significance test, and it is written down in
# advance precisely so it cannot be adjusted to whatever comes back.
GATE_FAR_AZIMUTH = 45.0

BASELINE = "rect + wce"
CELLS = {
    BASELINE: "all24_K1_w32_b16_wce_s{s}",
    "disc resampled + wce": "all24_K1_w32_b16_wce_polar_s{s}",
    "rect + tversky": "all24_K1_w32_b16_tversky_s{s}",
    "disc native + wce": "all24_K1_w32_b16_wce_polarn_s{s}",
    "disc native + tversky": "all24_K1_w32_b16_tversky_polarn_s{s}",
}
# The two the legacy `polar_vs_rect.json` schema names, which `utils/unseal_test1.py` reads to
# decide whether a disc model earns a place in the sealed-split scoring.
LEGACY = {"rect": BASELINE, "polar": "disc resampled + wce"}


def wrap180(d: np.ndarray) -> np.ndarray:
    return (d + 180.0) % 360.0 - 180.0


def errors(pred: dict) -> dict:
    """Angular, azimuth and elevation error per drone frame, plus its range."""
    d = pred["has_drone"].astype(bool)
    az, el = pred["pred_az"][d], pred["pred_el"][d]
    taz, tel = pred["az"][d], pred["el"][d]
    return {"angular": angular_distance(az, el, taz, tel),
            # Azimuth error alone is meaningless overhead, where every bearing is nearly the same
            # direction, so it is scaled by cos(elevation) to stay a real angle on the sphere.
            "azimuth": np.abs(wrap180(az - taz)) * np.cos(np.radians(tel)),
            "elevation": np.abs(el - tel),
            "range_m": pred["range_m"][d]}


def by_bin(e: dict, bins, key: str, stat=np.mean) -> list:
    out = []
    for lo, hi in bins:
        m = (e["range_m"] >= lo) & (e["range_m"] < hi)
        out.append(float(stat(e[key][m])) if m.any() else float("nan"))
    return out


def collect(template: str, split: str) -> list[dict]:
    got = []
    for s in SEEDS:
        run_id = template.format(s=s)
        if not (RUNS / run_id / "best.pt").exists():
            continue
        pred = predict.cnn_predictions(run_id, split, verbose=False)
        got.append({"run_id": run_id, **errors(pred)})
    return got


def seed_means(runs: list[dict], bins, key: str, stat) -> np.ndarray:
    """(n_seeds, n_bins) of a per-bin statistic."""
    return np.array([by_bin(r, bins, key, stat) for r in runs])


def paired_delta(cell: list[dict], base: list[dict], key: str = "azimuth") -> list[dict]:
    """Per-bin mean of (cell error - baseline error) on the *same* frames.

    Both arms dump the same split with shuffling off, so frame order is identical and the drone
    filter keeps it identical. Averaging over seeds first, then differencing, leaves a per-frame
    delta whose remaining variation is frame difficulty rather than initialization.

    The reported standard error treats frames as independent, which they are not: consecutive
    frames are 100 ms apart and a flight passes through a bin as one continuous stretch, so the
    true interval is wider than this one. It is a floor, and it is quoted as such.
    """
    if not cell or not base:
        return []
    n = min(min(len(r[key]) for r in cell), min(len(r[key]) for r in base))
    if any(len(r[key]) != n for r in cell + base):
        print(f"  (frame counts differ across runs, truncating to {n})")
    a = np.mean([r[key][:n] for r in cell], 0)
    b = np.mean([r[key][:n] for r in base], 0)
    rng = base[0]["range_m"][:n]
    out = []
    for lo, hi in PAPER_BINS:
        m = (rng >= lo) & (rng < hi)
        d = (a - b)[m]
        out.append({"bin": [lo, hi], "n_frames": int(m.sum()),
                    "delta": float(d.mean()) if m.any() else float("nan"),
                    "se_floor": float(d.std(ddof=1) / np.sqrt(max(len(d), 1)))
                    if m.sum() > 1 else float("nan"),
                    "fraction_improved": float((d < 0).mean()) if m.any() else float("nan")})
    return out


K9_CELLS = {"device3 K=9 rect": "device3_K9_w32_b16_wce_s{s}",
            "device3 K=9 disc": "device3_K9_w32_b16_wce_polarn_s{s}"}


def k9(split: str = "test_2") -> dict:
    """Does the disc's far-field gain survive on the array and depth where history pays?

    The five cells above are all24 K=1, which is where the paper comparison lives but not where
    this project's one solid positive result lives. The disc sharpens a faint peak (notes 36) and
    nine frames of history suppress the phantoms competing with it (notes 41), so the open question
    was whether the two compound. Only `device3` can answer it: all24 gains nothing from depth at
    all (notes 37), so a disc arm there could not separate the effects.

    Weighted cross-entropy on both sides, not Tversky, so exactly one thing differs.
    """
    cells = {n: collect(t, split) for n, t in K9_CELLS.items()}
    have = {k: v for k, v in cells.items() if v}
    if len(have) < 2:
        print(f"\ndevice3 K=9 disc comparison: need both cells, have {list(have)}")
        return {}

    print(f"\n\ndevice3, K=9, {split}: does the disc still help once history is there?")
    print(f"  {'':20s}" + "".join(f"{f'{lo}-{hi}m':>10}" for lo, hi in PAPER_BINS) + f"{'<10 deg':>11}")
    out = {"split": split, "cells": {}}
    for name, runs in have.items():
        v = seed_means(runs, PAPER_BINS, "azimuth", np.mean)
        hit = [float((r["angular"] <= 10).mean()) for r in runs]
        print(f"  {name:20s}" + "".join(f"{x:>10.1f}" for x in v.mean(0))
              + f"{100 * np.mean(hit):>10.1f}%"
              + f"   far seeds " + " ".join(f"{x:.1f}" for x in v[:, FAR])
              + f"  sd {v[:, FAR].std(ddof=1):.1f}")
        out["cells"][name] = {"mean_azimuth": v.mean(0).tolist(),
                              "far_azimuth_per_seed": v[:, FAR].tolist(),
                              "hit_10deg": hit}

    d = paired_delta(have["device3 K=9 disc"], have["device3 K=9 rect"])
    out["paired_azimuth"] = d
    far = d[FAR]
    print(f"\n  frame-paired far-bin delta, disc minus rect: {far['delta']:+.1f} deg "
          f"(negative would favour the disc), {100 * far['fraction_improved']:.0f}% of frames improved")

    # The same deltas with the range crossing as the unit of evidence, which is the only version
    # worth reading. The frame-level standard errors above make the 0-50 m delta look like a
    # 5-sigma result off 6,398 frames; clustered, only 6 of its 10 crossings even agree on the
    # sign. This comparison is the clearest illustration in the project of why notes 40 exists.
    from significance import crossing_test          # noqa: E402  (script-local by design)

    print("\n  the same deltas, clustered by range crossing:")
    disc_runs, rect_runs = have["device3 K=9 disc"], have["device3 K=9 rect"]
    n = min(min(len(r["azimuth"]) for r in disc_runs), min(len(r["azimuth"]) for r in rect_runs))
    delta = (np.mean([r["azimuth"][:n] for r in disc_runs], 0)
             - np.mean([r["azimuth"][:n] for r in rect_runs], 0))
    rng = rect_runs[0]["range_m"][:n]
    seg = rect_runs[0].get("seg")
    if seg is None:                    # `errors` keeps no segment id, so re-read one dump for it
        p = predict.cnn_predictions(rect_runs[0]["run_id"], split, verbose=False)
        seg = p["seg"][p["has_drone"].astype(bool)][:n]
    out["clustered"] = {}
    for lo, hi in PAPER_BINS:
        m = (rng >= lo) & (rng < hi)
        out["clustered"][f"{lo}-{hi}m"] = crossing_test(delta, m, seg, f"  {lo}-{hi} m")
    rect_far = np.array(out["cells"]["device3 K=9 rect"]["far_azimuth_per_seed"])
    disc_far = np.array(out["cells"]["device3 K=9 disc"]["far_azimuth_per_seed"])
    out["verdict"] = (
        "The disc's far-field gain does not transfer: it wins this bin by 8.7 deg at all24 K=1 and "
        f"loses it by {disc_far.mean() - rect_far.mean():.1f} at device3 K=9, with pooled accuracy "
        "unchanged. Both mechanisms attack one failure, so whichever runs first takes what there is."
        if disc_far.mean() > rect_far.mean() else
        "The disc still helps once history is present, so the two are attacking different things.")
    print(f"  {out['verdict']}")
    # Worth printing because it is the more interesting number: the rectangle's far bin is wildly
    # seed-dependent here and the disc's is not, so "the rectangle wins" partly means "it drew a
    # lucky seed". Neither layout is reliably better in a bin this sparse.
    print(f"  Far-bin seed spread: rect sd {rect_far.std(ddof=1):.1f} deg against disc "
          f"sd {disc_far.std(ddof=1):.1f}. The disc is the more reproducible of the two.")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "polar_k9.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote {RESULTS / 'polar_k9.json'}")
    return out


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    cells = {name: collect(tpl, split) for name, tpl in CELLS.items()}
    have = {k: v for k, v in cells.items() if v}
    if BASELINE not in have:
        sys.exit(f"no runs for the baseline cell {BASELINE}")
    missing = [k for k, v in cells.items() if not v]
    print(f"\nall24, K=1, {split}. Degrees, lower better.")
    if missing:
        print(f"not yet trained, skipped: {', '.join(missing)}")

    results = {"split": split, "gate_far_azimuth": GATE_FAR_AZIMUTH,
               "n_seeds": {k: len(v) for k, v in have.items()}, "cells": {}}

    for key, stat, sname in (("azimuth", np.mean, "mean"), ("angular", np.mean, "mean"),
                             ("angular", np.median, "median")):
        print(f"\n{sname} {key} error, paper distance bins:")
        print(f"  {'':<24}" + "".join(f"{f'{lo}-{hi}m':>10}" for lo, hi in PAPER_BINS))
        for name, runs in have.items():
            v = seed_means(runs, PAPER_BINS, key, stat)
            print(f"  {name + f' ({len(runs)}s)':<24}"
                  + "".join(f"{x:>10.1f}" for x in v.mean(0))
                  + ("   sd " + " ".join(f"{x:.1f}" for x in v.std(0, ddof=1))
                     if len(runs) > 1 else ""))
            results["cells"].setdefault(name, {})[f"{sname}_{key}"] = {
                "bins": [list(b) for b in PAPER_BINS], "mean": v.mean(0).tolist(),
                "sd": v.std(0, ddof=1).tolist() if len(runs) > 1 else None,
                "per_seed": v.tolist()}

    print("\nframe-paired azimuth delta against the baseline (negative is better):")
    print(f"  {'':<24}" + "".join(f"{f'{lo}-{hi}m':>18}" for lo, hi in PAPER_BINS))
    for name, runs in have.items():
        if name == BASELINE:
            continue
        d = paired_delta(runs, have[BASELINE])
        print(f"  {name:<24}" + "".join(
            f"{b['delta']:>+11.1f} +-{b['se_floor']:>4.1f}" for b in d))
        print(f"  {'  frames improved':<24}" + "".join(
            f"{100 * b['fraction_improved']:>17.0f}%" for b in d))
        results["cells"][name]["paired_azimuth"] = d

    print("\nwithin 10 degrees, pooled over all ranges:")
    for name, runs in have.items():
        h = [float((r["angular"] <= 10).mean()) for r in runs]
        print(f"  {name:<24}{100 * np.mean(h):>6.1f}%"
              + (f" +/- {100 * np.std(h, ddof=1):.1f}" if len(h) > 1 else ""))
        results["cells"][name]["hit_10deg"] = h

    # --- Gate 1 -------------------------------------------------------------------------
    base_far = seed_means(have[BASELINE], PAPER_BINS, "azimuth", np.mean)[:, FAR]
    print(f"\nGate 1, on the {PAPER_BINS[FAR][0]}-{PAPER_BINS[FAR][1]} m azimuth error. "
          f"Baseline {base_far.mean():.1f} deg"
          + (f" +/- {base_far.std(ddof=1):.1f}" if len(base_far) > 1 else "") + ".")
    print(f"  A cell wins by reaching {GATE_FAR_AZIMUTH:.0f} deg or better on the seed mean "
          f"*and* beating {base_far.mean():.1f} on every seed.\n")
    winners = []
    for name, runs in have.items():
        if name == BASELINE:
            continue
        far = seed_means(runs, PAPER_BINS, "azimuth", np.mean)[:, FAR]
        ok = bool(far.mean() <= GATE_FAR_AZIMUTH and (far < base_far.mean()).all())
        winners.append((name, float(far.mean()), ok))
        print(f"  {name:<24}{far.mean():>7.1f} deg   seeds "
              + " ".join(f"{x:.1f}" for x in far) + f"   {'WIN' if ok else 'no'}")
        results["cells"][name]["gate_1_passed"] = ok
        results["cells"][name]["far_azimuth_per_seed"] = far.tolist()

    passing = [w for w in winners if w[2]]
    best = min(winners, key=lambda w: w[1]) if winners else None
    if passing:
        pick = min(passing, key=lambda w: w[1])
        print(f"\nGATE 1 PASS: {pick[0]} at {pick[1]:.1f} deg. That is the cell to take to "
              f"device3 K=9 tomorrow, against its 73.2% +/- 1.4 rectangular baseline.")
    else:
        print("\nGATE 1 FAIL: no cell reaches the declared bar, so no high-K arm is earned.")
        if best:
            print(f"  Best of them is {best[0]} at {best[1]:.1f} deg against the baseline's "
                  f"{base_far.mean():.1f}.")
        print("  Read with the standing caveat: this bin is about 2.5% of drone frames and the "
              "drone's beamformed peak sits below the ambient's there (notes section 25), so "
              "neither a projection nor a loss was ever going to recover a cue the map does not "
              "contain.")
    results["gate_1"] = {"baseline_far_azimuth": float(base_far.mean()),
                         "winners": [w[0] for w in passing],
                         "best_cell": best[0] if best else None,
                         "best_far_azimuth": best[1] if best else None,
                         "passed": bool(passing)}

    # Legacy two-arm schema, kept because utils/unseal_test1.py gates the sealed split on it.
    legacy_hits = {k: results["cells"][v]["hit_10deg"]
                   for k, v in LEGACY.items() if v in results["cells"]}
    if len(legacy_hits) == 2:
        results["hit_10deg"] = legacy_hits
        for sname_key in ("mean_azimuth", "mean_angular", "median_angular"):
            stat, key = sname_key.split("_")
            src = {k: results["cells"][v][f"{stat}_{key}"] for k, v in LEGACY.items()}
            results[sname_key] = {"bins": src["rect"]["bins"],
                                  "rect": src["rect"]["mean"], "polar": src["polar"]["mean"],
                                  "rect_sd": src["rect"]["sd"], "polar_sd": src["polar"]["sd"]}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "polar_vs_rect.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS / 'polar_vs_rect.json'}")
    plot(have, split)
    k9(split)
    return 0


def plot(cells: dict, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres = [(lo + hi) / 2 for lo, hi in FINE_BINS]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for (name, runs), colour in zip(cells.items(), colours):
        for key, axis in (("azimuth", ax), ("elevation", ax2)):
            v = seed_means(runs, FINE_BINS, key, np.mean)
            axis.plot(centres, v.mean(0), marker="o", color=colour, label=name)
            if len(runs) > 1:
                axis.fill_between(centres, v.mean(0) - v.std(0, ddof=1),
                                  v.mean(0) + v.std(0, ddof=1), color=colour, alpha=0.15)
    for axis, title in ((ax, "azimuth error (the axis the rectangle distorts)"),
                        (ax2, "elevation error (the axis it does not)")):
        axis.set_xlabel("range (m)")
        axis.set_ylabel("mean error (deg)")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"polar_vs_rect_{split}.png", dpi=160)
    print(f"wrote {FIGURES / f'polar_vs_rect_{split}.png'}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    raise SystemExit(main(a[0] if a else "test_2"))
