"""Stage E: does putting the map on a zenith disc fix the long-range azimuth failure?

The hypothesis, from design notes section 7 and the reproduction gap in section 23. Our map is a
rectangle of elevation by azimuth, which is a plate carree projection of the sky. Circular padding
fixed its azimuth seam, but nothing fixed the other defect: near the zenith, one azimuth cell
covers almost no solid angle, so a single learned kernel means different things at different
elevations. Our one reproduction gap is beyond 100 m, and it is almost entirely azimuth — 53
degrees against 11 in elevation. Azimuth is the axis the rectangle distorts.

The polar disc puts the zenith at the centre and the horizon on the rim, which is what a fisheye
lens does. A kernel then means the same thing everywhere, and azimuth resolution is *highest* at
the rim, which is where a distant drone sits. If the far-field azimuth failure is a projection
artefact, this is the change that fixes it.

What this script decides it on: mean azimuth error inside the 100-200 m bin, three seeds each,
against the rectangular runs in the same cell (all24, K=1) that our paper comparison is made in.
Mean rather than median, and the paper's own distance bins, so the number is comparable to the
table in section 5.1 of the report. Everything comes from the per-frame dumps, so no training
happens here.
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


def collect(polar: bool, split: str) -> list[dict]:
    tag = "_polar" if polar else ""
    got = []
    for s in SEEDS:
        run_id = f"all24_K1_w32_b16_wce{tag}_s{s}"
        if not (RUNS / run_id / "best.pt").exists():
            continue
        pred = predict.cnn_predictions(run_id, split, verbose=False)
        got.append({"run_id": run_id, **errors(pred)})
    return got


def table(name: str, runs: list[dict], bins, key: str, stat) -> np.ndarray:
    vals = np.array([by_bin(r, bins, key, stat) for r in runs])
    label = f"{name} ({len(runs)} seeds)"
    print(f"  {label:<22}" + "".join(f"{v:>10.1f}" for v in vals.mean(0)))
    if len(runs) > 1:
        print(f"  {'  seed sd':<22}" + "".join(f"{v:>10.1f}" for v in vals.std(0, ddof=1)))
    return vals


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    rect, polar = collect(False, split), collect(True, split)
    if not polar:
        sys.exit("no polar runs yet")
    print(f"\nall24, K=1, {split}. Degrees, lower better.\n")

    results = {"split": split, "n_seeds": {"rect": len(rect), "polar": len(polar)}}
    for key, stat, sname in (("azimuth", np.mean, "mean"), ("angular", np.mean, "mean"),
                             ("angular", np.median, "median")):
        print(f"{sname} {key} error, paper distance bins:")
        print(f"  {'':<22}" + "".join(f"{f'{lo}-{hi}m':>10}" for lo, hi in PAPER_BINS))
        r = table("rectangular", rect, PAPER_BINS, key, stat)
        p = table("polar disc", polar, PAPER_BINS, key, stat)
        print(f"  {'polar - rect':<22}" + "".join(f"{v:>+10.1f}" for v in
                                                  p.mean(0) - r.mean(0)))
        results[f"{sname}_{key}"] = {"bins": [list(b) for b in PAPER_BINS],
                                     "rect": r.mean(0).tolist(), "polar": p.mean(0).tolist(),
                                     "rect_sd": r.std(0, ddof=1).tolist() if len(rect) > 1
                                     else None,
                                     "polar_sd": p.std(0, ddof=1).tolist() if len(polar) > 1
                                     else None}
        print()

    hits = {"rect": [float((r["angular"] <= 10).mean()) for r in rect],
            "polar": [float((r["angular"] <= 10).mean()) for r in polar]}
    for k, v in hits.items():
        print(f"  within 10 deg, {k:<6}{100 * np.mean(v):>6.1f}%"
              + (f" +/- {100 * np.std(v, ddof=1):.1f}" if len(v) > 1 else ""))
    results["hit_10deg"] = hits

    far = {k: [np.mean(by_bin(r, PAPER_BINS, "azimuth")[2:]) for r in runs]
           for k, runs in (("rect", rect), ("polar", polar))}
    gap = np.mean(far["polar"]) - np.mean(far["rect"])
    spread = max(np.std(far["rect"], ddof=1) if len(rect) > 1 else 0.0,
                 np.std(far["polar"], ddof=1) if len(polar) > 1 else 0.0)
    print(f"\nVerdict on the 100-200 m azimuth error, which is the comparison this stage was "
          f"run for:\n  polar - rectangular = {gap:+.1f} deg against a seed spread of "
          f"{spread:.1f} deg.")
    print("  " + ("Inside the noise: the projection is not what the far-field failure was."
                  if abs(gap) < 2 * max(spread, 1e-9) else
                  ("Polar wins; the distortion was part of the problem." if gap < 0 else
                   "Polar is worse, which rules the projection out as the cause.")))
    print("\nEither way this bin is ~2.5% of drone frames and the drone's beamformed peak sits "
          "below the ambient there\n(notes section 25), so a projection change was never going "
          "to recover a cue that is not in the audio.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "polar_vs_rect.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS / 'polar_vs_rect.json'}")
    plot(rect, polar, split)
    return 0


def plot(rect: list[dict], polar: list[dict], split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres = [(lo + hi) / 2 for lo, hi in FINE_BINS]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, runs, colour in (("rectangular", rect, "tab:blue"),
                               ("polar disc", polar, "tab:orange")):
        if not runs:
            continue
        for key, axis in (("azimuth", ax), ("elevation", ax2)):
            v = np.array([by_bin(r, FINE_BINS, key, np.mean) for r in runs])
            axis.plot(centres, v.mean(0), marker="o", color=colour, label=name)
            if len(runs) > 1:
                axis.fill_between(centres, v.mean(0) - v.std(0, ddof=1),
                                  v.mean(0) + v.std(0, ddof=1), color=colour, alpha=0.2)
    for axis, title in ((ax, "azimuth error (the axis the rectangle distorts)"),
                        (ax2, "elevation error (the axis it does not)")):
        axis.set_xlabel("range (m)")
        axis.set_ylabel("mean error (deg)")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"polar_vs_rect_{split}.png", dpi=160)
    print(f"wrote {FIGURES / f'polar_vs_rect_{split}.png'}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    raise SystemExit(main(a[0] if a else "test_2"))
