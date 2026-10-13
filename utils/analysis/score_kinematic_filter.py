"""The kinematic filter, and the two readings of angular error that must not be mixed.

The filter keeps a stored direction and takes a new decode at once when it lies within a
fixed 10° of the stored one. A larger jump is taken only when the last five raw directions
all agree with it. The cap is the same angle at every range; GPS range is never an input to
the decision, only to the scoring afterwards. Runs on dumps in results/preds/; nothing trains.

Two summaries, and they do not convert into each other. **External** is the paper's: mean
angular error in their 0-50 / 50-100 / 100-200 m bins, U-Net rows on stable frames only.
**Internal** is ours: every drone frame, share inside 10° in 30 m bins, and the sideways
miss range * sin(error).

    .venv/bin/python utils/analysis/score_kinematic_filter.py
    .venv/bin/python utils/analysis/score_kinematic_filter.py --split test_1
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import AZ, EL, pred_file, run_id                     # noqa: E402
from src.geometry import angular_distance, to_angles, unit_vectors  # noqa: E402
from utils.analysis.check_vs_paper import PAPER, PAPER_BINS, binned_mean, stability_flags  # noqa: E402

# One sky-angle cap at every range. Five frames is the half-second a jump must hold.
CAP_DEG = 10.0
K_AGREE = 5

OUR_BINS = ((0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 120.0), (120.0, 180.0))
HIT_DEG = 10.0
SD_SHOW = 1.5

# test_2 only. Rounded 3-seed mean of the overall share inside 10°, raw then filtered with
# the fixed 10° cap, and the delay-and-sum paper-bin means. A regression pin, not a fit.
GATE_HIT = {
    "all24_K1_wce": (70, 75),
    "device3_K1_wce": (69, 73),
    "device3_K9_wce": (73, 73),
}
GATE_DAS = (40, 72, 31)

# label, kind, spec, gate key. kind is baseline / seeds / once.
# A seeds spec is a run id with {s}; a once spec is the seed-0 run id.
VARIANTS = [
    ("Delay-and-sum, 24 mics", "baseline", "all24", "baseline_all24"),
    ("Delay-and-sum, device 3", "baseline", "device3", ""),
    ("Delay-and-sum, searched 6", "baseline", "best6", ""),
    ("24 mics, K=1", "seeds", run_id("all24", 1, seed="{s}"), "all24_K1_wce"),
    ("24 mics, K=5", "seeds", run_id("all24", 5, seed="{s}"), ""),
    ("24 mics, K=1, Tversky", "seeds", run_id("all24", 1, loss="tversky", seed="{s}"), ""),
    ("24 mics, K=1, disc resampled", "seeds", run_id("all24", 1, polar=True, seed="{s}"), ""),
    ("24 mics, K=1, disc native", "seeds",
     run_id("all24", 1, polar=True, disc="native", seed="{s}"), ""),
    ("24 mics, K=1, native disc + Tversky", "seeds",
     run_id("all24", 1, loss="tversky", polar=True, disc="native", seed="{s}"), ""),
    ("Device 3, K=1", "seeds", run_id("device3", 1, seed="{s}"), "device3_K1_wce"),
    ("Device 3, K=5", "seeds", run_id("device3", 5, seed="{s}"), ""),
    ("Device 3, K=9", "seeds", run_id("device3", 9, seed="{s}"), "device3_K9_wce"),
    ("Device 3, K=9, disc native", "seeds",
     run_id("device3", 9, polar=True, disc="native", seed="{s}"), ""),
    ("Searched 6, K=1", "seeds", run_id("best6", 1, seed="{s}"), ""),
    ("Searched 6, K=5", "seeds", run_id("best6", 5, seed="{s}"), ""),
    ("Device 1+3, K=1", "once", run_id("device1_3", 1), ""),
    ("Device 1, K=1", "once", run_id("device1", 1), ""),
    ("Device 4, K=1", "once", run_id("device4", 1), ""),
    ("Symmetric 6, K=1", "once", run_id("symmetric6", 1), ""),
    ("Quad 4, K=1", "once", run_id("quad4", 1), ""),
]

NAME_W = 40


def apply_filter(az: np.ndarray, el: np.ndarray):
    """One segment. The stored direction starts at frame 0 and resets with the caller.

    The cap is a fixed sky angle. Range is not read here.
    """
    vec = unit_vectors(az, el)
    out = np.empty_like(vec)
    stored = vec[0].copy()
    out[0] = stored
    limit = float(np.cos(np.radians(CAP_DEG)))
    for t in range(1, len(vec)):
        new = vec[t]
        if float(np.dot(stored, new)) >= limit:
            stored = new.copy()
        else:
            start = t - K_AGREE + 1
            if start >= 0 and bool(np.all(vec[start:t + 1] @ new >= limit)):
                stored = new.copy()
        out[t] = stored
    return to_angles(out)


def decoded_directions(dump: dict) -> tuple[np.ndarray, np.ndarray]:
    """Brightest cell. Rectangular indices are the 4° grid; a disc dump is not."""
    az_idx = np.asarray(dump["az_idx"])
    el_idx = np.asarray(dump["el_idx"])
    if az_idx.max() <= 90 and el_idx.max() <= 22:
        return np.asarray(AZ)[az_idx].astype(float), np.asarray(EL)[el_idx].astype(float)
    return np.asarray(dump["argmax_az"], float), np.asarray(dump["argmax_el"], float)


def drone_frames(dump: dict) -> dict:
    """Drone segments only, in time order, with the filter reset at each segment."""
    pred_az, pred_el = decoded_directions(dump)
    seg = np.asarray(dump["seg"])
    frame = np.asarray(dump["frame"])
    pieces = []
    for i, name in enumerate(dump["seg_ids"]):
        if not str(name).endswith("_drone"):
            continue
        idx = np.flatnonzero(seg == i)
        idx = idx[np.argsort(frame[idx], kind="mergesort")]
        keep = dump["has_drone"][idx].astype(bool) & np.isfinite(dump["az"][idx])
        idx = idx[keep]
        if len(idx) == 0:
            continue
        paz = np.asarray(pred_az[idx], float)
        pel = np.asarray(pred_el[idx], float)
        rng = np.asarray(dump["range_m"][idx], float)
        faz, fel = apply_filter(paz, pel)
        pieces.append({
            "az": np.asarray(dump["az"][idx], float),
            "el": np.asarray(dump["el"][idx], float),
            "range_m": rng,
            "pred_az": paz,
            "pred_el": pel,
            "filt_az": faz,
            "filt_el": fel,
            "stable": stability_flags(paz, pel),
        })
    if not pieces:
        raise ValueError("no drone frames")
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def _our_masks(range_m: np.ndarray) -> list[np.ndarray]:
    masks = [(range_m >= lo) & (range_m < hi) for lo, hi in OUR_BINS]
    masks.append(np.ones(len(range_m), dtype=bool))
    return masks


def _hit(err: np.ndarray, range_m: np.ndarray) -> np.ndarray:
    out = []
    for m in _our_masks(range_m):
        out.append(100.0 * float(np.mean(err[m] <= HIT_DEG)) if m.any() else np.nan)
    return np.asarray(out, float)


def _miss(err: np.ndarray, range_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sideways miss in metres: range * sin(error). Median, then mean, per bin."""
    miss = range_m * np.sin(np.radians(err))
    meds, means = [], []
    for m in _our_masks(range_m):
        x = miss[m]
        if len(x) == 0:
            meds.append(np.nan)
            means.append(np.nan)
        else:
            meds.append(float(np.median(x)))
            means.append(float(np.mean(x)))
    return np.asarray(meds, float), np.asarray(means, float)


def _score_arrays(dump: dict) -> dict:
    fr = drone_frames(dump)
    raw = angular_distance(fr["pred_az"], fr["pred_el"], fr["az"], fr["el"])
    filt = angular_distance(fr["filt_az"], fr["filt_el"], fr["az"], fr["el"])
    r = fr["range_m"]
    st = fr["stable"]
    med_raw, mean_raw = _miss(raw, r)
    med_filt, mean_filt = _miss(filt, r)
    return {
        "ext_raw": np.asarray([m for m, _ in binned_mean(raw, r)], float),
        "ext_stable": np.asarray([m for m, _ in binned_mean(raw[st], r[st])], float),
        "ext_filt": np.asarray([m for m, _ in binned_mean(filt, r)], float),
        "hit_raw": _hit(raw, r),
        "hit_filt": _hit(filt, r),
        "med_raw": med_raw,
        "med_filt": med_filt,
        "mean_raw": mean_raw,
        "mean_filt": mean_filt,
    }


def load_dump(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def score_path(path: Path) -> dict:
    return _score_arrays(load_dump(path))


def variant_paths(kind: str, spec: str, split: str) -> list[Path]:
    if kind == "baseline":
        candidates = [pred_file(f"baseline_{spec}_{split}")]
    elif kind == "once":
        candidates = [pred_file(f"{spec}_{split}_causal")]
    else:
        candidates = [pred_file(f"{spec.format(s=s)}_{split}_causal") for s in (0, 1, 2)]
    return [p for p in candidates if p.exists()]


def collect(split: str) -> list[dict]:
    rows = []
    for label, kind, spec, key in VARIANTS:
        paths = variant_paths(kind, spec, split)
        if not paths:
            continue
        scored = [score_path(p) for p in paths]
        row = {"label": label, "key": key, "n": len(scored)}
        for field in scored[0]:
            row[field] = np.vstack([s[field] for s in scored])
        rows.append(row)
    return rows


def _mu_sd(col: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(col))
    sd = float(np.std(col, ddof=1)) if len(col) > 1 else float("nan")
    return mu, sd


def fmt_val(mu: float, sd: float, digits: int) -> str:
    if not np.isfinite(mu):
        return "—"
    text = f"{mu:.{digits}f}"
    if np.isfinite(sd) and sd >= SD_SHOW:
        text += f" ± {sd:.{digits}f}"
    return text


def fmt_arrow(a: np.ndarray, b: np.ndarray, digits: int) -> str:
    mu_a, sd_a = _mu_sd(a)
    mu_b, sd_b = _mu_sd(b)
    return f"{fmt_val(mu_a, sd_a, digits)} → {fmt_val(mu_b, sd_b, digits)}"


def _name(row: dict) -> str:
    return row["label"] + (" *" if row["n"] == 1 else "")


def _print_row(name: str, cells: list[str], width: int) -> None:
    print(f"  {name:<{NAME_W}}" + "".join(f"{c:>{width}}" for c in cells))


def _table(title: str, labels: list[str], body: list[tuple[str, list[str]]]) -> None:
    width = max(len(lab) for lab in labels)
    if body:
        width = max(width, max(len(c) for _, cells in body for c in cells))
    width += 2
    print(f"\n  {title:<{NAME_W}}" + "".join(f"{lab:>{width}}" for lab in labels))
    for name, cells in body:
        _print_row(name, cells, width)


def print_external(split: str, rows: list[dict]) -> None:
    print("\nEXTERNAL — the paper's metric. Only this block may be compared with the paper.")
    print("Mean angular error in degrees, bins 0-50 / 50-100 / 100-200 m.")
    print("Raw is every drone frame. Stable is their three-frame rule on the raw directions.")
    print("Filtered is the kinematic filter, still on every frame.")
    print("Paper delay-and-sum is an every-frame bar, so it sits in the raw column.")
    print("Paper U-Net is a stable-frame bar. A star is one seed. ± is the seed sd,")
    print(f"shown only when it is at least {SD_SHOW:.1f}.")
    labels = [f"{lo:.0f}-{hi:.0f} m" for lo, hi in zip(PAPER_BINS[:-1], PAPER_BINS[1:])]
    width = 12
    print(f"\n  {'':<{NAME_W}}" + "".join(f"{lab:^{width * 3}}" for lab in labels))
    print(f"  {'':<{NAME_W}}" + ("{:>12}{:>12}{:>12}".format("raw", "stable", "filt") * 3))

    def paper_cells(values, slot: int) -> list[str]:
        cells = []
        for v in values:
            triple = ["—", "—", "—"]
            triple[slot] = f"{v:.0f}"
            cells.extend(triple)
        return cells

    _print_row("Paper, delay-and-sum", paper_cells(PAPER[("beamforming", split)], 0), width)
    _print_row("Paper, U-Net", paper_cells(PAPER[("unet", split)], 1), width)
    for row in rows:
        cells = []
        for b in range(3):
            for field in ("ext_raw", "ext_stable", "ext_filt"):
                mu, sd = _mu_sd(row[field][:, b])
                cells.append(fmt_val(mu, sd, 0))
        _print_row(_name(row), cells, width)


def print_internal(rows: list[dict]) -> None:
    print("\nINTERNAL — our metric. Not for comparison with the paper.")
    print("Every drone frame. Share with error <= 10°, bins 0-30 / 30-60 / 60-90 /")
    print("90-120 / 120-180 m and all. Arrow is raw → filtered. Then the sideways miss")
    print("in metres, range * sin(error), median and mean in the same bins.")
    labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in OUR_BINS] + ["all"]
    _table("hit rate, %", labels, [
        (_name(row), [fmt_arrow(row["hit_raw"][:, b], row["hit_filt"][:, b], 0) for b in range(6)])
        for row in rows])
    for title, raw_key, filt_key in (
            ("sideways miss, median, m", "med_raw", "med_filt"),
            ("sideways miss, mean, m", "mean_raw", "mean_filt")):
        _table(title, labels, [
            (_name(row), [fmt_arrow(row[raw_key][:, b], row[filt_key][:, b], 1)
                          for b in range(6)])
            for row in rows])


def gate(rows: list[dict]) -> int:
    by_key = {r["key"]: r for r in rows if r["key"]}
    ok = True
    print("\nGate on test_2, values rounded to the nearest integer.")
    for key, (exp_raw, exp_fil) in GATE_HIT.items():
        row = by_key.get(key)
        if row is None:
            print(f"  {key}: missing")
            ok = False
            continue
        raw = int(round(float(row["hit_raw"][:, -1].mean())))
        fil = int(round(float(row["hit_filt"][:, -1].mean())))
        match = (raw, fil) == (exp_raw, exp_fil)
        ok = ok and match
        print(f"  {key} overall hit@10  {raw} / {fil}   expected {exp_raw} / {exp_fil}")
    das = by_key.get("baseline_all24")
    if das is None:
        print("  baseline_all24: missing")
        ok = False
    else:
        got = tuple(int(round(float(x))) for x in das["ext_raw"].mean(0))
        match = got == GATE_DAS
        ok = ok and match
        print(f"  DAS all24 paper-bin raw means  {got[0]}/{got[1]}/{got[2]}"
              f"   expected {GATE_DAS[0]}/{GATE_DAS[1]}/{GATE_DAS[2]}")
    print("\nGATE PASS" if ok else "\nGATE FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", default="test_2", choices=("test_1", "test_2"))
    args = parser.parse_args(argv)
    print(__doc__.split("\n\n")[0])
    print(f"\n{args.split}. Dumps in results/preds/. Missing runs are skipped.")
    rows = collect(args.split)
    print_external(args.split, rows)
    print_internal(rows)
    if args.split != "test_2":
        return 0
    return gate(rows)


if __name__ == "__main__":
    raise SystemExit(main())
