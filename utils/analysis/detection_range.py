"""How far away can we hear a drone, and how many labelled positives carry a usable cue?

Two questions that a pooled AUROC cannot answer, and one of them undermines the other.

**Detection range.** An operator does not want an AUROC, they want "detects out to X metres at
Y false alarms per hour". That needs a threshold fixed on the no-drone frames alone and then
recall measured inside each range bin. The reference paper never reports it.

**Energy separability of the positive class.** Beyond 100 m the drone's beamformed peak energy
sits inside the ambient distribution, so those frames are labelled "drone present" while
carrying no evidence *an energy detector could use*. Quantifying that separates "our detector is
weak" from "the frame does not contain a usable cue".

Note the careful wording: energy-inseparable is not the same as inaudible. The network still
localizes many of these frames correctly, so there is spatial structure in them that a scalar
energy statistic throws away. Two references are reported, because "inaudible" has no single
definition: below the negatives' **median** is the honest reading of "quieter than a typical
quiet frame", while below their **95th percentile** only means "uncatchable by an energy
threshold that fires on 5% of ambient" — and since the two distributions overlap heavily, that
second figure is near 90% for reasons that are mostly about the extremity of the quantile.

Audibility is measured with **absolute beamformed peak energy**, not peak-to-mean contrast.
Both are in the baseline dumps, and contrast is at or below chance as a detector on this data
(AUROC 0.455 on all24, 0.549 on device3, against 0.649 and 0.689 for peak energy), so it would
be a proxy for nothing. Peak energy is also what design notes section 23 used, which keeps one
definition of audibility across the whole report.

A hard limit to keep in view: test 2 holds only minutes of no-drone audio, so the finest
non-zero false-alarm rate that exists is one alarm over that duration. Budgets below it cannot
be measured, only asserted.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import FIGURES, FRAME_SECONDS, run_id, table_file  # noqa: E402
from src.geometry import angular_distance                       # noqa: E402
from src.metrics import DISTANCE_BINS, threshold_for_count      # noqa: E402
from src.predict import load                                    # noqa: E402

# Counts rather than rates, because with a fixed amount of negative audio only integer counts
# of false alarms are achievable and quoting "1 per hour" would imply a resolution we lack.
FA_COUNTS = (0, 1, 5, 10, 25)
SEEDS = (0, 1, 2)
MODELS = {"all24 K=1": run_id("all24", 1, seed="{s}"),
          "all24 K=5": run_id("all24", 5, seed="{s}"),
          "device3 K=1": run_id("device3", 1, seed="{s}"),
          "device3 K=5": run_id("device3", 5, seed="{s}")}


def bin_edges(bins) -> list[tuple[float, float]]:
    return list(zip(bins[:-1], bins[1:]))


def trim_empty(pred: dict, bins=DISTANCE_BINS) -> tuple:
    """Drop trailing bins with no drone frames, so averages do not warn about empty slices.

    The dataset's flights top out around 170 m, so the open-ended bin is always empty here,
    but it is trimmed from the data rather than hard-coded in case a split reaches further.
    """
    d = pred["has_drone"].astype(bool)
    r = np.asarray(pred["range_m"], float)
    keep = [0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        if (d & (r >= lo) & (r < hi)).any():
            keep.append(hi)
    return tuple(keep)


def label(lo: float, hi: float) -> str:
    return f"{int(lo)}-{'inf' if not np.isfinite(hi) else int(hi)}m"


def recall_by_range(pred: dict, thr: float, bins) -> list[dict]:
    d = pred["has_drone"].astype(bool)
    fired = pred["score"] >= thr
    r = np.asarray(pred["range_m"], float)
    out = []
    for lo, hi in bin_edges(bins):
        m = d & (r >= lo) & (r < hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "recall": float(fired[m].mean()) if m.any() else float("nan")})
    return out


def audibility(pred: dict, bins) -> dict:
    """Fraction of drone frames whose beamformed peak is inside the ambient distribution."""
    d = pred["has_drone"].astype(bool)
    peak = np.asarray(pred["peak_db"], float)
    neg_median, neg_p95 = np.median(peak[~d]), np.quantile(peak[~d], 0.95)
    r = np.asarray(pred["range_m"], float)
    rows = []
    for lo, hi in bin_edges(bins):
        m = d & (r >= lo) & (r < hi)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "below_median": float((peak[m] < neg_median).mean()) if m.any() else
                     float("nan"),
                     "below_p95": float((peak[m] < neg_p95).mean()) if m.any() else
                     float("nan")})
    return {"neg_median_db": float(neg_median), "neg_p95_db": float(neg_p95),
            "n_negative": int((~d).sum()), "by_range": rows,
            "below_p95_overall": float((peak[d] < neg_p95).mean())}


def localization_when_audible(pred: dict, neg_p95: float, bins) -> dict:
    """Direction accuracy over all positives and over the energy-separable ones only.

    Always a pair: the loud subset alone selects the frames we like, the pooled number alone
    credits frames with no cue. The loud frames are the near ones, where the network is worst
    (plane-wave steering breaks in the near field); see docs/design_notes.md section 25.
    """
    d = pred["has_drone"].astype(bool)
    err = angular_distance(pred["pred_az"][d], pred["pred_el"][d],
                           pred["az"][d], pred["el"][d])
    loud = np.asarray(pred["peak_db"], float)[d] >= neg_p95
    r = np.asarray(pred["range_m"], float)[d]
    out = {"n": int(d.sum()), "n_audible": int(loud.sum()),
           "hit_10deg_all": float((err <= 10).mean()),
           "hit_10deg_audible": float((err[loud] <= 10).mean()) if loud.any()
           else float("nan"),
           "median_all": float(np.median(err)),
           "median_audible": float(np.median(err[loud])) if loud.any() else float("nan"),
           "by_range": []}
    for lo, hi in bin_edges(bins):
        m = (r >= lo) & (r < hi)
        out["by_range"].append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                                "hit_10deg": float((err[m] <= 10).mean()) if m.any()
                                else float("nan")})
    return out


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    base = {s: load(f"baseline_{s}_{split}") for s in ("all24", "device3")}
    bins = trim_empty(base["all24"])
    n_neg = int((~base["all24"]["has_drone"].astype(bool)).sum())
    neg_hours = n_neg * FRAME_SECONDS / 3600.0
    print(f"\n{split}: {n_neg} no-drone frames = {n_neg * FRAME_SECONDS / 60:.1f} min, so one "
          f"false alarm is {1 / neg_hours:.1f}/h and nothing finer is measurable.\n")

    # --- audibility ---------------------------------------------------------------------
    results = {"split": split, "n_negative": n_neg, "bins": list(bins),
               "audibility": {}, "detection": {}, "localization": {}}
    for subset, pred in base.items():
        a = audibility(pred, bins)
        results["audibility"][subset] = a
        print(f"Energy separability on {subset}: ambient peak median {a['neg_median_db']:.1f} dB,"
              f" 95th percentile {a['neg_p95_db']:.1f} dB")
        print(f"  {'range':<10}{'frames':>8}{'quieter than typical ambient':>30}"
              f"{'uncatchable at 5% FA':>23}")
        for row in a["by_range"]:
            print(f"  {label(row['lo'], row['hi']):<10}{row['n']:>8}"
                  f"{row['below_median'] * 100:>28.0f}%{row['below_p95'] * 100:>22.0f}%")
        near, far = a["by_range"][0], a["by_range"][3]
        print(f"  the cue decays with range: {near['below_median'] * 100:.0f}% of frames under "
              f"{int(near['hi'])} m are quieter than typical ambient, against "
              f"{far['below_median'] * 100:.0f}% beyond {int(far['lo'])} m\n")

    # --- detection range ---------------------------------------------------------------
    for name, template in MODELS.items():
        runs = [template.format(s=s) for s in SEEDS]
        preds = []
        for run in runs:
            try:
                preds.append(load(f"{run}_{split}_causal"))
            except FileNotFoundError:
                pass
        if not preds:
            continue
        subset = "all24" if name.startswith("all24") else "device3"
        peak = np.asarray(base[subset]["peak_db"], float)
        # Same frames, so the baseline's audibility column can be joined onto a model dump by
        # (seg, frame) rather than assumed to be in the same order.
        key = {(s, f): i for i, (s, f) in enumerate(zip(base[subset]["seg"],
                                                        base[subset]["frame"]))}
        print(f"{name}: recall by range, threshold set on no-drone frames only")
        print(f"  {'budget':<12}" + "".join(f"{label(lo, hi):>10}" for lo, hi in bin_edges(bins)))
        rows = {}
        for count in FA_COUNTS:
            curves = []
            for pred in preds:
                thr = threshold_for_count(pred["score"], ~pred["has_drone"].astype(bool),
                                          count)
                curves.append([b["recall"] for b in recall_by_range(pred, thr, bins)])
            mean = np.nanmean(curves, 0)
            rows[count] = {"per_hour": count / neg_hours, "recall": mean.tolist(),
                           "sd": np.nanstd(curves, 0, ddof=1).tolist() if len(curves) > 1
                           else None}
            print(f"  {f'{count / neg_hours:.0f}/h':<12}"
                  + "".join(f"{v * 100:>9.0f}%" for v in mean))
        results["detection"][name] = {"runs": runs, "by_budget": rows,
                                      "n_bins": [b["n"] for b in recall_by_range(preds[0], 0, bins)]}

        # Per-frame recall reads as a catastrophe and is the wrong unit for an alarm. A drone
        # overflies for tens of seconds, so what matters is how long until the first frame
        # fires: at 10 frames per second, 20% recall means an alarm within half a second on
        # average. The rate is only meaningful where recall is non-negligible.
        mid = np.array(rows[10]["recall"], float)[:3]
        wait = FRAME_SECONDS / np.maximum(mid, 1e-9)
        print(f"  at {rows[10]['per_hour']:.0f} false alarms/h, mean wait for a first alarm "
              f"while the drone is present: "
              + ", ".join(f"{label(*b)} {w:.1f}s" for b, w in zip(bin_edges(bins)[:3], wait)))

        loc = []
        for pred in preds:
            idx = np.array([key[(s, f)] for s, f in zip(pred["seg"], pred["frame"])])
            loc.append(localization_when_audible({**pred, "peak_db": peak[idx]},
                                                 results["audibility"][subset]["neg_p95_db"],
                                                 bins))
        results["localization"][name] = loc
        by_range = np.array([[b["hit_10deg"] for b in l["by_range"]] for l in loc])
        print(f"  within 10 deg: {np.mean([l['hit_10deg_all'] for l in loc]) * 100:.1f}% over "
              f"all positives, {np.mean([l['hit_10deg_audible'] for l in loc]) * 100:.1f}% over "
              f"the loudest {np.mean([l['n_audible'] / l['n'] for l in loc]) * 100:.0f}% — the "
              f"loud frames are the near ones, and the near field is where we are worst")
        print(f"  by range:    "
              + "  ".join(f"{label(*b)} {v * 100:.0f}%"
                          for b, v in zip(bin_edges(bins), np.nanmean(by_range, 0))) + "\n")

    path = table_file("detection_range.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")
    plot(results, split)
    return 0


def plot(results: dict, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = results["bins"]
    centres = [(lo + (hi if np.isfinite(hi) else lo + 60)) / 2 for lo, hi in bin_edges(bins)]
    budget = 10
    rate = None
    for name, d in results["detection"].items():
        # str() because this is also called on a reloaded JSON, where int keys came back as text
        r = d["by_budget"].get(budget) or d["by_budget"][str(budget)]
        rate = r["per_hour"]
        ax.plot(centres, 100 * np.array(r["recall"], float), marker="o", label=name)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("per-frame recall (%)")
    ax.set_title(f"detection vs range at {rate:.0f} false alarms/h")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Per-frame recall of 30% looks like failure and is the wrong unit: frames arrive at 10 Hz
    # and an overflight lasts tens of seconds, so the operational quantity is the wait for the
    # first alarm. The right-hand axis converts one into the other. It assumes independent
    # frames, which is optimistic — hence the label — but the qualitative point survives any
    # correlation: under 90 m the wait is a second, beyond it there is no alarm at all.
    seconds = np.array([0.1, 0.2, 0.5, 1.0, 5.0])
    ax_s = ax.twinx()
    ax_s.set_ylim(0, 105)
    ax_s.set_yticks(100 * FRAME_SECONDS / seconds)
    ax_s.set_yticklabels([f"{s:g}" for s in seconds], fontsize=8)
    ax_s.set_ylabel("s to first alarm (independent frames)", fontsize=8)

    # below_median, not below_p95: the 95th percentile of the ambient is such an extreme
    # threshold that nearly every drone frame sits under it at every range, so plotting it
    # produces a flat line that says more about the quantile than about the drone.
    for subset, a in results["audibility"].items():
        ax2.plot(centres, [100 * r["below_median"] for r in a["by_range"]], marker="s",
                 label=subset)
    ax2.set_xlabel("range (m)")
    ax2.set_ylabel("quieter than typical ambient (%)")
    ax2.set_title("labelled drone frames below the ambient median")
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"detection_range_{split}.png", dpi=160)
    print(f"wrote {FIGURES / f'detection_range_{split}.png'}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test_2"))
