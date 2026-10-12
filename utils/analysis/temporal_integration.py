"""Does integrating scores over time buy detection what frame stacking failed to buy direction?

Frame stacking hands the network five raw maps and hopes it learns something. Integration
averages the network's *output*. The second is far more constrained, but on this problem it has
the better statistical footing, for one specific reason: the presence label is constant across
a half-second window — a drone audible now was audible 300 ms ago — while the direction label
is not, because the target moves. Integration therefore averages repeated measurements of a
fixed quantity, which is the textbook case where averaging works, whereas stacking has to track
something that changes. So time can pay for detection even where it did not visibly pay for
localization.

Costs nothing: it is a post-hoc sweep over per-frame scores already on disk.

**The honest caveat, stated wherever these numbers appear.** Every segment in this dataset is
homogeneous — a flight segment is all drone, a mined ambient segment is all no-drone — so no
window here ever straddles a drone's onset. A deployed system pays a latency cost exactly at
onset, when the window is half full of silence. These numbers are therefore an upper bound on
what integration buys, and the dataset cannot measure the onset penalty at all.

That caveat is also why the sweep stops at three seconds. Pushed far enough, the window would
cover a whole segment and the task would degenerate into telling two recordings apart, which
would score beautifully and mean nothing. Thirty frames is about 1% of a segment, so the
averaging still happens inside a soundscape rather than across labels.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import FIGURES, FRAME_SECONDS, table_file       # noqa: E402
from src.metrics import auroc, specificity_at_recall            # noqa: E402
from src.predict import load                                    # noqa: E402

WINDOWS = (1, 2, 3, 5, 10, 20, 30)
SEEDS = (0, 1, 2)
MODELS = {"all24 K=1": "all24_K1_w32_b16_wce_s{s}",
          "device3 K=1": "device3_K1_w32_b16_wce_s{s}",
          "device3 K=5": "device3_K5_w32_b16_wce_s{s}",
          "best6 K=1": "best6_K1_w32_b16_wce_s{s}"}


def smooth_causal(score: np.ndarray, seg: np.ndarray, frame: np.ndarray,
                  window: int) -> np.ndarray:
    """Mean of the last `window` scores within each segment, expanding at segment starts.

    Expanding rather than dropping the first frames, so every frame still gets a decision and
    the frame count stays identical across windows — otherwise a longer window would quietly
    change the denominator of every rate it is compared on.
    """
    out = np.empty_like(score, dtype=float)
    order = np.lexsort((frame, seg))
    for s in np.unique(seg):
        idx = order[seg[order] == s]
        x = np.asarray(score, float)[idx]
        c = np.concatenate([[0.0], np.cumsum(x)])
        lo = np.maximum(np.arange(len(x)) + 1 - window, 0)
        out[idx] = (c[1:] - c[lo]) / (np.arange(len(x)) + 1 - lo)
    return out


def n_of_m(score: np.ndarray, seg: np.ndarray, frame: np.ndarray, thr: float,
           window: int) -> np.ndarray:
    """Fraction of the last `window` frames above `thr`, within each segment."""
    return smooth_causal((np.asarray(score, float) >= thr).astype(float), seg, frame, window)


def threshold_for_count(score: np.ndarray, negatives: np.ndarray, count: int) -> float:
    neg = np.sort(np.asarray(score, float)[negatives])[::-1]
    return -np.inf if count >= len(neg) else float(np.nextafter(neg[count], np.inf))


def recall_at_count(score: np.ndarray, drone: np.ndarray, count: int) -> float:
    thr = threshold_for_count(score, ~drone, count)
    return float((score[drone] >= thr).mean())


def sweep(pred: dict, neg_hours: float) -> list[dict]:
    d = pred["has_drone"].astype(bool)
    rows = []
    for w in WINDOWS:
        s = smooth_causal(pred["score"], pred["seg"], pred["frame"], w)
        rows.append({"window": w, "latency_s": (w - 1) * FRAME_SECONDS,
                     "auroc": auroc(s, d),
                     "recall_at_0_fa": recall_at_count(s, d, 0),
                     "recall_at_1_fa": recall_at_count(s, d, 1),
                     "reject_at_95_recall": specificity_at_recall(s, d)})
    return rows


def vote_sweep(pred: dict, count: int = 1) -> list[dict]:
    """Majority and unanimous voting on per-frame decisions, as an alternative to averaging.

    A vote is what a deployed alarm actually does, and it is not the same as averaging: one
    enormous score cannot carry a window on its own.
    """
    d = pred["has_drone"].astype(bool)
    thr = threshold_for_count(pred["score"], ~d, count)
    rows = []
    for w in WINDOWS:
        frac = n_of_m(pred["score"], pred["seg"], pred["frame"], thr, w)
        for rule, need in (("majority", np.ceil(w / 2) / w), ("unanimous", 1.0)):
            fired = frac >= need - 1e-9
            n_fa = int(fired[~d].sum())
            rows.append({"window": w, "rule": rule, "recall": float(fired[d].mean()),
                         "false_alarms": n_fa})
    return rows


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    results = {"split": split, "windows": list(WINDOWS), "models": {}}
    n_neg = neg_hours = None

    for name, template in MODELS.items():
        preds = []
        for s in SEEDS:
            try:
                preds.append(load(f"{template.format(s=s)}_{split}_causal"))
            except FileNotFoundError:
                pass
        if not preds:
            continue
        if n_neg is None:
            n_neg = int((~preds[0]["has_drone"].astype(bool)).sum())
            neg_hours = n_neg * FRAME_SECONDS / 3600.0
            print(f"\n{split}: {n_neg} no-drone frames = {n_neg * FRAME_SECONDS / 60:.1f} min. "
                  f"One false alarm is {1 / neg_hours:.1f}/h.")
            print("Every segment is homogeneous, so no window straddles a drone onset: these "
                  "are upper bounds.\n")

        sweeps = [sweep(p, neg_hours) for p in preds]
        votes = [vote_sweep(p) for p in preds]
        results["models"][name] = {"average": sweeps, "vote": votes,
                                   "runs": [template.format(s=s) for s in SEEDS]}

        print(f"{name}: causal moving average of the presence score, {len(preds)} seeds")
        print(f"  {'window':<9}{'latency':>9}{'AUROC':>10}{'recall @0 FA':>14}"
              f"{'recall @1 FA':>14}{'reject @95%':>13}")
        for i, w in enumerate(WINDOWS):
            col = lambda k: np.mean([s[i][k] for s in sweeps])                # noqa: E731
            sd = lambda k: np.std([s[i][k] for s in sweeps], ddof=1) if len(sweeps) > 1 else 0.0
            print(f"  {w:<9}{(w - 1) * FRAME_SECONDS:>8.1f}s{col('auroc'):>10.3f}"
                  f"{col('recall_at_0_fa') * 100:>13.1f}%{col('recall_at_1_fa') * 100:>13.1f}%"
                  f"{col('reject_at_95_recall') * 100:>12.1f}%"
                  + (f"   (AUROC sd {sd('auroc'):.3f})" if len(sweeps) > 1 else ""))
        best = max(range(len(WINDOWS)), key=lambda i: np.mean([s[i]["auroc"] for s in sweeps]))
        gain = (np.mean([s[best]["auroc"] for s in sweeps])
                - np.mean([s[0]["auroc"] for s in sweeps]))
        print(f"  best window {WINDOWS[best]} ({(WINDOWS[best] - 1) * FRAME_SECONDS:.1f}s "
              f"latency): AUROC {gain:+.3f} over single-frame")
        # Integration is a monotone gain within every seed, but the seeds themselves are far
        # apart, so printing the individual values keeps the mean from implying a stable model.
        # This spread is the strongest argument for the separate per-channel detector: the
        # map model's presence head is a 128-unit afterthought trained on 3 negative scenes.
        per_seed = [(s[0]["auroc"], s[best]["auroc"]) for s in sweeps]
        print("  per seed, single frame -> best window: "
              + ",  ".join(f"{a:.3f} -> {b:.3f}" for a, b in per_seed) + "\n")

    path = table_file("temporal_integration.json")
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
    for name, d in results["models"].items():
        lat = [r["latency_s"] for r in d["average"][0]]
        au = np.mean([[r["auroc"] for r in s] for s in d["average"]], 0)
        rec = np.mean([[r["recall_at_1_fa"] for r in s] for s in d["average"]], 0)
        ax.plot(lat, au, marker="o", label=name)
        ax2.plot(lat, 100 * rec, marker="o", label=name)
    for a, ylab, title in ((ax, "detection AUROC", "integrating the score over time"),
                           (ax2, "recall at 1 false alarm (%)", "recall at a fixed budget")):
        a.set_xlabel("decision latency (s)")
        a.set_ylabel(ylab)
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"temporal_integration_{split}.png", dpi=160)
    print(f"wrote {FIGURES / f'temporal_integration_{split}.png'}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test_2"))
