"""The report's figure, assembled from the analysis JSONs rather than recomputed.

Each analysis script writes its own diagnostic figure, which is right for reading a result and
wrong for a five-page report: four separate figures cost a page and a half in captions and
whitespace, and shrinking them to fit makes the axes unreadable, which is worse than having no
figure at all. So this builds one 2x2 panel sized for the page it will actually be printed on --
about 7 inches wide at 9 pt labels -- and takes every number from `results/*.json`, so a panel can
never disagree with the table beside it.

The polar comparison is deliberately not here. Its result is a null stated in one line of the
limitations, which does not earn a quarter of the figure budget; `utils/compare_polar.py` keeps its
own plot for the design notes.

Run after detection_range.py, temporal_integration.py and check_history_depth.py (both K).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES, FRAME_SECONDS, RESULTS               # noqa: E402

FINE_BINS = ((0, 30), (30, 60), (60, 90), (90, 120), (120, 180))
CENTRES = [(lo + hi) / 2 for lo, hi in FINE_BINS]
BUDGET = "10"


def load(name: str) -> dict:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        sys.exit(f"missing {path}; run the stage that writes it first")
    return json.loads(path.read_text())


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    dr, ti = load("detection_range"), load("temporal_integration")
    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.2, 4.0))

    rate = None
    for name, d in dr["detection"].items():
        r = d["by_budget"].get(BUDGET) or d["by_budget"][int(BUDGET)]
        rate = r["per_hour"]
        ax.plot(CENTRES, 100 * np.array(r["recall"], float), marker="o", ms=4, label=name)
    ax.set_ylabel("per-frame recall (%)")
    ax.set_title(f"(a) detection vs range, {rate:.0f} false alarms/h")
    ax.set_ylim(0, 60)
    # Per-frame recall is not what an operator waits for: at 10 Hz, 30% per frame is a first alarm
    # in 0.3 s. The second axis says that without needing a sentence.
    seconds = np.array([0.2, 0.3, 0.5, 1.0, 3.0])
    ax_s = ax.twinx()
    ax_s.set_ylim(0, 60)
    ax_s.set_yticks(100 * FRAME_SECONDS / seconds)
    ax_s.set_yticklabels([f"{s:g}" for s in seconds])
    ax_s.set_ylabel("s to first alarm", fontsize=8)

    for subset, a in dr["audibility"].items():
        ax2.plot(CENTRES, [100 * r["below_median"] for r in a["by_range"]], marker="s", ms=4,
                 label=subset)
    ax2.set_ylabel("quieter than ambient (%)")
    ax2.set_title("(b) drone frames with no energy cue")
    ax2.set_ylim(0, 100)

    for name, m in ti["models"].items():
        runs = np.array([[w["auroc"] for w in seed] for seed in m["average"]], float)
        lat = [w["latency_s"] for w in m["average"][0]]
        ax3.plot(lat, runs.mean(0), marker="o", ms=4, label=name)
    ax3.set_xlabel("decision latency (s)")
    ax3.set_ylabel("detection AUROC")
    ax3.set_title("(c) averaging the score over time")

    # Both history curves on one axis: three arrays at K=5, and the K=9 cell stage G added, which
    # is where the climb finally stops. Plotting them together is the whole argument.
    for source, style in ((load("history_depth"), {}), (load("history_depth_K9"), {"ls": "--"})):
        by_array: dict[str, list] = {}
        for run_id, depths in source["runs"].items():
            by_array.setdefault(run_id.split("_K")[0], []).append(
                [100 * depths[str(j)]["hit_10deg"] for j in source["depths"]])
        for array, rows in by_array.items():
            v = np.array(rows, float)
            ax4.errorbar(source["depths"], v.mean(0),
                         yerr=v.std(0, ddof=1) if len(rows) > 1 else None,
                         marker="o", ms=4, capsize=3,
                         label=f"{array} K={source['K']}", **style)
    ax4.set_xlabel("frames of real history (j)")
    ax4.set_ylabel("within 10 deg (%)")
    ax4.set_xticks(range(1, 10))
    ax4.set_title("(d) history depth, weights held fixed")

    for a in (ax, ax2):
        a.set_xlabel("range (m)")
    for a in (ax, ax2, ax3, ax4):
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout(pad=0.6)
    out = FIGURES / f"report_figure_{split}.png"
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    raise SystemExit(main(a[0] if a else "test_2"))
