"""The three figures in section 6. Every localization bar is the test-2 dumps, mean of three seeds.

    MPLCONFIGDIR=/tmp/mpl .venv/bin/python utils/plots/plot_analysis.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.analysis.score_kinematic_filter import run_id, score_path, variant_paths  # noqa: E402

DOCS = Path(__file__).resolve().parents[2] / "docs"
BINS = ["0–30", "30–60", "60–90", "90–120", "120–180"]
# MATLAB's "gem" palette, the light-theme default (orderedcolors("gem") before the R2025a tweak).
GEM = (
    (0.0000, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
)
INK = "#1C1917"


def _seeds(array: str, k: int, field: str) -> np.ndarray:
    paths = variant_paths("seeds", run_id(array, k, seed="{s}"), "test_2")
    if len(paths) != 3:
        raise SystemExit(f"{array} K={k}: expected 3 dumps, found {len(paths)}")
    cols = [score_path(p)[field][:5] for p in paths]
    return np.vstack(cols)


def _mu_sd(cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cols.mean(0), cols.std(0, ddof=1)


def _style(ax):
    ax.set_facecolor("#FAFAF9")
    for sp in ax.spines.values():
        sp.set_color("#E7E5E4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#44403C", labelsize=9, length=0)
    ax.grid(axis="y", color="#E7E5E4", zorder=0)


def _grouped(ax, series, ylabel):
    n = len(series)
    x = np.arange(len(BINS))
    w = 0.78 / n
    for i, (label, color, mu) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, mu, w, label=label, color=color, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.02))
    _style(ax)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
        "axes.unicode_minus": False,
    })
    history_gain = {
        "device3": _seeds("device3", 9, "hit_raw") - _seeds("device3", 1, "hit_raw"),
        "all24": _seeds("all24", 9, "hit_raw") - _seeds("all24", 1, "hit_raw"),
    }
    gain = {}
    for key in (("all24", 1), ("device3", 1), ("device3", 9)):
        gain[key] = _seeds(*key, "hit_filt") - _seeds(*key, "hit_raw")

    level = {
        "device3": _seeds("device3", 9, "hit_raw"),
        "all24": _seeds("all24", 1, "hit_raw"),
    }
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 2.7), dpi=160)
    fig.patch.set_facecolor("#FAFAF9")
    _grouped(ax0, [
        ("Device 3", GEM[0], _mu_sd(history_gain["device3"])[0]),
        ("24 mics", GEM[1], _mu_sd(history_gain["all24"])[0]),
    ], "Gain, one frame to nine (points)")
    ax0.axhline(0, color="#A8A29E", lw=0.8, zorder=1)
    ax0.set_xlabel("Range (m)", fontsize=9)
    _grouped(ax1, [
        ("Device 3, nine frames", GEM[0], _mu_sd(level["device3"])[0]),
        ("24 mics, one frame", GEM[1], _mu_sd(level["all24"])[0]),
    ], "Within 10° (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Range (m)", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    history = DOCS / "analysis_history.png"
    fig.savefig(history, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 2.55), dpi=160)
    fig.patch.set_facecolor("#FAFAF9")
    _grouped(ax, [
        ("24 mics, one frame", GEM[1], _mu_sd(gain[("all24", 1)])[0]),
        ("Device 3, one frame", GEM[0], _mu_sd(gain[("device3", 1)])[0]),
        ("Device 3, nine frames", GEM[4], _mu_sd(gain[("device3", 9)])[0]),
    ], "Filter gain (points)")
    ax.axhline(0, color="#A8A29E", lw=0.8, zorder=1)
    ax.set_xlabel("Range (m)", fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    filt = DOCS / "analysis_filter.png"
    fig.savefig(filt, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    # Published detector readings. Test 2 is the other half of the training afternoon
    # (0.66). Leave-one-scene-out over the three recordings averages 0.49. The sealed
    # site, one microphone, is 0.53. A forced 95% recall alarm rate is omitted: a
    # detector at chance alarms constantly at that operating point.
    fig, ax = plt.subplots(figsize=(5.4, 2.55), dpi=160)
    fig.patch.set_facecolor("#FAFAF9")
    vals = [0.66, 0.49, 0.53]
    ax.bar(np.arange(3), vals, color=[GEM[0], GEM[1], GEM[4]], width=0.62, zorder=2)
    ax.axhline(0.5, color="#A8A29E", ls="--", lw=0.9, zorder=1)
    ax.set_xlim(-0.55, 2.85)
    ax.text(2.42, 0.50, "chance", ha="left", va="bottom", color="#57534E", fontsize=8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([
        "Test 2\nsame afternoon",
        "One scene\nheld out",
        "Test 1\none microphone",
    ], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
    _style(ax)
    fig.tight_layout()
    det = DOCS / "analysis_detector.png"
    fig.savefig(det, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print("history gain, nine frames minus one, mean ± sd")
    for key, cols in history_gain.items():
        mu, sd = _mu_sd(cols)
        print(f"  {key}", [f"{m:+.1f}±{s:.1f}" for m, s in zip(mu, sd)])
    print("filter gain means ± sd")
    for key, cols in gain.items():
        mu, sd = _mu_sd(cols)
        print(f"  {key}", [f"{m:+.0f}±{s:.0f}" for m, s in zip(mu, sd)])
    print(history)
    print(filt)
    print(det)


if __name__ == "__main__":
    main()
