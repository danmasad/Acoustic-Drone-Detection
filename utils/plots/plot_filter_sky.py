"""Eight seconds of the kinematic filter on the matched-site sky.

Device 3, one frame, about 45 m out. The GPS path is one fading trail, as on
the cover. The U-Net and the filter are each a line from the array to the
direction they hold on that frame.

    .venv/bin/python utils/plots/plot_filter_sky.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import REPO, pred_file, run_id                        # noqa: E402
from src.geometry import unit_vectors                                 # noqa: E402
from utils.analysis.score_kinematic_filter import apply_filter        # noqa: E402
from utils.plots.hemisphere import (BG, CX, FG, H, MUTED, W,          # noqa: E402
                                    circle_xy, draw_wire, project, setup_axes)

START, N = 7015, 80
YAW = np.radians(150)
TRAIL_S = 100.0
GPS, RAW, FILT = "#f8fafc", "#fb923c", "#4ade80"
OUT = REPO / "docs" / "filter_sky.gif"


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def trail(xy, k, color):
    """Consecutive segments up to sample k, cosine-faded like the cover clip."""
    if k < 1:
        return np.empty((0, 2, 2)), np.empty((0, 4)), np.empty((0,))
    i0 = max(0, k - int(TRAIL_S / 0.1))
    age = (k - np.arange(i0, k)) * 0.1
    fade = 0.5 * (1.0 + np.cos(np.pi * np.clip(age, 0.0, TRAIL_S) / TRAIL_S))
    segs = np.stack([xy[i0:k], xy[i0 + 1:k + 1]], 1)
    rgb = np.tile(_rgb(color), (len(segs), 1))
    rgba = np.column_stack([rgb, fade])
    return segs, rgba, 0.7 + 1.6 * fade


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon, Rectangle

    d = np.load(pred_file(f"{run_id('device3', 1)}_test_2_causal"), allow_pickle=False)
    names = [str(s) for s in d["seg_ids"]]
    si = names.index("Mar_18_2025_f2_drone")
    m = (d["seg"] == si) & d["has_drone"].astype(bool) & np.isfinite(d["az"]) & (d["el"] >= 0)
    az = d["az"][m].astype(float)
    el = np.clip(d["el"][m].astype(float), 0, 89.5)
    paz = d["pred_az"][m].astype(float)
    pel = np.clip(d["pred_el"][m].astype(float), 0, 89.5)
    faz, fel = apply_filter(paz, pel)
    fel = np.clip(fel, 0, 89.5)
    rng = d["range_m"][m].astype(float)
    sl = slice(START, START + N)
    az, el, paz, pel = az[sl], el[sl], paz[sl], pel[sl]
    faz, fel, rng = faz[sl], fel[sl], rng[sl]
    tracks = {
        "raw": (np.column_stack(project(unit_vectors(paz, pel), YAW)), RAW),
        "gps": (np.column_stack(project(unit_vectors(az, el), YAW)), GPS),
        "filt": (np.column_stack(project(unit_vectors(faz, fel), YAW)), FILT),
    }
    gxyz = unit_vectors(az, el)
    ox, oy = project(np.zeros(3), YAW)

    fig, ax = plt.subplots(figsize=(7.36, 5.0), facecolor=BG, dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)

    def legend():
        y, h, box = 458.0, 12.0, 12.0
        labels = ((GPS, "GPS", 70.0), (RAW, "U-Net", 84.0), (FILT, "Filter", 80.0))
        total = sum(w for *_, w in labels)
        x = CX - total / 2
        for color, lab, w in labels:
            ax.add_patch(Rectangle((x, y), box, h, facecolor=color, edgecolor="none", zorder=9))
            ax.text(x + box + 6, y + h / 2, lab, color=FG, fontsize=9, va="center", zorder=9)
            x += w

    ax.clear()
    draw_wire(ax, YAW)
    ax.text(24, 28, "Device 3  ·  one frame  ·  matched site", color=FG, fontsize=12, va="top")
    legend()
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba()).copy()

    def draw(i):
        ax.clear()
        ax.imshow(bg, extent=(0, W, H, 0), origin="upper", interpolation="nearest", zorder=0)
        setup_axes(ax)
        cap = circle_xy(gxyz[i], 10.0, YAW)
        ax.add_patch(Polygon(cap, closed=True, facecolor=FILT, edgecolor=FILT,
                             lw=1.6, alpha=0.35, zorder=3))
        gxy = tracks["gps"][0]
        segs, cols, widths = trail(gxy, i, GPS)
        if len(segs):
            ax.add_collection(LineCollection(
                segs, colors=cols, linewidths=widths, capstyle="round", zorder=4))
        rxy, fxy = tracks["raw"][0][i], tracks["filt"][0][i]
        ax.plot([ox, rxy[0]], [oy, rxy[1]], color=RAW, lw=2.2,
                solid_capstyle="round", zorder=5)
        ax.plot([ox, fxy[0]], [oy, fxy[1]], color=FILT, lw=2.4,
                solid_capstyle="round", zorder=6)
        ax.scatter([rxy[0]], [rxy[1]], s=36, c=RAW, zorder=7)
        ax.scatter([fxy[0]], [fxy[1]], s=42, c=FILT, zorder=8)
        ax.scatter([gxy[i, 0]], [gxy[i, 1]], s=46, c=GPS, zorder=9,
                   edgecolors=FILT, linewidths=0.9)
        ax.text(24, 50, f"{rng[i]:.0f} m   ·   {i / 10:.1f} s", color=MUTED, fontsize=10, va="top")
        return []

    anim = FuncAnimation(fig, draw, frames=N, interval=100, blit=False)
    anim.save(OUT, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
