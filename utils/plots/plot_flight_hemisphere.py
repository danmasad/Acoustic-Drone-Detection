"""One GPS flight drawn on the black sky hemisphere (no turntable).

Uses the committed test-2 dump so it does not need the 24 GB of audio: Mar 18 2025
flight 2, the drone segment of test 2, 10 Hz GPS labels.

    .venv/bin/python utils/plots/plot_flight_hemisphere.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ANGULAR_THRESHOLD, FIGURES, pred_file        # noqa: E402
from src.geometry import angular_distance                          # noqa: E402
from utils.plots.plot_angular_error_hemisphere import orthonormal   # noqa: E402

DUMP = pred_file("device3_K1_w32_b16_wce_s0_test_2_causal")
SEG_NAME = "Mar_18_2025_f2_drone"
YAW0 = 50.0
TRAIL_S = 100.0
BG, FG, MUTED, BORDER = "#0a0a0a", "#f5f5f4", "#a8a29e", "#3f3f46"
BLUE, HEAD = "#60a5fa", "#f8fafc"
HIT, MISS = "#4ade80", "#f87171"
NEAR, FAR = np.array([56, 189, 248]) / 255, np.array([251, 146, 60]) / 255
W, H, CX, CY, SCALE, PITCH = 736.0, 500.0, 368.0, 278.0, 188.0, np.radians(22)


def load_flight():
    d = np.load(DUMP, allow_pickle=False)
    names = [str(s) for s in d["seg_ids"]]
    si = names.index(SEG_NAME)
    m = (d["seg"] == si) & d["has_drone"].astype(bool) & np.isfinite(d["az"]) & (d["el"] >= 0)
    az, el, r = d["az"][m].astype(float), d["el"][m].astype(float), d["range_m"][m].astype(float)
    paz, pel = d["pred_az"][m].astype(float), d["pred_el"][m].astype(float)
    t = np.arange(len(az)) / 10.0
    return az, np.clip(el, 0, 89.5), r, t, paz, pel


def sph_many(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], 1)


def rotate(p, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = p[..., 0] * c - p[..., 1] * s, p[..., 0] * s + p[..., 1] * c, p[..., 2]
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    return np.stack([x, y * cp - z * sp, y * sp + z * cp], -1)


def project(p, yaw):
    r = rotate(np.asarray(p, float), yaw)
    k = SCALE * (1.35 / (1.35 - r[..., 1] * 0.45))
    return CX + r[..., 0] * k, CY - r[..., 2] * k


def circle_xy(u, rad_deg, yaw, n=72):
    a = np.radians(rad_deg)
    e1, e2 = orthonormal(u)
    t = np.linspace(0, 2 * np.pi, n)
    pts = np.cos(a) * u + np.sin(a) * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)
    x, y = project(pts, yaw)
    return np.column_stack([x, y])


def range_rgba(r):
    t = np.clip((np.asarray(r, float) - 10.0) / 140.0, 0, 1)[:, None]
    rgb = (1 - t) * NEAR + t * FAR
    a = np.full((len(r), 1), 0.95)
    return np.hstack([rgb, a])


def draw_wire(ax, yaw):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)
    for i in range(12):
        a = i / 12 * 2 * np.pi
        e = np.linspace(0, np.pi / 2, 17)
        pts = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], 1)
        x, y = project(pts, yaw)
        ax.plot(x, y, color=BORDER, lw=1.0, solid_capstyle="round")
    for j in range(1, 6):
        e = j / 6 * np.pi / 2
        a = np.linspace(0, 2 * np.pi, 73)
        pts = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.full_like(a, np.sin(e))], 1)
        x, y = project(pts, yaw)
        ax.plot(x, y, color=BORDER, lw=1.0)
    hz = np.linspace(0, 2 * np.pi, 73)
    hpts = np.stack([np.cos(hz), np.sin(hz), np.zeros_like(hz)], 1)
    hx, hy = project(hpts, yaw)
    ax.fill(hx, hy, facecolor="#171717", edgecolor=FG, lw=1.5, alpha=0.92, zorder=0)
    ox, oy = project(np.array([0.0, 0.0, 0.0]), yaw)
    zx, zy = project(np.array([0.0, 0.0, 1.05]), yaw)
    ax.plot([ox, zx], [oy, zy], color=MUTED, lw=1.0)
    ax.scatter([ox], [oy], s=28, c=FG, zorder=6)
    ax.text(zx + 8, zy - 4, "zenith", color=MUTED, fontsize=9, va="bottom")
    ax.text(ox + 10, oy + 16, "array", color=FG, fontsize=9, va="top")


def slerp(a, b, w):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = np.arccos(d)
    if omega < 1e-6:
        return a
    so = np.sin(omega)
    return (np.sin((1.0 - w) * omega) * a + np.sin(w * omega) * b) / so


def trail_segments(xy, r, t, t_now, wrap_step):
    """Last TRAIL_S seconds, cosine-faded, broken at azimuth wraps."""
    if len(xy) < 2:
        return np.empty((0, 2, 2)), np.empty((0, 4)), np.empty((0,))
    age = t_now - t[:-1]
    fade = 0.5 * (1.0 + np.cos(np.pi * np.clip(age, 0.0, TRAIL_S) / TRAIL_S))
    keep = (age >= 0.0) & (age < TRAIL_S) & ~np.asarray(wrap_step, bool)
    segs = np.stack([xy[:-1], xy[1:]], 1)[keep]
    rgba = range_rgba(r[:-1])
    rgba[:, 3] *= fade
    return segs, rgba[keep], (0.7 + 1.6 * fade)[keep]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Polygon, Rectangle

    az, el, rng, t, paz, pel = load_flight()
    yaw = np.radians(YAW0)
    xyz = sph_many(az, el)
    pxyz = sph_many(paz, np.clip(pel, 0, 89.5))
    xs, ys = project(xyz, yaw)
    pxs, pys = project(pxyz, yaw)
    xy = np.column_stack([xs, ys])
    pxy = np.column_stack([pxs, pys])
    wrap = angular_distance(az[:-1], el[:-1], az[1:], el[1:]) > 12.0
    err = angular_distance(az, el, paz, pel)
    ox, oy = project(np.array([0.0, 0.0, 0.0]), yaw)

    # 30 s clip. 30 fps is the GIF-native rate (3/100 s delay) and looks smooth
    # without the huge palette of a 60 fps encode. Playhead slerps every sample.
    fps = 30
    n = 30 * fps
    fig, ax = plt.subplots(figsize=(7.36, 5.0), facecolor=BG, dpi=110)
    fig.subplots_adjust(0, 0, 1, 1)

    def legend():
        y, h, box = 458.0, 12.0, 12.0
        labels = ((HEAD, "GPS", 58.0), (HIT, "model ≤10°", 108.0), (MISS, "model >10°", 112.0))
        bar_w = 88.0
        bar_block = 28.0 + bar_w + 24.0
        total = sum(w for *_, w in labels) + bar_block
        x = CX - total / 2
        for color, lab, w in labels:
            ax.add_patch(Rectangle((x, y), box, h, facecolor=color, edgecolor="none", zorder=9))
            ax.text(x + box + 6, y + h / 2, lab, color=FG, fontsize=9, va="center", zorder=9)
            x += w
        x += 22
        ax.text(x, y + h / 2, "near", color=MUTED, fontsize=8, ha="right", va="center", zorder=9)
        gx = x + 6
        for j in range(20):
            rgb = (1 - j / 19) * NEAR + (j / 19) * FAR
            ax.add_patch(Rectangle((gx + j * bar_w / 20, y + 1), bar_w / 20 + 0.3, h - 2,
                                   facecolor=rgb, edgecolor="none", zorder=9))
        ax.text(gx + bar_w + 6, y + h / 2, "far", color=MUTED, fontsize=8, ha="left",
                va="center", zorder=9)

    ax.clear()
    draw_wire(ax, yaw)
    ax.text(24, 28, "Mar 18 2025  ·  flight 2  ·  same site as training",
            color=FG, fontsize=13, va="top")
    legend()
    fig.canvas.draw()
    bg = np.asarray(fig.canvas.buffer_rgba()).copy()

    def draw(i):
        ax.clear()
        ax.imshow(bg, extent=(0, W, H, 0), origin="upper", interpolation="nearest", zorder=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(BG)
        u = i / max(n - 1, 1) * (len(xy) - 1)
        k0 = int(np.floor(u))
        k1 = min(k0 + 1, len(xy) - 1)
        a = float(u - k0)
        t_now = (1.0 - a) * t[k0] + a * t[k1]
        rng_now = (1.0 - a) * rng[k0] + a * rng[k1]
        err_now = (1.0 - a) * err[k0] + a * err[k1]
        g = slerp(xyz[k0], xyz[k1], a)
        p = slerp(pxyz[k0], pxyz[k1], a)
        gx, gy = project(g, yaw)
        px, py = project(p, yaw)
        k = k1 + 1
        j0 = int(np.searchsorted(t[:k], t_now - TRAIL_S))
        segs, cols, widths = trail_segments(
            xy[j0:k], rng[j0:k], t[j0:k], t_now, wrap[j0:k - 1])
        if len(segs):
            lc = LineCollection(segs, colors=cols, linewidths=widths,
                                capstyle="round", zorder=4)
            ax.add_collection(lc)
        inside = bool(err_now <= ANGULAR_THRESHOLD)
        tone = HIT if inside else MISS
        cap = circle_xy(g, ANGULAR_THRESHOLD, yaw)
        ax.add_patch(Polygon(cap, closed=True, facecolor=tone, edgecolor=tone,
                             lw=1.8, alpha=0.40, zorder=5))
        ax.plot([ox, px], [oy, py], color=tone, lw=2.2, solid_capstyle="round", zorder=6)
        ax.scatter([px], [py], s=36, c=tone, zorder=7)
        ax.scatter([gx], [gy], s=42, c=HEAD, zorder=8, edgecolors=tone, linewidths=0.8)
        status = "hit" if inside else "miss"
        ax.text(24, 50,
                f"GPS  ·  t={t_now:.0f}s of {t[-1]:.0f}s  ·  distance {rng_now:.0f} m  ·  "
                f"error {err_now:.1f}°  {status}",
                color=MUTED, fontsize=10, va="top")

    draw(0)

    def update(i):
        draw(i)
        return []

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    out = FIGURES / "flight_hemisphere_mar18_f2.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
