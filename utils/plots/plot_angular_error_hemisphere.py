"""Illustrative figure: the 10° hit disc on the sky, and its physical radius vs range.

The metric in `src/metrics.py` / `src/geometry.angular_distance` is the angle between
two unit vectors. A prediction is a hit when that angle is ≤ ANGULAR_THRESHOLD (10°).
The left panel is that disc on the sky hemisphere. The right panel is the same disc's
cross-range radius at distance R from the array:

    radius(R) = R tan(10°)

    .venv/bin/python utils/plots/plot_angular_error_hemisphere.py
    .venv/bin/python utils/plots/plot_angular_error_hemisphere.py --gif
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import ANGULAR_THRESHOLD, FIGURES, HIT_THRESHOLDS  # noqa: E402
from src.geometry import angular_distance, unit_vectors as unit     # noqa: E402
from src.metrics import DISTANCE_BINS                               # noqa: E402
from utils.plots import hemisphere as hemi                          # noqa: E402
from utils.plots.hemisphere import orthonormal, spherical_circle    # noqa: E402


TRUE_AZ, TRUE_EL = 8.0, 36.0
HIT_AZ, HIT_EL = 12.0, 41.0
MISS_AZ, MISS_EL = 22.0, 26.0
D_HIT = float(angular_distance(TRUE_AZ, TRUE_EL, HIT_AZ, HIT_EL))
D_MISS = float(angular_distance(TRUE_AZ, TRUE_EL, MISS_AZ, MISS_EL))
# Report range-bin edges, dropping the 0 and inf sentinels.
RANGE_MARKS = [b for b in DISTANCE_BINS if np.isfinite(b) and b > 0]


def spherical_cap(u, radius_deg, n_ring=72, n_rad=8):
    e1, e2 = orthonormal(u)
    rings, rads = [], np.linspace(0, np.radians(radius_deg), n_rad)
    t = np.linspace(0, 2 * np.pi, n_ring)
    for a in rads:
        rings.append(np.cos(a) * u + np.sin(a) * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2))
    return np.stack(rings, 0)


def disc_radius(range_m, theta_deg=ANGULAR_THRESHOLD):
    return np.asarray(range_m, float) * np.tan(np.radians(theta_deg))


def draw_hemisphere(ax, azim: float = 52) -> None:
    az = np.linspace(-np.pi, np.pi, 73)
    el = np.linspace(0, np.pi / 2, 25)
    AZ, EL = np.meshgrid(az, el)
    ax.plot_surface(np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL),
                    rstride=2, cstride=2, color="#d9e4f0",
                    edgecolor="#8aa0b8", linewidth=0.25, alpha=0.35, shade=True)

    for a in np.linspace(-np.pi, np.pi, 13, endpoint=False):
        e = np.linspace(0, np.pi / 2, 40)
        ax.plot(np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e),
                color="#7b8ea3", lw=0.4, alpha=0.55)
    for e in np.linspace(0, np.pi / 2, 7):
        a = np.linspace(-np.pi, np.pi, 73)
        ax.plot(np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e),
                color="#7b8ea3", lw=0.4 if e > 0 else 1.1, alpha=0.7 if e == 0 else 0.5)

    t = np.linspace(0, 2 * np.pi, 120)
    ax.plot(np.cos(t), np.sin(t), 0, color="#4a5563", lw=1.2)
    ax.plot([0], [0], [0], "o", color="#1f2933", ms=6, zorder=10)

    u = unit(TRUE_AZ, TRUE_EL)
    styles = {5: dict(color="#6b8f71", lw=1.0, ls="--"),
              10: dict(color="#2f6fad", lw=2.2, ls="-"),
              20: dict(color="#8a6a3a", lw=1.0, ls=":")}
    cap = spherical_cap(u, ANGULAR_THRESHOLD)
    ax.plot_surface(cap[:, :, 0], cap[:, :, 1], cap[:, :, 2],
                    color="#2f6fad", alpha=0.72, linewidth=0, shade=False)
    for r in HIT_THRESHOLDS:
        c = spherical_circle(u, r)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], **styles[r])

    def ray(p, color, lw, mk, ms, label):
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], color=color, lw=lw, solid_capstyle="round")
        ax.plot([p[0]], [p[1]], [p[2]], mk, color=color, ms=ms, label=label, zorder=12)

    ray(u, "#1f2933", 2.0, "o", 8,
        f"true  ({TRUE_AZ:.0f}° az, {TRUE_EL:.0f}° el)")
    p_hit, p_miss = unit(HIT_AZ, HIT_EL), unit(MISS_AZ, MISS_EL)
    ray(p_hit, "#2a7a4b", 1.6, "^", 8, f"hit   {D_HIT:.1f}°  (≤ {ANGULAR_THRESHOLD}°)")
    ray(p_miss, "#b23a3a", 1.6, "x", 8, f"miss  {D_MISS:.1f}°  (> {ANGULAR_THRESHOLD}°)")

    ax.plot([0, 0], [0, 0], [0, 1.02], color="#4a5563", lw=0.8)
    ax.text(0.04, 0.04, 1.10, "zenith", fontsize=8, color="#4a5563")
    ax.text(0.06, 0.06, -0.10, "array", fontsize=8, color="#1f2933")
    ax.view_init(elev=22, azim=azim)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-0.08, 1.15)
    ax.set_box_aspect((1, 1, 0.58))
    ax.set_axis_off()
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelspacing=0.3)
    ax.set_title(f"Same {ANGULAR_THRESHOLD:.0f}° disc on the sky", fontsize=10, pad=2)


def draw_radius_vs_range(ax) -> None:
    R = np.linspace(0, 180, 361)
    ax.fill_between(R, disc_radius(R, 10), color="#2f6fad", alpha=0.18, lw=0)
    ax.plot(R, disc_radius(R, 5), color="#6b8f71", ls="--", lw=1.2, label="5°")
    ax.plot(R, disc_radius(R, 10), color="#2f6fad", lw=2.2, label="10°  r = R tan 10°")

    marks = np.array(RANGE_MARKS, float)
    ax.plot(marks, disc_radius(marks, 10), "o", color="#2f6fad", ms=6, zorder=5)
    label_at = {30: (8, 8), 90: (6, 8), 180: (-8, -18)}
    for r, (dx, dy) in label_at.items():
        rad = float(disc_radius(r, 10))
        ax.annotate(f"{r:.0f} m → {rad:.1f} m",
                    xy=(r, rad), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if r < 180 else "right", va="bottom" if r < 180 else "top",
                    fontsize=8, color="#1f2933")

    ax.set_xlim(0, 185)
    ax.set_ylim(0, disc_radius(180, 10) * 1.18)
    ax.set_xlabel("distance R from the array (m)")
    ax.set_ylabel("circle radius (m)")
    ax.set_title("Physical 10° radius vs range", fontsize=10, pad=2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_rotation_gif() -> Path:
    """Slow Z-axis turntable in the conversation (SVG wireframe) style, not mplot3d."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Polygon, Rectangle

    def draw_frame(ax, yaw_deg):
        ax.clear()
        yaw = np.radians(yaw_deg)
        u = unit(TRUE_AZ, TRUE_EL)
        hemi.draw_wire(ax, yaw)

        ax.text(24, 36, f"Hit if great-circle error ≤ {ANGULAR_THRESHOLD:.0f}°",
                color=hemi.FG, fontsize=13, va="top")
        ax.text(24, 58, "Sky hemisphere from the array", color=hemi.MUTED, fontsize=10, va="top")
        sw = [(hemi.FG, "True"), (hemi.BLUE, f"{ANGULAR_THRESHOLD:.0f}° disc"),
              (hemi.SUCCESS, f"Hit {D_HIT:.1f}°"), (hemi.DANGER, f"Miss {D_MISS:.1f}°")]
        x0 = 430
        for color, lab in sw:
            ax.add_patch(Rectangle((x0, 32), 12, 12, facecolor=color, edgecolor="none"))
            ax.text(x0 + 16, 38, lab, color=hemi.FG, fontsize=9, va="center")
            x0 += 76

        cap = hemi.circle_xy(u, ANGULAR_THRESHOLD, yaw, 73)
        ax.add_patch(Polygon(cap, closed=True, facecolor=hemi.BLUE, edgecolor=hemi.BLUE,
                             lw=2.0, alpha=0.55))
        gxy = hemi.circle_xy(u, 5, yaw, 65)
        ax.plot(gxy[:, 0], gxy[:, 1], color=hemi.SUCCESS, lw=1.0, ls=(0, (5, 4)))
        oxy = hemi.circle_xy(u, 20, yaw, 65)
        ax.plot(oxy[:, 0], oxy[:, 1], color=hemi.ORANGE, lw=1.0, ls=(0, (3, 3)))

        o = hemi.project(np.zeros(3), yaw)
        for p, color, size in ((u, hemi.FG, 36), (unit(HIT_AZ, HIT_EL), hemi.SUCCESS, 32),
                               (unit(MISS_AZ, MISS_EL), hemi.DANGER, 32)):
            qx, qy = hemi.project(p, yaw)
            ax.plot([o[0], qx], [o[1], qy], color=color, lw=2.0, solid_capstyle="round")
            ax.scatter([qx], [qy], s=size, c=color, zorder=5)

    fig, ax = plt.subplots(figsize=(7.36, 5.0), facecolor=hemi.BG)
    fig.subplots_adjust(0, 0, 1, 1)
    n, fps = 120, 12
    draw_frame(ax, 50)

    def update(i):
        draw_frame(ax, 50 + i * (360.0 / n))
        return []

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    out = FIGURES / "angular_error_hemisphere_rotate.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    return out


def main(gif: bool = False) -> int:
    assert D_HIT < ANGULAR_THRESHOLD < D_MISS

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(11.2, 5.4))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_proj_type("persp", focal_length=1.4)
    draw_hemisphere(ax3d)

    ax2 = fig.add_subplot(1, 2, 2)
    draw_radius_vs_range(ax2)

    fig.suptitle(f"Hit if great-circle error ≤ {ANGULAR_THRESHOLD}°",
                 fontsize=11, y=0.98)
    fig.tight_layout()
    out = FIGURES / "angular_error_hemisphere.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(out)
    for r in RANGE_MARKS:
        print(f"  R={r:3.0f} m   radius={disc_radius(r):5.2f} m")
    if gif:
        print(write_rotation_gif())
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gif", action="store_true", help="also write the rotating GIF")
    raise SystemExit(main(p.parse_args().gif))
