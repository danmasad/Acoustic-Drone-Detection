"""Shared drawing helpers for the black sky-hemisphere figures.

A fixed-pitch camera looks at the unit hemisphere from the side; `yaw` turns the scene
about the vertical. Screen coordinates are pixels in a W x H canvas with y pointing down,
so callers set `ax.set_ylim(H, 0)`. Used by plot_flight_hemisphere.py and
plot_angular_error_hemisphere.py.
"""
import numpy as np

BG, FG, MUTED, BORDER, FILL = "#0a0a0a", "#f5f5f4", "#a8a29e", "#3f3f46", "#171717"
BLUE, SUCCESS, DANGER, ORANGE = "#60a5fa", "#4ade80", "#f87171", "#fb923c"
W, H, CX, CY, SCALE, PITCH = 736.0, 500.0, 368.0, 278.0, 188.0, np.radians(22)


def orthonormal(u):
    """Two unit vectors spanning the plane perpendicular to `u`."""
    tmp = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(tmp, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    return e1, e2


def spherical_circle(u, radius_deg, n=180):
    """(n, 3) points on the small circle of angular radius `radius_deg` around `u`."""
    a = np.radians(radius_deg)
    e1, e2 = orthonormal(u)
    t = np.linspace(0, 2 * np.pi, n)
    return np.cos(a) * u + np.sin(a) * (np.cos(t)[:, None] * e1 + np.sin(t)[:, None] * e2)


def rotate(p, yaw):
    """Yaw about z, then pitch the camera; works on (..., 3)."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y, z = p[..., 0] * c - p[..., 1] * s, p[..., 0] * s + p[..., 1] * c, p[..., 2]
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    return np.stack([x, y * cp - z * sp, y * sp + z * cp], -1)


def project(p, yaw):
    """(..., 3) scene points -> (x, y) pixel arrays with a mild perspective."""
    r = rotate(np.asarray(p, float), yaw)
    k = SCALE * (1.35 / (1.35 - r[..., 1] * 0.45))
    return CX + r[..., 0] * k, CY - r[..., 2] * k


def circle_xy(u, radius_deg, yaw, n=72):
    """(n, 2) pixel outline of the small circle around `u`."""
    x, y = project(spherical_circle(u, radius_deg, n), yaw)
    return np.column_stack([x, y])


def setup_axes(ax):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


def draw_wire(ax, yaw):
    """Meridians, parallels, the filled horizon disc, the zenith axis and labels."""
    setup_axes(ax)
    for i in range(12):
        a = i / 12 * 2 * np.pi
        e = np.linspace(0, np.pi / 2, 17)
        x, y = project(np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], 1), yaw)
        ax.plot(x, y, color=BORDER, lw=1.0, solid_capstyle="round")
    for j in range(1, 6):
        e = j / 6 * np.pi / 2
        a = np.linspace(0, 2 * np.pi, 73)
        x, y = project(np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a),
                                 np.full_like(a, np.sin(e))], 1), yaw)
        ax.plot(x, y, color=BORDER, lw=1.0)
    hz = np.linspace(0, 2 * np.pi, 73)
    hx, hy = project(np.stack([np.cos(hz), np.sin(hz), np.zeros_like(hz)], 1), yaw)
    ax.fill(hx, hy, facecolor=FILL, edgecolor=FG, lw=1.5, alpha=0.92, zorder=0)
    ox, oy = project(np.array([0.0, 0.0, 0.0]), yaw)
    zx, zy = project(np.array([0.0, 0.0, 1.05]), yaw)
    ax.plot([ox, zx], [oy, zy], color=MUTED, lw=1.0)
    ax.scatter([ox], [oy], s=28, c=FG, zorder=6)
    ax.text(zx + 8, zy - 4, "zenith", color=MUTED, fontsize=9, va="bottom")
    ax.text(ox + 10, oy + 16, "array", color=FG, fontsize=9, va="top")
