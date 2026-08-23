"""Plot the 24-microphone array, evaluate candidate thinned subsets, and show what
thinning does to the beam pattern.

Answers a design question: if we drop microphones, which ones should go? Prints
aperture, minimum spacing, the spatial-aliasing limit and a 3-fold symmetry test for
each candidate subset, then renders two figures:

  figures/array_geometry.png    the array itself and three candidate subsets
  figures/array_beampattern.png response of each subset to ONE synthetic source

The second figure is the one that matters. It shows the false peaks a sparse array
invents -- the artefacts a brightest-pixel baseline cannot reject and that the CNN plus
temporal context is supposed to undo.

Geometry comes from the session's own alignment_params.json -- the array was
recalibrated between sessions, so it is never hardcoded.

Run: .venv/bin/python utils/plot_array_geometry.py
"""
import json
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = 343.0
REPO = Path(__file__).resolve().parent.parent
SESSION = REPO / "data/test_1/Oct_11_2024/flight_5/alignment_params.json"

AZ = np.arange(-180, 181, 4)
EL = np.arange(0, 91, 4)
BAND = np.linspace(200.0, 2200.0, 16)      # the paper's model band
SRC_AZ, SRC_EL = 40.0, 28.0                # a plausible drone: 100 m out, 50 m up

DEVICES = {f"device {k + 1}": list(range(6 * k, 6 * k + 6)) for k in range(4)}

SUBSETS = {
    "all 24":                 list(range(24)),
    "greedy-spread 6":        [2, 9, 11, 15, 16, 17],
    "symmetric 6 (top+base)": [0, 1, 2, 15, 16, 17],
    "device 3 (base legs)":   list(range(12, 18)),
    "device 1 (leg tops)":    list(range(0, 6)),
    "device 4 (flat ring)":   list(range(18, 24)),
    "symmetric 12 (1+3)":     list(range(0, 6)) + list(range(12, 18)),
    "symmetric 12 (4 rings)": [0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17],
    "tetrahedron 4":          [0, 15, 16, 17],
    "nested 4":               [0, 1, 15, 16],
}


def mic_positions(cfg):
    """24 positions: six 3-microphone rings up the tripod legs, then a flat ring of six."""
    a, a2, h, r = cfg["a_"], cfg["a2_"], cfg["h_"], cfg["r_"]
    pos = []
    for ri in range(6):
        for ai in range(3):
            pos.append([r[ri] * np.cos(np.radians(a[ai])),
                        r[ri] * np.sin(np.radians(a[ai])), h[ri]])
    for ai in range(6):
        pos.append([r[2] * np.cos(np.radians(a2[ai])),
                    r[2] * np.sin(np.radians(a2[ai])), h[2]])
    return np.array(pos)


def directions(az_deg, el_deg):
    az, el = np.meshgrid(np.radians(az_deg), np.radians(el_deg), indexing="ij")
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)], axis=-1).reshape(-1, 3)


def steering(pos, freqs, dirs):
    tau = dirs @ pos.T / C                                       # (D, M)
    return np.exp(-2j * np.pi * freqs[:, None, None] * tau[None]).astype(np.complex64)


def beampattern(pos):
    """Energy map produced by a single unit-amplitude source at (SRC_AZ, SRC_EL)."""
    W = steering(pos, BAND, directions(AZ, EL))                  # (F, D, M)
    src = steering(pos, BAND, directions([SRC_AZ], [SRC_EL]))     # (F, 1, M)
    a = np.conj(src[:, 0, :])                                    # source at unit gain
    y = np.einsum("fdm,fm->fd", W, a, optimize=True)
    e = (np.abs(y) ** 2).sum(0).reshape(len(AZ), len(EL))
    return e / e.max()


def spacings(pos):
    d = [np.linalg.norm(pos[i] - pos[j]) for i, j in itertools.combinations(range(len(pos)), 2)]
    return min(d), max(d)


def is_threefold(pos, tol=0.02):
    """Does rotating the subset by 120 degrees about the vertical map it onto itself?"""
    t = np.radians(120.0)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    for p in pos @ R.T:
        if np.min(np.linalg.norm(pos - p, axis=1)) > tol:
            return False
    return True


def coplanar_thickness(pos):
    """Smallest spread perpendicular to the best-fit plane; ~0 means no 3-D resolving power."""
    q = pos - pos.mean(0)
    return np.linalg.svd(q, compute_uv=False)[-1]


def sidelobe_db(e):
    """Peak level outside a 15-degree exclusion zone around the true source, in dB."""
    d = directions(AZ, EL)
    d0 = directions([SRC_AZ], [SRC_EL])[0]
    ang = np.degrees(np.arccos(np.clip(d @ d0, -1, 1))).reshape(e.shape)
    far = e[ang > 15.0]
    return 10 * np.log10(far.max())


def report(pos):
    hdr = (f"{'subset':<24}{'n':>3}{'apert':>8}{'mingap':>8}{'alias':>9}"
           f"{'beam2k2':>9}{'sidelobe':>10}{'flatness':>10}{'3fold':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name, idx in SUBSETS.items():
        p = pos[idx]
        dmin, dmax = spacings(p)
        e = beampattern(p)
        rows[name] = e
        print(f"{name:<24}{len(idx):>3}{dmax:>7.2f}m{dmin:>7.2f}m"
              f"{C / (2 * dmin):>7.0f}Hz{np.degrees(C / 2200.0 / dmax):>8.1f}d"
              f"{sidelobe_db(e):>9.1f}dB{coplanar_thickness(p):>9.2f}m"
              f"{'yes' if is_threefold(p) else 'no':>7}")
    print("\nsidelobe = strongest false peak more than 15 deg from the true source,")
    print("relative to the true peak. Higher is worse; 0 dB means the array cannot tell")
    print("the true direction from a false one. flatness ~ 0 means the mics are coplanar,")
    print("so elevation and its mirror image are indistinguishable.")
    return rows


def plot_geometry(pos, out):
    fig = plt.figure(figsize=(15, 9))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, 4))
    dev_of = np.repeat(np.arange(4), 6)

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    for k, (name, idx) in enumerate(DEVICES.items()):
        ax.scatter(*pos[idx].T, s=55, color=colors[k], label=name, depthshade=False)
    for ai in range(3):
        leg = [3 * ri + ai for ri in range(6)]
        ax.plot(*pos[leg].T, color="0.6", lw=1, zorder=0)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_title("Full array, coloured by recorder", pad=12)
    ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(pos[:, 0], pos[:, 1], s=55, c=colors[dev_of])
    for i, (x, y) in enumerate(pos[:, :2]):
        ax.annotate(str(i), (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="x (m)", ylabel="y (m)")
    ax.set_title("Top view: three legs 120 deg apart", pad=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2], s=55, c=colors[dev_of])
    ax.set(xlabel="radius (m)", ylabel="height (m)")
    ax.set_title("Side view: radius grows as height falls", pad=12)
    ax.grid(alpha=0.3)

    for j, name in enumerate(["symmetric 6 (top+base)", "device 3 (base legs)",
                              "tetrahedron 4"]):
        idx = SUBSETS[name]
        ax = fig.add_subplot(2, 3, 4 + j, projection="3d")
        ax.scatter(*pos.T, s=14, color="0.85", depthshade=False)
        ax.scatter(*pos[idx].T, s=80, color="crimson", depthshade=False,
                   edgecolors="black", linewidths=0.5)
        dmin, dmax = spacings(pos[idx])
        ax.set_title(f"{name}\naperture {dmax:.2f} m, unaliased below "
                     f"{C / (2 * dmin):.0f} Hz", fontsize=9, pad=14)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])

    fig.suptitle("WSU 24-channel array (test_1 geometry) and candidate thinned subsets",
                 fontsize=13)
    fig.subplots_adjust(hspace=0.28, wspace=0.18, top=0.90)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def plot_beampatterns(maps, out):
    names = ["all 24", "symmetric 6 (top+base)", "greedy-spread 6",
             "device 3 (base legs)", "device 4 (flat ring)", "tetrahedron 4"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        db = 10 * np.log10(np.maximum(maps[name], 1e-6)).T
        im = ax.imshow(db, origin="lower", aspect="auto", vmin=-20, vmax=0,
                       extent=[AZ[0], AZ[-1], EL[0], EL[-1]], cmap="magma")
        ax.plot(SRC_AZ, SRC_EL, "c+", ms=14, mew=2)
        ax.set_title(f"{name}   worst false peak {sidelobe_db(maps[name]):.1f} dB",
                     fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("azimuth (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("elevation (deg)")
    fig.colorbar(im, ax=axes, label="response (dB, peak = 0)", shrink=0.8)
    fig.suptitle("Response to a SINGLE source (cyan cross). Everything else is an "
                 "artefact of the array.", fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    cfg = json.load(open(SESSION))["mic_array"]
    pos = mic_positions(cfg)
    print(f"session: {SESSION.relative_to(REPO)}")
    print(f"source for beam patterns: azimuth {SRC_AZ} deg, elevation {SRC_EL} deg, "
          f"band {BAND[0]:.0f}-{BAND[-1]:.0f} Hz\n")
    maps = report(pos)
    print()
    plot_geometry(pos, REPO / "figures/array_geometry.png")
    plot_beampatterns(maps, REPO / "figures/array_beampattern.png")
