"""Exhaustively search microphone subsets for the best achievable beam pattern.

Hand-picking subsets risks comparing the full array against a needlessly bad thinning,
which would flatter our own claim. This enumerates every subset of a given size, scores
each by its worst false peak across several source directions, and reports the best.

Score: peak response more than 15 degrees from the true source, relative to the true
peak, in dB. Lower is better. Subsets whose microphones are near-coplanar are rejected,
because a flat array cannot separate an elevation from its mirror image.

Run: .venv/bin/python utils/search/search_mic_subsets.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import C, REPO, SESSION_DIRS, load_align                  # noqa: E402
from src.geometry import coplanar_thickness as flatness                   # noqa: E402
from src.geometry import directions, mic_positions, unit_vectors          # noqa: E402

SESSION = SESSION_DIRS[3]            # Oct 11 flight 5, the test_1 geometry
BAND = np.linspace(200.0, 2200.0, 16)
EXCLUDE_DEG = 15.0
MIN_FLATNESS = 0.15          # metres of spread perpendicular to the best-fit plane
CHUNK = 800

# Sources spanning the elevations a drone actually occupies; worst case is reported.
SOURCES = [(az, el) for az in (-140.0, -60.0, 20.0, 100.0) for el in (10.0, 30.0, 60.0)]

NAMED = {
    "all 24":                 list(range(24)),
    "greedy-spread 6":        [2, 9, 11, 15, 16, 17],
    "symmetric 6 (top+base)": [0, 1, 2, 15, 16, 17],
    "device 1":               list(range(0, 6)),
    "device 2":               list(range(6, 12)),
    "device 3":               list(range(12, 18)),
    "device 4":               list(range(18, 24)),
    "tetrahedron 4":          [0, 15, 16, 17],
    "top triangle + base":    [0, 1, 2, 17],
}


def per_source_terms(pos, src):
    """phi[f, d, m]: the term microphone m contributes when steering to d.

    Summing phi over any microphone subset gives that subset's beamformer output, so
    every subset is a masked sum of one precomputed tensor.
    """
    dirs = directions()
    d0 = unit_vectors(*src)
    dtau = (dirs @ pos.T) - (d0 @ pos.T)                       # (D, M)
    phi = np.exp(-2j * np.pi * BAND[:, None, None] * dtau[None] / C).astype(np.complex64)
    ang = np.degrees(np.arccos(np.clip(dirs @ d0, -1, 1)))
    return phi.reshape(len(BAND) * len(dirs), len(pos)), ang > EXCLUDE_DEG


def score_all(pos, n):
    """Worst-case sidelobe, in dB, for every subset of size n that is not coplanar."""
    combos = np.array(list(itertools.combinations(range(len(pos)), n)), dtype=np.int64)
    keep = np.array([flatness(pos[c]) >= MIN_FLATNESS for c in combos])
    combos = combos[keep]
    masks = np.zeros((len(combos), len(pos)), dtype=np.complex64)
    np.put_along_axis(masks, combos, 1.0, axis=1)

    worst = np.full(len(combos), -np.inf)
    for src in SOURCES:
        phi, far = per_source_terms(pos, src)
        for lo in range(0, len(combos), CHUNK):
            m = masks[lo:lo + CHUNK]
            y = phi @ m.T                                       # (F*D, chunk)
            e = (np.abs(y) ** 2).reshape(len(BAND), -1, m.shape[0]).sum(0)
            db = 10 * np.log10(e[far].max(0) / e.max(0))
            worst[lo:lo + CHUNK] = np.maximum(worst[lo:lo + CHUNK], db)
    return combos, worst


def dense_scores(pos, combos, sources):
    """Re-score a shortlist on many held-out directions. Returns worst and median dB."""
    masks = np.zeros((len(combos), len(pos)), dtype=np.complex64)
    np.put_along_axis(masks, combos, 1.0, axis=1)
    per_src = np.empty((len(sources), len(combos)))
    for si, src in enumerate(sources):
        phi, far = per_source_terms(pos, src)
        y = phi @ masks.T
        e = (np.abs(y) ** 2).reshape(len(BAND), -1, len(combos)).sum(0)
        per_src[si] = 10 * np.log10(e[far].max(0) / e.max(0))
    return per_src.max(0), np.median(per_src, axis=0)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def held_out_sources(n, seed=1):
    """Directions disjoint from SOURCES, drawn uniformly over the grid's own range."""
    rng = np.random.default_rng(seed)
    return list(zip(rng.uniform(-180, 180, n), rng.uniform(0, 88, n)))


def named_scores(pos):
    out = {}
    for name, idx in NAMED.items():
        mask = np.zeros((1, len(pos)), dtype=np.complex64)
        mask[0, idx] = 1.0
        worst = -np.inf
        for src in SOURCES:
            phi, far = per_source_terms(pos, src)
            e = (np.abs(phi @ mask.T) ** 2).reshape(len(BAND), -1).sum(0)
            worst = max(worst, 10 * np.log10(e[far].max() / e.max()))
        out[name] = (worst, flatness(pos[idx]))
    return out


if __name__ == "__main__":
    pos = mic_positions(load_align(SESSION))
    print(f"session: {SESSION.relative_to(REPO)}")
    print(f"{len(SOURCES)} source directions, band {BAND[0]:.0f}-{BAND[-1]:.0f} Hz, "
          f"exclusion {EXCLUDE_DEG:.0f} deg\n")

    print("hand-picked candidates (worst-case sidelobe, lower is better)")
    for name, (db, fl) in sorted(named_scores(pos).items(), key=lambda kv: kv[1][0]):
        print(f"  {name:<24}{db:>7.1f} dB   flatness {fl:.2f} m")

    held_out = held_out_sources(200)
    for n in (4, 6):
        combos, worst = score_all(pos, n)
        order = np.argsort(worst)
        print(f"\nbest {n}-microphone subsets out of {len(combos)} non-coplanar options")
        for k in order[:5]:
            idx = combos[k].tolist()
            p = pos[idx]
            d = [np.linalg.norm(p[i] - p[j])
                 for i, j in itertools.combinations(range(n), 2)]
            print(f"  {str(idx):<26}{worst[k]:>7.1f} dB   aperture {max(d):.2f} m   "
                  f"min gap {min(d):.2f} m   flatness {flatness(p):.2f} m")
        print(f"  worst of the {len(combos)}: {worst[order[-1]]:.1f} dB "
              f"(spread {worst[order[0]]:.1f} to {worst[order[-1]]:.1f} dB)")

        # Does the shortlist survive directions it was not selected on?
        short = order[:200]
        d_worst, d_med = dense_scores(pos, combos[short], held_out)
        rank = np.argsort(np.argsort(d_worst))
        print(f"  held out: {len(held_out)} unseen directions, top-{len(short)} rescored")
        print(f"    rank correlation screen vs held out : {spearman(worst[short], d_worst):+.2f}")
        print(f"    screen winner {combos[short[0]].tolist()}: held-out worst "
              f"{d_worst[0]:.1f} dB, median {d_med[0]:.1f} dB, rank {rank[0] + 1}/{len(short)}")
        best = int(np.argmin(d_worst))
        print(f"    held-out winner {combos[short[best]].tolist()}: worst "
              f"{d_worst[best]:.1f} dB, median {d_med[best]:.1f} dB")
        print(f"    spread across the shortlist: {d_worst.min():.1f} to {d_worst.max():.1f} dB")

    # dense_scores needs equal-length index rows, so score each size group separately
    print("\nnamed subsets on the same held-out directions")
    out = []
    for size in sorted({len(v) for v in NAMED.values()}):
        labels = [k for k, v in NAMED.items() if len(v) == size]
        idx = np.array([NAMED[k] for k in labels], dtype=np.int64)
        dw, dm = dense_scores(pos, idx, held_out)
        out += list(zip(labels, dw, dm))
    for lab, w, m in sorted(out, key=lambda t: t[1]):
        print(f"  {lab:<24} worst {w:>6.1f} dB   median {m:>6.1f} dB")
