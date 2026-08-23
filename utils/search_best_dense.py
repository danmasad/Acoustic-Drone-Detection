"""Single-stage dense search: score EVERY non-coplanar subset on the same large set of
source directions, so "best-N" means best outright rather than best of a shortlist.

The two-stage search in search_mic_subsets.py screens on 12 directions and only re-scores a
shortlist. That is cheap but leaves the pool selected on a coarse criterion. This scores
everything on 250 directions.

Ranking statistic: the 95th percentile of the false-peak level across directions, not the
maximum. A maximum over 250 samples is a high-variance statistic dominated by one unlucky
direction, which is how the 4-microphone ranking turned out to be noise. The 95th
percentile is robust while still describing bad cases. Maximum and median are reported
alongside.

Writes results/subset_search.npz so downstream plots need not recompute.

Run: .venv/bin/python utils/search_best_dense.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import search_mic_subsets as s

REPO = Path(__file__).resolve().parent.parent
N_DIRS = 250
SEED = 7                      # disjoint from the screen (SOURCES) and held-out (seed 1)
CHUNK = 800
SIZES = (4, 6)

NAMED = {
    "all 24": list(range(24)),
    "device 1": list(range(0, 6)),
    "device 2": list(range(6, 12)),
    "device 3": list(range(12, 18)),
    "device 4": list(range(18, 24)),
    "device 1+3": list(range(0, 6)) + list(range(12, 18)),
    "symmetric-6": [0, 1, 2, 15, 16, 17],
    "symmetric-12": [0, 1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 17],
    "greedy-spread 6": [2, 9, 11, 15, 16, 17],
    "two-stage best-6": [12, 13, 14, 18, 20, 22],
}


def sources(n, seed):
    rng = np.random.default_rng(seed)
    return list(zip(rng.uniform(-180, 180, n), rng.uniform(0, 88, n)))


def score(pos, combos, srcs, label):
    """Returns (max, p95, median) in dB for every row of `combos`."""
    masks = np.zeros((len(combos), len(pos)), dtype=np.complex64)
    np.put_along_axis(masks, combos, 1.0, axis=1)
    per_src = np.empty((len(srcs), len(combos)), dtype=np.float32)
    t0 = time.time()
    for si, src in enumerate(srcs):
        phi, far = s.per_source_terms(pos, src)
        for lo in range(0, len(combos), CHUNK):
            m = masks[lo:lo + CHUNK]
            e = (np.abs(phi @ m.T) ** 2).reshape(len(s.BAND), -1, m.shape[0]).sum(0)
            per_src[si, lo:lo + CHUNK] = 10 * np.log10(e[far].max(0) / e.max(0))
        if (si + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{label}] {si + 1}/{len(srcs)} directions, {el:.0f}s elapsed, "
                  f"{el / (si + 1) * (len(srcs) - si - 1):.0f}s left", flush=True)
    return (per_src.max(0), np.percentile(per_src, 95, axis=0), np.median(per_src, axis=0))


def describe(pos, idx):
    p = pos[idx]
    d = [np.linalg.norm(p[i] - p[j]) for i, j in itertools.combinations(range(len(idx)), 2)]
    devs = sorted({int(i) // 6 + 1 for i in idx})
    return (f"aperture {max(d):.2f} m  min gap {min(d):.2f} m  "
            f"flatness {s.flatness(p):.2f} m  recorders {devs}")


if __name__ == "__main__":
    pos = s.mic_positions(json.load(open(s.SESSION))["mic_array"])
    srcs = sources(N_DIRS, SEED)
    print(f"session: {s.SESSION.relative_to(REPO)}")
    print(f"{N_DIRS} directions (seed {SEED}), band {s.BAND[0]:.0f}-{s.BAND[-1]:.0f} Hz, "
          f"ranking on the 95th percentile\n", flush=True)

    store = {}
    for n in SIZES:
        combos = np.array(list(itertools.combinations(range(len(pos)), n)), dtype=np.int64)
        combos = combos[[s.flatness(pos[c]) >= s.MIN_FLATNESS for c in combos]]
        print(f"n={n}: {len(combos):,} non-coplanar subsets", flush=True)
        mx, p95, med = score(pos, combos, srcs, f"n={n}")
        store[f"combos_{n}"] = combos.astype(np.int8)
        store[f"max_{n}"], store[f"p95_{n}"], store[f"med_{n}"] = mx, p95, med

        order = np.argsort(p95)
        print(f"\n  best {n}-microphone subsets by 95th percentile")
        for k in order[:8]:
            print(f"    {str(combos[k].tolist()):<26}p95 {p95[k]:>6.1f}  max {mx[k]:>6.1f}  "
                  f"med {med[k]:>6.1f}   {describe(pos, combos[k])}")
        print(f"  distribution: p95 ranges {p95.min():.1f} to {p95.max():.1f} dB, "
              f"median subset {np.median(p95):.1f} dB", flush=True)

        # How much does ranking by max disagree with ranking by p95?
        print(f"  rank correlation, max vs p95: {s.spearman(mx, p95):+.2f}\n", flush=True)

    print("named subsets on the same directions")
    rows = []
    for size in sorted({len(v) for v in NAMED.values()}):
        labels = [k for k, v in NAMED.items() if len(v) == size]
        idx = np.array([NAMED[k] for k in labels], dtype=np.int64)
        mx, p95, med = score(pos, idx, srcs, f"named n={size}")
        rows += list(zip(labels, p95, mx, med, [NAMED[k] for k in labels]))
    for lab, p, m, md, idx in sorted(rows, key=lambda t: t[1]):
        rank = ""
        key = f"p95_{len(idx)}"
        if key in store:
            rank = f"   percentile {100 * (store[key] < p).mean():.2f}"
        print(f"  {lab:<20}p95 {p:>6.1f}  max {m:>6.1f}  med {md:>6.1f}{rank}")
        store[f"named_{lab}"] = np.array([p, m, md])

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "subset_search.npz", n_dirs=N_DIRS, seed=SEED, **store)
    print(f"\nwrote {out / 'subset_search.npz'}")
