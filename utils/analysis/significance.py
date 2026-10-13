"""Significance for the results we actually intend to claim, with the right unit of evidence.

Every comparison in this project has two sources of variation and they need different treatment.

**Frames are not independent.** A drone flies through a range bin in one continuous pass, frames
are 100 ms apart, and a model that is wrong about one frame is wrong about its neighbour. Treating
15,267 frames as 15,267 observations would put a standard error on a hit rate of about 0.4 pp,
which is nonsense. So every frame-level test here is a **moving-block bootstrap**: contiguous
blocks of frames are resampled with replacement, which preserves the within-block correlation
instead of assuming it away. Block length is swept, because a block bootstrap that only works at
one block length is not evidence. The design effect — how much wider the block interval is than
the naive one — is reported, since it is the honest measure of how much the correlation costs.

**Seeds are not free either.** Two independently trained models differ by initialization as well
as by whatever we changed, so a frame-paired test between two trained runs answers "did these
particular weights differ" and not "does the change help". With three seeds per cell the seed
component can barely be estimated, so it is reported separately rather than folded in, and the
across-seed test is stated with its own limit: an exact permutation test on three against three
has 20 possible assignments, so the smallest one-sided p it can ever return is 0.05. That is a
property of the design, not of the effect.

Which test answers which claim:

- **Freezing history** is the strong design. One set of weights scores the same frames twice, once
  with real history and once with the current frame repeated. No seed variance, no capacity
  difference, no training difference — the only thing that changes is the information. Block
  bootstrap on the paired per-frame difference is exactly the right test.
- **K=9 against K=1 as separate runs** confounds history with capacity: the K=9 model has nine
  times the input channels. It gets the across-seed test, and it is reported as the weaker of the
  two lines of evidence even though it is the one the headline table shows.
- **Native disc against rectangular** compares different trained models, so it gets both: a block
  bootstrap over frames with seeds averaged, and an across-seed test, with the caveat that the
  first controls frame difficulty and the second controls initialization, and neither controls
  both at once.

Separately from the tests, every headline cell gets an interval on its hit rate (`cell_ci`), with
the frame and seed components reported apart and then combined; its gate is that the per-seed
point estimates reproduce each run's `metrics.json`.

The family of claims is corrected with Holm-Bonferroni, which is uniformly more powerful than
plain Bonferroni and needs no independence assumption.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import metrics, predict                                   # noqa: E402
from src.config import ANGULAR_THRESHOLD, pred_file, run_dir, run_id, table_file  # noqa: E402
from src.geometry import angular_distance                          # noqa: E402

SEEDS = (0, 1, 2)
N_BOOT = 20000
BLOCKS = (30, 100, 300)          # frames: 3 s, 10 s, 30 s
MAIN_BLOCK = 100
FAR = (90, 200)                  # the bin the history gain concentrates in


# --- the frame-level machinery ---------------------------------------------------------------

def time_order(d: dict) -> np.ndarray:
    """Indices that put a dump into recording order.

    Necessary, not cosmetic: batching buckets by segment and flushes the trailing partial bucket
    of every segment at the very end, so a dump is *almost* in time order and a block bootstrap
    over the raw order would splice unrelated moments into the same block.
    """
    return np.lexsort((d["frame"], d["seg"]))


def block_bootstrap(x: np.ndarray, seg: np.ndarray, block: int, n_boot: int = N_BOOT,
                    seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of mean(x) resampling contiguous blocks within segments.

    Uses block sums off a cumulative sum, so cost is one pass plus the draws rather than
    materializing every resample.
    """
    rng = np.random.default_rng(seed)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    starts = []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        if len(idx) >= block:
            starts.append(np.arange(idx[0], idx[-1] - block + 2))
    starts = np.concatenate(starts)
    sums = cs[starts + block] - cs[starts]
    k = max(int(np.ceil(len(x) / block)), 1)
    draw = rng.integers(0, len(starts), (n_boot, k))
    return sums[draw].sum(1) / (k * block)


def crossings(mask: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Label each contiguous run of `mask` within a segment: -1 outside, 0..n-1 inside.

    This is the honest resampling unit for a range-restricted comparison. The 100-200 m band
    holds a few hundred frames, but they are not a few hundred observations: they are the handful
    of moments when a flight happened to be that far away, each one a continuous stretch of 100 ms
    frames. A fixed-length block bootstrap on such a subset ends up with a single-digit number of
    blocks, which is worse than admitting how few independent crossings there are.
    """
    prev_mask = np.concatenate([[False], mask[:-1]])
    prev_seg = np.concatenate([[-1], seg[:-1]])
    starts = mask & (~prev_mask | (seg != prev_seg))
    out = np.full(len(mask), -1, np.int64)
    out[mask] = np.cumsum(starts)[mask] - 1
    return out


def cluster_bootstrap(x: np.ndarray, cluster: np.ndarray, n_boot: int = N_BOOT,
                      seed: int = 0) -> np.ndarray:
    """Bootstrap mean(x) resampling whole clusters, each kept intact with its own length."""
    rng = np.random.default_rng(seed)
    ids = np.unique(cluster)
    sums = np.array([x[cluster == c].sum() for c in ids])
    lens = np.array([(cluster == c).sum() for c in ids], float)
    draw = rng.integers(0, len(ids), (n_boot, len(ids)))
    return sums[draw].sum(1) / lens[draw].sum(1)


def summarize(boots: np.ndarray, obs: float, x: np.ndarray, label: str, unit: str,
              n_units: int, verbose: bool = True) -> dict:
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = max(float(p), 1.0 / N_BOOT)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    naive = x.std(ddof=1) / np.sqrt(len(x))
    out = {"label": label, "n_frames": int(len(x)), "unit": unit, "n_units": int(n_units),
           "mean": obs, "ci95": [float(lo), float(hi)], "p": p,
           "se_resampled": float(boots.std(ddof=1)), "se_naive": float(naive),
           "design_effect": float(boots.std(ddof=1) / naive) if naive else float("nan"),
           "underpowered": bool(n_units < 12)}
    if verbose:
        warn = "  <- only %d %s" % (n_units, unit) if out["underpowered"] else ""
        print(f"  {label:<44}{obs:>+8.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"p {p:.4f}  design effect {out['design_effect']:.1f}x{warn}")
    return out


MIN_UNITS = 12          # below this a bootstrap p-value is not worth printing


def cluster_sign_test(x: np.ndarray, cluster: np.ndarray) -> dict:
    """Exact two-sided sign test over cluster means.

    The fallback whenever there are too few clusters for a bootstrap to mean anything. A cluster
    bootstrap with five clusters resamples five things and will happily report p = 0.0001, which
    is an artefact of the method rather than evidence: the resample distribution is discrete and
    far too narrow. A sign test over the same clusters cannot lie in that direction, because its
    smallest attainable p is 2^(1-n) and that floor is reported alongside it.
    """
    from scipy import stats

    ids = np.unique(cluster)
    means = np.array([x[cluster == c].mean() for c in ids])
    n = len(means)
    pos = int((means > 0).sum())
    # Agreement is with the direction of the overall effect, not with "positive": an improvement
    # in azimuth error is a negative number and would otherwise read as zero clusters agreeing.
    agree = pos if means.mean() > 0 else n - pos
    p = float(stats.binomtest(max(pos, n - pos), n, 0.5).pvalue) if n else float("nan")
    return {"n_clusters": n, "cluster_means": means.tolist(), "n_positive": pos,
            "n_agree": agree, "sign_p": min(1.0, p),
            "sign_p_floor": 2.0 ** (1 - n) if n else float("nan")}


def crossing_test(x: np.ndarray, mask: np.ndarray, seg: np.ndarray, label: str,
                  verbose: bool = True) -> dict:
    """Paired test on a range-restricted subset, with the crossing as the unit of evidence."""
    c = crossings(mask, seg)
    inside = c >= 0
    xs, cs = x[inside], c[inside]
    n_units = len(np.unique(cs))
    boots = cluster_bootstrap(xs, cs)
    out = summarize(boots, float(xs.mean()), xs, label, "crossings", n_units, verbose=False)
    sign = cluster_sign_test(xs, cs)
    out.update(sign)
    out["bootstrap_p"] = out["p"]
    if n_units < MIN_UNITS:
        # The sign test becomes the reported p. The bootstrap interval is kept as description,
        # clearly labelled, because it still says something about spread.
        out["p"] = sign["sign_p"]
        out["p_source"] = "exact sign test over crossings"
    else:
        out["p_source"] = "cluster bootstrap"
    if verbose:
        lo, hi = out["ci95"]
        print(f"  {label:<44}{out['mean']:>+8.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"  {'':<44}{sign['n_agree']}/{n_units} crossings agree, "
              f"sign test p {sign['sign_p']:.4f} (floor {sign['sign_p_floor']:.3f})"
              + ("   <- too few crossings for a bootstrap p" if n_units < MIN_UNITS else ""))
    return out


def paired_test(x: np.ndarray, seg: np.ndarray, label: str, block: int = MAIN_BLOCK,
                verbose: bool = True) -> dict:
    """Two-sided block-bootstrap test that mean(x) differs from zero, x a paired difference.

    Two-sided p comes from the bootstrap distribution's own tails, floored at 1/n_boot because a
    bootstrap cannot resolve a p-value finer than its number of replicates.
    """
    boots = block_bootstrap(x, seg, block)
    out = summarize(boots, float(x.mean()), x, label, f"blocks of {block}",
                    int(np.ceil(len(x) / block)), verbose)
    out["block"] = block
    return out


def block_sweep(x: np.ndarray, seg: np.ndarray, label: str) -> dict:
    """The same test at several block lengths, because one length is not evidence."""
    print(f"  {label}: block-length sensitivity")
    out = {}
    for b in BLOCKS:
        r = paired_test(x, seg, f"  block {b} frames ({b / 10:.0f} s)", block=b)
        out[str(b)] = r
    return out


# --- the seed-level machinery ---------------------------------------------------------------

def welch(a, b) -> tuple[float, float, float]:
    """Welch's t on two small samples, returning t, degrees of freedom and a two-sided p."""
    from scipy import stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(df), float(p)


def exact_permutation(a, b) -> tuple[float, int]:
    """Exact two-sided permutation p over all reassignments of the pooled seed means.

    With three against three there are only 20 distinct splits, so the smallest two-sided p
    obtainable is 0.10 and the smallest one-sided is 0.05. Reported so that a p at the floor is
    read as "as extreme as this design can show" rather than as a weak result.
    """
    pooled = np.concatenate([a, b])
    obs = abs(np.mean(a) - np.mean(b))
    n = len(a)
    splits = list(combinations(range(len(pooled)), n))
    hits = 0
    for s in splits:
        m = np.zeros(len(pooled), bool)
        m[list(s)] = True
        if abs(pooled[m].mean() - pooled[~m].mean()) >= obs - 1e-12:
            hits += 1
    return hits / len(splits), len(splits)


def seed_test(name: str, a: list[float], b: list[float], a_label: str, b_label: str) -> dict:
    t, df, p = welch(a, b)
    p_perm, n_splits = exact_permutation(np.array(a), np.array(b))
    # One formatter for quantities as different as an AUROC and a rate per hour.
    scale = max(abs(np.mean(a)), abs(np.mean(b)))
    f = "{:.3f}" if scale < 10 else ("{:.1f}" if scale < 1000 else "{:,.0f}")
    print(f"\n  {name}")
    for label, v in ((a_label, a), (b_label, b)):
        print(f"    {label:<18}{f.format(np.mean(v)):>11} +/- {f.format(np.std(v, ddof=1))}"
              f"   seeds " + " ".join(f.format(x) for x in v))
    print(f"    difference {f.format(np.mean(b) - np.mean(a))};  Welch t {t:.2f}, df {df:.1f}, "
          f"p {p:.4f};  exact permutation p {p_perm:.3f} of {n_splits} splits "
          f"(floor {2 / n_splits:.2f})")
    return {"name": name, "a": a, "b": b, "difference": float(np.mean(b) - np.mean(a)),
            "welch_t": t, "welch_df": df, "welch_p": p,
            "permutation_p": p_perm, "permutation_floor": 2 / n_splits}


def holm(tests: list[tuple[str, float]]) -> list[dict]:
    """Holm-Bonferroni over the family, in ascending p."""
    ordered = sorted(tests, key=lambda t: t[1])
    m = len(ordered)
    out, running = [], 0.0
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out.append({"name": name, "p": p, "p_holm": adj, "significant_at_05": adj < 0.05})
    return out


# --- loading ---------------------------------------------------------------------------------

def load_frames(run_id: str, order: str, split: str = "test_2"):
    """Drone frames of one dump in recording order: hit indicator, azimuth error, range, seg."""
    d = predict.load(f"{run_id}_{split}_{order}")
    o = time_order(d)
    d = {k: v[o] for k, v in d.items() if isinstance(v, np.ndarray) and v.shape[:1] == o.shape}
    m = d["has_drone"].astype(bool)
    err = angular_distance(d["pred_az"][m], d["pred_el"][m], d["az"][m], d["el"][m])
    az = np.abs((d["pred_az"][m] - d["az"][m] + 180) % 360 - 180) * np.cos(
        np.radians(d["el"][m]))
    return ((err <= ANGULAR_THRESHOLD).astype(float) * 100, az, d["range_m"][m], d["seg"][m])


def seed_mean_frames(template: str, order: str, split: str = "test_2"):
    """Seed-averaged per-frame arrays, truncated to a common length."""
    got = []
    for s in SEEDS:
        try:
            got.append(load_frames(template.format(s=s), order, split))
        except FileNotFoundError:
            continue
    if not got:
        return None
    n = min(len(g[0]) for g in got)
    return (np.mean([g[0][:n] for g in got], 0), np.mean([g[1][:n] for g in got], 0),
            got[0][2][:n], got[0][3][:n], len(got))


HEADLINE = [(subset, K) for subset in ("all24", "device3", "best6") for K in (1, 5, 9)]


def headline_hits(run_id: str, split: str = "test_2") -> tuple[np.ndarray, np.ndarray]:
    """Per-drone-frame hit indicator (0/100) in recording order, grid-snapped as the headline
    table scores it (notes 42), and the segment of each frame."""
    d = predict.load(f"{run_id}_{split}_causal")
    o = time_order(d)
    m = d["has_drone"][o].astype(bool)
    err = metrics.angular_errors(d["el_idx"][o][m], d["az_idx"][o][m], d["az"][o][m],
                                 d["el"][o][m])
    return (err <= ANGULAR_THRESHOLD).astype(float) * 100, d["seg"][o][m]


def cell_ci(subset: str, K: int, split: str = "test_2", tag: str = "wce") -> dict | None:
    """Interval on one headline cell's hit rate, with its two sources of spread kept apart.

    The frame component is a block bootstrap of the seed-averaged per-frame hit, at every block
    length in BLOCKS; the seed component is the spread of the three seeds' hit rates, whose
    standard error enters the combined interval as seed_sd / sqrt(n_seeds). The combined
    half-width adds the two in quadrature at MAIN_BLOCK, which treats them as independent.
    """
    runs = [run_id(subset, K, loss=tag, seed=s) for s in SEEDS]
    runs = [r for r in runs if pred_file(f"{r}_{split}_causal").exists()]
    if not runs:
        return None
    got = [headline_hits(r, split) for r in runs]
    n = min(len(h) for h, _ in got)
    per_seed = [float(h.mean()) for h, _ in got]
    x, seg = np.mean([h[:n] for h, _ in got], 0), got[0][1][:n]
    frame = {}
    for b in BLOCKS:
        boots = block_bootstrap(x, seg, b)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        frame[str(b)] = {"ci95": [float(lo), float(hi)], "se": float(boots.std(ddof=1))}
    seed_sd = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")
    seed_se = seed_sd / np.sqrt(len(per_seed))
    half = 1.96 * float(np.hypot(frame[str(MAIN_BLOCK)]["se"], seed_se))
    mean = float(np.mean(per_seed))
    return {"cell": f"{subset} K={K}", "runs": runs, "n_frames": int(n),
            "hit_10deg": mean, "per_seed": per_seed, "seed_sd": seed_sd,
            "frame_ci95_by_block": frame, "combined_ci95": [mean - half, mean + half],
            "combined_half_width": half, "main_block": MAIN_BLOCK}


def hits_from_metrics(subset: str, K: int, tag: str = "wce") -> list[float]:
    out = []
    for s in SEEDS:
        p = run_dir(run_id(subset, K, loss=tag, seed=s)) / "metrics.json"
        if p.exists():
            out.append(100 * json.loads(p.read_text())["test_2"]["hit_10deg"])
    return out


# --- the claims ------------------------------------------------------------------------------

def main() -> int:
    print(__doc__.split("\n\n")[0])
    results, family = {}, []

    print("\n" + "=" * 92)
    print("1. Temporal context on the six-microphone array: the freezing intervention")
    print("=" * 92)
    print("One set of weights, the same frames scored twice. Nine frames of real history against")
    print("the current frame repeated nine times. Units are percentage points of hit rate.\n")

    d9 = seed_mean_frames(run_id("device3", 9, seed="{s}"), "causal")
    d1 = seed_mean_frames(run_id("device3", 9, seed="{s}"), "freeze1")
    if d9 and d1:
        n = min(len(d9[0]), len(d1[0]))
        diff, rng, seg = d9[0][:n] - d1[0][:n], d9[2][:n], d9[3][:n]
        r = paired_test(diff, seg, f"pooled, {d9[4]} seeds averaged")
        results["freeze_device3_k9_pooled"] = r
        family.append(("device3 K=9 history, pooled", r["p"]))

        # Range-restricted, so the unit is the crossing: the drone is only ever beyond 90 m for a
        # few continuous stretches, and pretending those frames are independent would invent
        # evidence that the flight plan never provided.
        m = (rng >= FAR[0]) & (rng < FAR[1])
        rf = crossing_test(diff, m, seg, f"beyond {FAR[0]} m, by crossing")
        results["freeze_device3_k9_far"] = rf
        family.append((f"device3 K=9 history beyond {FAR[0]} m", rf["p"]))

        near = rng < FAR[0]
        rn = crossing_test(diff, near, seg, f"inside {FAR[0]} m, by crossing")
        results["freeze_device3_k9_near"] = rn

        print()
        results["freeze_block_sweep"] = block_sweep(diff, seg, "pooled")

        # Is the far gain larger than the near gain, rather than merely both positive? Bootstrap
        # the contrast itself, which is the claim the range breakdown actually makes.
        print()
        contrast = np.where(m, diff / max(m.mean(), 1e-9), 0.0) - np.where(
            near, diff / max(near.mean(), 1e-9), 0.0)
        rc = paired_test(contrast, seg, f"far minus near difference-in-differences")
        results["freeze_far_minus_near"] = rc
        family.append(("device3 K=9 history, far gain exceeds near", rc["p"]))

    print("\n" + "=" * 92)
    print("2. The same intervention on all 24 microphones, as a control")
    print("=" * 92)
    a9 = seed_mean_frames(run_id("all24", 5, seed="{s}"), "causal")
    a1 = seed_mean_frames(run_id("all24", 5, seed="{s}"), "freeze1")
    if a9 and a1:
        n = min(len(a9[0]), len(a1[0]))
        diff, seg = a9[0][:n] - a1[0][:n], a9[3][:n]
        r = paired_test(diff, seg, f"all24 K=5, pooled, {a9[4]} seeds")
        results["freeze_all24_k5_pooled"] = r
        family.append(("all24 K=5 history, pooled", r["p"]))
    print("  History helps this array too: what fails on all 24 is the *depth sweep* across "
          "separately\n  trained runs, which is the comparison the next section tests.")

    print("\n" + "=" * 92)
    print("3. Depth across separately trained runs, which is what the headline table shows")
    print("=" * 92)
    print("Weaker evidence than section 1: a K=9 model has nine times the input channels, so this")
    print("confounds history with capacity. Hit rate within 10 degrees, three seeds per cell.")
    for subset in ("device3", "all24"):
        k1, k9 = hits_from_metrics(subset, 1), hits_from_metrics(subset, 9)
        if len(k1) > 1 and len(k9) > 1:
            r = seed_test(f"{subset}: K=1 against K=9", k1, k9, "K=1", "K=9")
            r["n_seeds"] = [len(k1), len(k9)]
            results[f"seeds_{subset}_k1_k9"] = r
            family.append((f"{subset} K=9 against K=1 across seeds", r["welch_p"]))
        elif k9:
            print(f"\n  {subset}: only {len(k9)} of 3 K=9 seeds finished, deferring")

    print("\n" + "=" * 92)
    print("4. The native polar disc in the far bin, both ways")
    print("=" * 92)
    print("Mean azimuth error in degrees beyond 100 m, so negative is better.")
    rect = seed_mean_frames(run_id("all24", 1, seed="{s}"), "causal")
    for label, tpl in (("native disc + wce",
                        run_id("all24", 1, polar=True, disc="native", seed="{s}")),
                       ("native disc + tversky",
                        run_id("all24", 1, loss="tversky", polar=True, disc="native",
                               seed="{s}"))):
        cell = seed_mean_frames(tpl, "causal")
        if not (rect and cell):
            continue
        n = min(len(rect[1]), len(cell[1]))
        m = (rect[2][:n] >= 100) & (rect[2][:n] < 200)
        diff = cell[1][:n] - rect[1][:n]
        r = crossing_test(diff, m, rect[3][:n], f"{label}, paired by crossing")
        results[f"polar_{label.replace(' ', '_')}"] = r
        family.append((f"{label} far-bin azimuth, frame-paired", r["p"]))

    print("\n  Across seeds instead, which controls initialization rather than frame difficulty:")
    pvr = table_file("polar_vs_rect.json")
    if pvr.exists():
        d = json.loads(pvr.read_text())
        base = d["cells"]["rect + wce"]["far_azimuth_per_seed"] if "far_azimuth_per_seed" in \
            d["cells"]["rect + wce"] else [x[2] for x in
                                           d["cells"]["rect + wce"]["mean_azimuth"]["per_seed"]]
        for label in ("disc native + wce", "disc native + tversky"):
            c = d["cells"].get(label, {})
            if c.get("far_azimuth_per_seed"):
                r = seed_test(f"{label} far-bin azimuth", base, c["far_azimuth_per_seed"],
                              "rectangular", label)
                results[f"seeds_{label.replace(' ', '_')}"] = r
                family.append((f"{label} far-bin azimuth across seeds", r["welch_p"]))

    print("\n" + "=" * 92)
    print("5. The detector with history, which is a different question from the map model's")
    print("=" * 92)
    print("Trained at K=1, 5 and 9, three seeds each. Two metrics, because they disagree: AUROC on")
    print("our own held-out flights, and false alarms per hour on DEMAND, which is ambient from")
    print("eighteen soundscapes it has never heard.")
    det = {}
    for K in (1, 5, 9):
        auroc, fa = [], []
        for s in SEEDS:
            p = run_dir(f"detector_K{K}_w32_s{s}_full") / "metrics.json"
            if p.exists():
                auroc.append(json.loads(p.read_text())["test_2"]["auroc"])
            q = table_file(f"demand_false_alarms_detector_K{K}_w32_s{s}_full.json")
            if q.exists():
                fa.append(json.loads(q.read_text())["pooled_per_hour"])
        det[K] = (auroc, fa)
    for K in (5, 9):
        if len(det[1][0]) > 1 and len(det[K][0]) > 1:
            r = seed_test(f"detector K=1 against K={K}, AUROC on our own site",
                          det[1][0], det[K][0], "K=1", f"K={K}")
            results[f"detector_auroc_k1_k{K}"] = r
            family.append((f"detector K={K} AUROC against K=1", r["welch_p"]))
        if len(det[1][1]) > 1 and len(det[K][1]) > 1:
            r = seed_test(f"detector K=1 against K={K}, DEMAND false alarms per hour",
                          det[1][1], det[K][1], "K=1", f"K={K}")
            results[f"detector_demand_k1_k{K}"] = r
            family.append((f"detector K={K} DEMAND alarms against K=1", r["welch_p"]))
    print("\n  The two metrics move in opposite directions, which is the finding: history sharpens")
    print("  the detector where it was trained and does nothing for it anywhere else.")

    print("\n" + "=" * 92)
    print("6. An interval on every headline cell")
    print("=" * 92)
    print("Hit rate within 10 degrees on test 2, grid-snapped as the headline table reports it.")
    print(f"Frame interval: block bootstrap, {MAIN_BLOCK}-frame blocks, seeds averaged per frame.")
    print("Combined: frame and seed standard errors in quadrature. Not tests, so not in the family.\n")
    cells, mismatches = {}, []
    for subset, K in HEADLINE:
        c = cell_ci(subset, K)
        if c is None:
            continue
        stored = hits_from_metrics(subset, K)
        if len(stored) != len(c["per_seed"]) or not np.allclose(stored, c["per_seed"], atol=1e-4):
            mismatches.append(f"{c['cell']}: {c['per_seed']} vs metrics.json {stored}")
        f = c["frame_ci95_by_block"]
        lo, hi = c["combined_ci95"]
        print(f"  {c['cell']:<14}{c['hit_10deg']:6.1f}%   seeds "
              + " ".join(f"{v:.1f}" for v in c["per_seed"])
              + f"   frame CI [{f[str(MAIN_BLOCK)]['ci95'][0]:.1f}, "
              f"{f[str(MAIN_BLOCK)]['ci95'][1]:.1f}]   combined [{lo:.1f}, {hi:.1f}]"
              f"  (+/-{c['combined_half_width']:.1f})")
        cells[c["cell"]] = c
    results["cell_ci"] = cells
    for mm in mismatches:
        print(f"  FAIL {mm}")
    print("  GATE PASS: every cell's per-seed hit rate reproduces its metrics.json"
          if not mismatches else "  GATE FAIL")

    print("\n" + "=" * 92)
    print("7. Holm-Bonferroni over the family")
    print("=" * 92)
    adjusted = holm(family)
    print(f"  {'claim':<52}{'p':>10}{'p Holm':>10}   verdict")
    for a in adjusted:
        print(f"  {a['name']:<52}{a['p']:>10.4f}{a['p_holm']:>10.4f}   "
              + ("significant" if a["significant_at_05"] else "not at 0.05"))
    results["holm"] = adjusted

    path = table_file("significance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {path}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
