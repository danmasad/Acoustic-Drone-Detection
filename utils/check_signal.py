"""Signal and label checks on test 2: label coverage, audio-to-GPS alignment, azimuth bias.

The review (Phase 1) asked for four checks nobody had run. Three of them need only the committed
per-frame dumps, because every dump carries the labels, the prediction, the segment and the
frame index:

1. **Coverage.** Histograms of range, azimuth and elevation over test 2's drone frames: what the
   headline number is an average over.
2. **Alignment lag.** Row k of the GPS log is paired with audio frame k. Sound takes r / c to
   arrive, one 100 ms frame per 34 m, so at 150 m the audio describes where the drone was four
   frames earlier. Each predictor is scored against labels shifted by L frames, on one common set
   of frames for every L, and the best L is found per range bin. Propagation predicts a lag that
   grows with range (prediction k matches label k - r / (c dt)); a clock offset between the
   recorders and the GPS predicts the same lag at every range. The emission-time labels of
   `labels.propagation_lagged` (the `label_lag` protocol flag) are scored alongside.
3. **Signed azimuth bias.** The median of predicted minus true azimuth, wrapped, over frames
   whose error is already under 20 degrees (a phantom peak is not a bias) and below 60 degrees
   of elevation (where azimuth still means something). A constant offset at every bearing is an
   array-heading error; one that flips sign with bearing is not.

Two more need the STFT cache: a **spectrogram** of the drone segment (`--spectrogram`), and the
**recorder sync** check notes 46 asks for (`--recorders`): the four recorders' clock offsets,
fitted from GCC-PHAT delays of every microphone pair without labels, against a gate of one
sample. A recorder offset mis-steers only subsets that span recorders, which is one of the two
candidate causes of `all24`'s near-field failure.

Gate: at L = 0 every predictor's hit rate on all its drone frames reproduces the stored one
(`results/baseline.json` for delay-and-sum, each run's `metrics.json` for the U-Net) to 1e-6.
`--self-test` plants offsets of 3, -2 and 5.5 samples in a synthetic recording and requires
the fit to recover them within half a sample.

Run: .venv/bin/python utils/check_signal.py [--spectrogram] [--recorders] [--self-test]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics, predict                              # noqa: E402
from src.config import (ANGULAR_THRESHOLD, C, CSV_HZ, FIGURES,  # noqa: E402
                        RESULTS, RUNS)
from src.geometry import angular_distance                     # noqa: E402
from src.labels import propagation_lagged                     # noqa: E402

SPLIT = "test_2"
SEEDS = (0, 1, 2)
DAS = ("all24", "device3")
UNETS = ("all24_K1_w32_b16_wce", "device3_K1_w32_b16_wce")
LAGS = np.arange(-40, 41)
RANGE_BINS = ((0, 30), (30, 60), (60, 100), (100, 200), (200, np.inf))
LOCKED_DEG = 15.0          # median error at lag 0 below which a lag curve is read
BIAS_MAX_ERR = 20.0
BIAS_MAX_EL = 60.0
BLOCK = 300
N_BOOT = 2000
FRAME_M = C / CSV_HZ       # metres of propagation per frame
C_MS = C                   # m/s, for the recorder-sync fit
SR_HZ = 48000.0
SYNC_NFFT = 4800           # one frame; the lag axis is then in samples
SYNC_OUTLIER = 3.0         # samples
SYNC_FRAMES = 300


def drone_segment(d: dict) -> dict:
    """The dump's drone segment as frame-indexed arrays, NaN where the dump has no frame."""
    seg = int(np.bincount(d["seg"][d["has_drone"].astype(bool)]).argmax())
    sel = d["seg"] == seg
    frames = d["frame"][sel]
    n = int(frames.max()) + 1
    out = {"seg": seg, "seg_id": str(d["seg_ids"][seg]), "n": n, "present": np.zeros(n, bool)}
    out["present"][frames] = True
    for k in ("az", "el", "range_m", "pred_az", "pred_el", "el_idx", "az_idx", "has_drone"):
        a = np.full(n, np.nan)
        a[frames] = d[k][sel]
        out[k] = a
    return out


def grid_pred(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """Grid-snapped predicted direction, as the headline table scores it."""
    ok = s["present"]
    pa, pe = np.full(s["n"], np.nan), np.full(s["n"], np.nan)
    pa[ok] = metrics.AZ[s["az_idx"][ok].astype(int)]
    pe[ok] = metrics.EL[s["el_idx"][ok].astype(int)]
    return pa, pe


def lag_curves(s: dict, pa: np.ndarray, pe: np.ndarray) -> dict:
    """Median angular error and hit rate against labels shifted by each lag, per range bin."""
    n = s["n"]
    drone = s["present"] & (s["has_drone"] == 1)
    lo, hi = -LAGS.min(), n - LAGS.max()
    k = np.arange(lo, hi)
    common = drone[k].copy()
    for L in LAGS:
        common &= drone[k + L]
    k = k[common]
    r = s["range_m"][k]
    out = {}
    for b0, b1 in ((0, np.inf),) + RANGE_BINS:
        kb = k[(r >= b0) & (r < b1)]
        if len(kb) < 100:
            continue
        med, hit = [], []
        for L in LAGS:
            err = angular_distance(pa[kb], pe[kb], s["az"][kb + L], s["el"][kb + L])
            med.append(float(np.median(err)))
            hit.append(float(np.mean(err <= ANGULAR_THRESHOLD)))
        i = int(np.argmin(med))
        zero = int(np.flatnonzero(LAGS == 0)[0])
        # A lag is only measurable where the predictor tracks the drone: otherwise the error
        # is dominated by phantom peaks, the curve is flat, and its minimum wanders to the edge
        # of the sweep.
        locked = med[zero] < LOCKED_DEG and 0 < i < len(LAGS) - 1
        best = None
        if locked:
            y0, y1, y2 = med[i - 1], med[i], med[i + 1]
            den = y0 - 2 * y1 + y2
            best = float(LAGS[i]) + (0.5 * (y0 - y2) / den if den > 0 else 0.0)
        r_med = float(np.median(s["range_m"][kb]))
        key = "all" if np.isinf(b1) and b0 == 0 else f"{b0:g}-{b1:g}"
        out[key] = {"n": int(len(kb)), "median_range_m": r_med, "locked": bool(locked),
                    "best_lag_frames": best, "propagation_lag_frames": -r_med / FRAME_M,
                    # What is left once propagation is removed: a clock offset if constant.
                    "residual_lag_frames": None if best is None else best + r_med / FRAME_M,
                    "median_err_deg": med, "hit_10deg": hit}
    return out


def lagged_hit(s: dict, pa: np.ndarray, pe: np.ndarray) -> dict:
    """Hit rate against the dataset's labels and against emission-time labels, same frames."""
    meta = {"az": s["az"].astype(np.float32), "el": s["el"].astype(np.float32),
            "range_m": s["range_m"].astype(np.float32),
            "has_drone": np.nan_to_num(s["has_drone"]).astype(np.uint8)}
    lag = propagation_lagged(meta)
    m = s["present"] & (s["has_drone"] == 1) & ~np.isnan(lag["az"])
    e0 = angular_distance(pa[m], pe[m], s["az"][m], s["el"][m])
    e1 = angular_distance(pa[m], pe[m], lag["az"][m].astype(float), lag["el"][m].astype(float))
    far = s["range_m"][m] >= 100
    return {"n": int(m.sum()),
            "hit_dataset_labels": float(np.mean(e0 <= ANGULAR_THRESHOLD)),
            "hit_emission_labels": float(np.mean(e1 <= ANGULAR_THRESHOLD)),
            "hit_dataset_labels_100m_plus": float(np.mean(e0[far] <= ANGULAR_THRESHOLD)),
            "hit_emission_labels_100m_plus": float(np.mean(e1[far] <= ANGULAR_THRESHOLD))}


def block_ci(x: np.ndarray, blocks: np.ndarray, rng) -> list[float]:
    ids = np.unique(blocks)
    groups = [x[blocks == b] for b in ids]
    boots = [np.median(np.concatenate([groups[j] for j in rng.integers(0, len(ids), len(ids))]))
             for _ in range(N_BOOT)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def azimuth_bias(s: dict, pa: np.ndarray, pe: np.ndarray, rng, lag: int = 0) -> dict:
    """Signed azimuth error against labels shifted by `lag` frames. A lag on a drone that
    circles mostly one way is itself a signed azimuth error, so the bias is also reported at
    the measured lag."""
    n = s["n"]
    k = np.arange(max(0, -lag), min(n, n - lag))
    m = s["present"][k] & (s["has_drone"][k] == 1) & (s["has_drone"][k + lag] == 1)
    idx = k[m]
    ta, te = s["az"][idx + lag], s["el"][idx + lag]
    err = angular_distance(pa[idx], pe[idx], ta, te)
    keep = (err <= BIAS_MAX_ERR) & (te < BIAS_MAX_EL)
    idx, ta = idx[keep], ta[keep]
    signed = ((pa[idx] - ta + 180) % 360) - 180
    blocks = idx // BLOCK
    out = {"lag_frames": int(lag), "n": int(len(idx)), "median_deg": float(np.median(signed)),
           "ci95": block_ci(signed, blocks, rng), "by_range": {}, "by_bearing": {}}
    r = s["range_m"][idx + lag]
    for b0, b1 in RANGE_BINS:
        sel = (r >= b0) & (r < b1)
        if sel.sum() >= 100:
            out["by_range"][f"{b0:g}-{b1:g}"] = {"n": int(sel.sum()),
                                                 "median_deg": float(np.median(signed[sel]))}
    oct_ = (((ta + 180) // 45).astype(int)) % 8
    for o in range(8):
        sel = oct_ == o
        if sel.sum() >= 100:
            out["by_bearing"][f"{-180 + 45 * o}..{-135 + 45 * o}"] = {
                "n": int(sel.sum()), "median_deg": float(np.median(signed[sel]))}
    return out


def coverage(s: dict) -> dict:
    m = s["present"] & (s["has_drone"] == 1)
    r, a, e = s["range_m"][m], s["az"][m], s["el"][m]
    q = lambda x: {p: float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)}   # noqa: E731
    return {"n": int(m.sum()), "range_m": q(r), "elevation_deg": q(e),
            "share_beyond_100m": float(np.mean(r >= 100)),
            "share_above_60deg": float(np.mean(e >= 60)),
            "azimuth_octant_share": np.bincount((((a + 180) // 45).astype(int)) % 8,
                                                minlength=8).tolist()}


def merge(per: list[dict]) -> dict:
    """Seeds of one cell: curves and scalars averaged; a bin is locked only if every seed is;
    the bias is the first seed's, with every seed's median alongside. DAS has one member."""
    if len(per) == 1:
        return per[0]

    def avg(vals):
        if isinstance(vals[0], list):
            return np.mean(vals, axis=0).tolist()
        if isinstance(vals[0], bool):
            return all(vals)
        if any(v is None for v in vals):
            return None
        return float(np.mean(vals))

    lag = {k: {f: avg([p["lag"][k][f] for p in per]) for f in per[0]["lag"][k]}
           for k in per[0]["lag"]}
    for v in lag.values():
        if not v["locked"]:
            v["best_lag_frames"] = v["residual_lag_frames"] = None
    out = {"lag": lag,
           "emission": {f: avg([p["emission"][f] for p in per]) for f in per[0]["emission"]}}
    for key in ("azimuth_bias", "azimuth_bias_at_lag"):
        out[key] = per[0][key] | {"per_seed_median_deg": [p[key]["median_deg"] for p in per],
                                  "per_seed_lag_frames": [p[key]["lag_frames"] for p in per]}
    return out


def predictors() -> dict:
    out = {}
    for arr in DAS:
        out[f"DAS {arr}"] = {"dumps": [f"baseline_{arr}_{SPLIT}"], "stored": [
            json.loads((RESULTS / "baseline.json").read_text())[f"{arr}/{SPLIT}"]["peak_db"]
            ["hit_10deg"]]}
    for cell in UNETS:
        runs = [f"{cell}_s{s}" for s in SEEDS if (predict.PREDS / f"{cell}_s{s}_{SPLIT}_causal.npz")
                .exists()]
        out[f"U-Net {cell.split('_')[0]} K=1"] = {
            "dumps": [f"{r}_{SPLIT}_causal" for r in runs],
            "stored": [json.loads((RUNS / r / "metrics.json").read_text())[SPLIT]["hit_10deg"]
                       for r in runs]}
    return out


def gcc_lags(X: np.ndarray, freqs: np.ndarray, max_lag: np.ndarray) -> np.ndarray:
    """(M, F) spectra of one frame -> (P,) GCC-PHAT delay in samples for every pair i < j,
    positive when microphone i hears the wavefront later than j. Searched within each pair's
    physical limit, peak refined by a parabola."""
    M = X.shape[0]
    iu, ju = np.triu_indices(M, 1)
    cross = X[iu] * np.conj(X[ju])
    cross /= np.abs(cross) + 1e-12
    full = np.zeros((len(iu), SYNC_NFFT // 2 + 1), complex)
    full[:, np.round(freqs / (SR_HZ / SYNC_NFFT)).astype(int)] = cross
    cc = np.fft.irfft(full, n=SYNC_NFFT, axis=1)
    lags = np.empty(len(iu))
    for p in range(len(iu)):
        w = int(np.ceil(max_lag[p])) + 2
        idx = np.arange(-w, w + 1)
        y = cc[p, idx % SYNC_NFFT]
        i = int(np.argmax(y))
        d = 0.0
        if 0 < i < len(y) - 1:
            den = y[i - 1] - 2 * y[i] + y[i + 1]
            d = 0.5 * (y[i - 1] - y[i + 1]) / den if den < 0 else 0.0
        lags[p] = idx[i] + d
    return lags


def recorder_offsets(stft: np.ndarray, freqs: np.ndarray, pos: np.ndarray, frames,
                     device_of: np.ndarray, iterations: int = 30) -> dict:
    """Clock offset of each recorder against the first, in samples, from pairwise delays alone;
    positive when that recorder hears everything later than a synchronized one would.

    Every frame's pair delays obey tau_ij = (m_i - m_j) . u_k / c + o_d(i) - o_d(j) under a plane
    wave from u_k. Pairs inside one recorder carry no offset and pin u_k down; pairs across
    recorders then expose the offsets. Solved by alternating least squares, dropping pairs whose
    residual exceeds SYNC_OUTLIER samples (a wrong GCC peak, which harmonic sources produce).
    No labels are used, so a label misalignment cannot leak in."""
    M = pos.shape[0]
    iu, ju = np.triu_indices(M, 1)
    A = (pos[iu] - pos[ju]) / C_MS * SR_HZ                       # (P, 3) samples per unit u
    max_lag = np.linalg.norm(pos[iu] - pos[ju], axis=1) / C_MS * SR_HZ
    tau = np.stack([gcc_lags(stft[k], freqs, max_lag) for k in frames])      # (N, P)
    n_dev = int(device_of.max()) + 1
    D = np.zeros((len(iu), n_dev - 1))
    for d in range(1, n_dev):
        D[:, d - 1] = (device_of[iu] == d).astype(float) - (device_of[ju] == d).astype(float)
    o = np.zeros(n_dev - 1)
    keep = np.ones_like(tau, bool)
    for _ in range(iterations):
        r = tau - D @ o
        U = np.stack([np.linalg.lstsq(A[keep[k]], r[k, keep[k]], rcond=None)[0]
                      for k in range(len(frames))])
        resid_u = tau - U @ A.T
        cross = np.abs(D).sum(1) > 0
        rows = [(D[cross & keep[k]], resid_u[k, cross & keep[k]]) for k in range(len(frames))]
        o = np.linalg.lstsq(np.concatenate([a for a, _ in rows]),
                            np.concatenate([b for _, b in rows]), rcond=None)[0]
        keep = np.abs(tau - U @ A.T - D @ o) <= SYNC_OUTLIER
    res = tau - U @ A.T - D @ o
    return {"offsets_samples": [0.0] + o.tolist(), "n_frames": len(frames),
            "pairs_kept": float(keep.mean()),
            "median_abs_residual_samples": float(np.median(np.abs(res[keep])))}


def sync_self_test() -> list[str]:
    """Plant offsets on a synthetic 24-mic recording and recover them."""
    from src.geometry import mic_positions
    rng = np.random.default_rng(0)
    align = {"mic_array": {"a_": [0, 120, 240], "a2_": [30, 90, 150, 210, 270, 330],
                           "h_": [0.05, 0.25, 0.45, 0.65, 0.85, 1.05],
                           "r_": [0.25, 0.35, 0.45, 0.55, 0.60, 0.65]}}
    pos = mic_positions(align)
    device_of = np.arange(24) // 6
    planted = np.array([0.0, 3.0, -2.0, 5.5])
    freqs = np.arange(200.0, 4000.0 + 1, 10.0)
    frames = []
    for _ in range(40):
        az, el = np.radians(rng.uniform(-180, 180)), np.radians(rng.uniform(10, 70))
        u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        adv = pos @ u / C_MS * SR_HZ - planted[device_of]        # samples of arrival advance
        s = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
        X = s[None] * np.exp(2j * np.pi * freqs[None] * adv[:, None] / SR_HZ)
        X += 0.3 * (rng.normal(size=X.shape) + 1j * rng.normal(size=X.shape))
        frames.append(X)
    got = recorder_offsets(np.stack(frames), freqs, pos, range(40), device_of)
    err = np.abs(np.array(got["offsets_samples"]) - (planted - planted[0]))
    return [] if err.max() <= 0.5 else [f"recorder offsets {got['offsets_samples']} vs "
                                        f"planted {(planted - planted[0]).tolist()}"]


def recorder_sync(seg_id: str) -> dict:
    """The four recorders' clock offsets on test 2's drone segment. Frames at 40-100 m, where
    a plane wave is within 0.7 samples of the true wavefront across the 1.30 m aperture and the
    drone is still loud enough to correlate."""
    from src.config import segments
    from src.geometry import mic_positions
    from src.stft_cache import load_cache

    seg = next(s for s in segments() if s.id == seg_id)
    stft, meta = load_cache(seg)
    r = np.asarray(meta["range_m"])
    ok = np.flatnonzero((np.asarray(meta["has_drone"]) == 1) & (r >= 40) & (r <= 100))
    frames = ok[np.linspace(0, len(ok) - 1, min(SYNC_FRAMES, len(ok))).astype(int)]
    out = recorder_offsets(np.asarray(stft), np.asarray(meta["freqs"]),
                           mic_positions(seg.align), frames, np.arange(24) // 6)
    out["gate_samples"] = 1.0
    out["in_sync"] = bool(np.max(np.abs(out["offsets_samples"])) <= 1.0)
    return out


def spectrogram(seg_id: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.config import segments
    from src.stft_cache import load_cache

    seg = next(s for s in segments() if s.id == seg_id)
    stft, meta = load_cache(seg)
    freqs = np.asarray(meta["freqs"])
    p = 10 * np.log10(np.mean(np.abs(np.asarray(stft)) ** 2, axis=1) + 1e-12)   # (T, F)
    t = np.arange(len(p)) / CSV_HZ
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": (3, 1)})
    a0.imshow(p.T, aspect="auto", origin="lower", cmap="magma",
              extent=(float(t[0]), float(t[-1]), float(freqs[0]), float(freqs[-1])),
              vmin=np.percentile(p, 5), vmax=np.percentile(p, 99.5))
    a0.set_ylabel("frequency (Hz)")
    a0.set_ylim(float(freqs[0]), float(freqs[-1]))
    a0.set_title(f"{seg_id}: power averaged over the 24 microphones")
    a1.plot(t, meta["range_m"][:len(t)], lw=0.8)
    a1.set_ylabel("range (m)")
    a1.set_xlabel("time (s)")
    fig.tight_layout()
    path = FIGURES / f"spectrogram_{SPLIT}.png"
    fig.savefig(path, dpi=130)
    return path


def plot(report: dict, segs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = next(iter(segs.values()))
    m = s["present"] & (s["has_drone"] == 1)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    ax[0, 0].hist(s["range_m"][m], bins=50)
    ax[0, 0].set_xlabel("range (m)")
    ax[0, 1].hist(s["az"][m], bins=np.arange(-180, 181, 10))
    ax[0, 1].set_xlabel("azimuth (deg)")
    ax[0, 2].hist(s["el"][m], bins=np.arange(0, 91, 4))
    ax[0, 2].set_xlabel("elevation (deg)")
    ax[0, 0].set_ylabel("drone frames")

    for name, r in report["predictors"].items():
        curve = r["lag"]["all"]
        ax[1, 0].plot(LAGS, curve["median_err_deg"], marker=".", label=name)
        bins = [k for k in r["lag"] if k != "all" and r["lag"][k]["locked"]]
        ax[1, 1].plot([r["lag"][k]["median_range_m"] for k in bins],
                      [r["lag"][k]["best_lag_frames"] for k in bins], marker="o", label=name)
        b = r["azimuth_bias"]["by_range"]
        ax[1, 2].plot([float(k.split("-")[0]) for k in b], [v["median_deg"] for v in b.values()],
                      marker="o", label=name)
    xs = np.linspace(0, 250, 50)
    ax[1, 1].plot(xs, -xs / FRAME_M, "k--", lw=1, label="propagation, -r / (c dt)")
    ax[1, 1].set_title(f"bins with median error under {LOCKED_DEG:g} deg", fontsize=9)
    ax[1, 0].set_ylim(0, 60)
    ax[1, 0].set_xlabel("label shift L (frames; prediction k vs label k + L)")
    ax[1, 0].set_ylabel("median angular error (deg)")
    ax[1, 1].set_xlabel("median range of bin (m)")
    ax[1, 1].set_ylabel("best lag (frames)")
    ax[1, 2].axhline(0, color="0.6", lw=0.8)
    ax[1, 2].set_xlabel("range bin start (m)")
    ax[1, 2].set_ylabel("median signed azimuth error (deg)")
    for a in ax[1]:
        a.legend(fontsize=7)
    fig.suptitle(f"{SPLIT}: label coverage (top); alignment lag and azimuth bias (bottom)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"signal_checks_{SPLIT}.png"
    fig.savefig(path, dpi=140)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spectrogram", action="store_true", help="needs the STFT cache")
    ap.add_argument("--recorders", action="store_true",
                    help="recorder clock offsets from pairwise delays; needs the STFT cache")
    ap.add_argument("--self-test", action="store_true",
                    help="recover planted recorder offsets from a synthetic recording")
    a = ap.parse_args()
    if a.self_test:
        bad = sync_self_test()
        print("GATE " + ("FAIL" if bad else "PASS") + " (planted recorder offsets recovered)")
        for b in bad:
            print(f"  {b}")
        return 1 if bad else 0
    rng = np.random.default_rng(0)
    report, failures, segs = {"split": SPLIT, "predictors": {}}, [], {}
    for name, spec in predictors().items():
        per = []
        for dump, stored in zip(spec["dumps"], spec["stored"]):
            d = predict.load(dump)
            m = d["has_drone"].astype(bool)
            hit0 = float(np.mean(metrics.angular_errors(d["el_idx"][m], d["az_idx"][m],
                                                        d["az"][m], d["el"][m])
                                 <= ANGULAR_THRESHOLD))
            if abs(hit0 - stored) > 1e-6:
                failures.append(f"{dump}: lag-0 hit {hit0:.6f} vs stored {stored:.6f}")
            s = drone_segment(d)
            segs[dump] = s
            pa, pe = grid_pred(s)
            lag = lag_curves(s, pa, pe)
            shift = lag["all"]["best_lag_frames"]
            shift = int(round(shift)) if shift is not None else 0
            per.append({"lag": lag, "emission": lagged_hit(s, pa, pe),
                        "azimuth_bias": azimuth_bias(s, pa, pe, rng),
                        "azimuth_bias_at_lag": azimuth_bias(s, pa, pe, rng, lag=shift)})
        merged = merge(per)
        merged["members"] = spec["dumps"]
        report["predictors"][name] = merged

        print(f"\n{name} ({len(per)} dump{'s' if len(per) > 1 else ''})")
        for k, v in merged["lag"].items():
            if not v["locked"]:
                print(f"  range {k:>8} m  n {v['n']:>5}  not locked (median error "
                      f"{v['median_err_deg'][int(np.flatnonzero(LAGS == 0)[0])]:.0f} deg)")
                continue
            print(f"  range {k:>8} m  n {v['n']:>5}  best lag {v['best_lag_frames']:+5.2f} "
                  f"frames, propagation {v['propagation_lag_frames']:+5.2f}, residual "
                  f"{v['residual_lag_frames']:+5.2f}")
        e = merged["emission"]
        print(f"  hit@10 dataset labels {e['hit_dataset_labels']:.4f}, emission-time "
              f"{e['hit_emission_labels']:.4f}; beyond 100 m {e['hit_dataset_labels_100m_plus']:.4f}"
              f" -> {e['hit_emission_labels_100m_plus']:.4f}")
        for key in ("azimuth_bias", "azimuth_bias_at_lag"):
            b = merged[key]
            print(f"  signed azimuth bias at lag {b['lag_frames']:+d}: {b['median_deg']:+.2f} deg "
                  f"[{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}] on {b['n']} frames")

    report["coverage"] = coverage(next(iter(segs.values())))
    c = report["coverage"]
    print(f"\ncoverage: {c['n']} drone frames, range median {c['range_m'][50]:.0f} m "
          f"(5-95%: {c['range_m'][5]:.0f}-{c['range_m'][95]:.0f}), "
          f"{100 * c['share_beyond_100m']:.0f}% beyond 100 m, "
          f"{100 * c['share_above_60deg']:.0f}% above 60 deg elevation")

    seg_id = next(iter(segs.values()))["seg_id"]
    path = RESULTS / "signal_checks.json"
    if a.recorders:
        from src.config import segments
        report["recorder_sync"] = recorder_sync(seg_id)
        report["recorder_sync_by_session"] = {}
        for seg in segments():
            if seg.has_drone:
                report["recorder_sync_by_session"][seg.id] = recorder_sync(seg.id)
        rs = report["recorder_sync"]
        print(f"recorder offsets on {seg_id} (samples, vs recorder 1): "
              f"{', '.join(f'{o:+.2f}' for o in rs['offsets_samples'])}; "
              f"{'within' if rs['in_sync'] else 'OUTSIDE'} the one-sample gate "
              f"(kept {100 * rs['pairs_kept']:.0f}% of pairs)")
        for sid, r in report["recorder_sync_by_session"].items():
            print(f"  {sid}: {[round(o, 2) for o in r['offsets_samples']]}  "
                  f"{'in' if r['in_sync'] else 'OUT'} sync")
    elif path.exists():
        prev = json.loads(path.read_text()).get("recorder_sync")
        if prev:
            report["recorder_sync"] = prev
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {path}\nwrote {plot(report, segs)}")
    if a.spectrogram:
        print(f"wrote {spectrogram(seg_id)}")
    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: lag 0 reproduces every stored hit rate" if not failures else "GATE FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
