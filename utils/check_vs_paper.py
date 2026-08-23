"""Do our numbers reproduce the reference paper's, or did we lose something in reimplementation?

Their results live in figures rather than tables, so the comparison has to be built rather than
looked up. Three things had to match before any number is comparable:

- **Their bins**, 0-50 / 50-100 / 100-200 m, not our 30 m bins.
- **Mean** angular error, not our median. On a heavy-tailed error distribution these differ by
  a lot, and quoting our median against their mean would flatter us by ~16 degrees.
- **Their false-positive rule.** They never threshold an energy; they call a frame a detection
  when the direction estimate is *stable*, defined as the centroid standard deviation over the
  last 3 observations staying under 10 degrees (section 4.1). On five minutes of no-drone audio
  DAS was stable 67.0% of the time and their U-Net 14.9%. Reproducing that means implementing
  their rule, not ours. The paper does not say how the standard deviation of three directions
  is reduced to one number, so we use the root-mean-square angular distance to the mean
  direction, which is the natural spherical reading.

Paper values below are read off Fig. 5 of arXiv:2508.00307v1 to the nearest degree; they are
bar heights, so treat them as +/- 1 degree.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import baseline                                              # noqa: E402
from src.config import AZ, EL, MIC_SUBSETS, N_SUBBANDS, RUNS          # noqa: E402
from src.data import SplitData, to_maps                               # noqa: E402
from src.geometry import angular_distance                             # noqa: E402
from src.metrics import angular_errors                                # noqa: E402
from src.models import UNet, decode_mask                              # noqa: E402
from src.train import beamformer_factory                              # noqa: E402

PAPER_BINS = (0.0, 50.0, 100.0, 200.0)
PAPER = {                       # Fig. 5, mean angular error in degrees
    ("beamforming", "test_2"): (38, 75, 33),
    ("unet", "test_2"): (10, 13, 14),
    ("beamforming", "test_1"): (8, 29, 92),
    ("unet", "test_1"): (12, 24, 54),
}
PAPER_FP = {"beamforming": 67.0, "unet": 14.9}      # section 4.1, five-minute no-drone audio
STABILITY_WINDOW = 3
STABILITY_DEG = 10.0


def binned_mean(err: np.ndarray, range_m: np.ndarray, bins=PAPER_BINS) -> list:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (range_m >= lo) & (range_m < hi) & np.isfinite(err)
        out.append((float(err[m].mean()) if m.any() else float("nan"), int(m.sum())))
    return out


def stability_flags(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Their detection rule: the last 3 direction estimates agree to within 10 degrees.

    Root-mean-square angular distance to the mean of the window, which is the spherical
    equivalent of a standard deviation. Frames before the window fills are excluded.
    """
    v = np.stack([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                  np.cos(np.radians(el)) * np.sin(np.radians(az)),
                  np.sin(np.radians(el))], 1)
    stable = np.zeros(len(az), bool)
    valid = np.zeros(len(az), bool)
    for t in range(STABILITY_WINDOW - 1, len(az)):
        w = v[t - STABILITY_WINDOW + 1:t + 1]
        mean = w.mean(0)
        n = np.linalg.norm(mean)
        if n < 1e-9:
            valid[t] = True
            continue
        mean = mean / n
        m_az = np.degrees(np.arctan2(mean[1], mean[0]))
        m_el = np.degrees(np.arcsin(np.clip(mean[2], -1, 1)))
        d = angular_distance(np.degrees(np.arctan2(w[:, 1], w[:, 0])),
                             np.degrees(np.arcsin(np.clip(w[:, 2], -1, 1))),
                             np.full(len(w), m_az), np.full(len(w), m_el))
        stable[t] = float(np.sqrt((d ** 2).mean())) <= STABILITY_DEG
        valid[t] = True
    return stable & valid


@torch.no_grad()
def cnn_predictions(run_id: str, split: str = "test_2", mode: str = "centroid") -> dict:
    """Per-frame predicted direction from a trained run, decoded their way or ours."""
    import json

    cfg = json.loads((RUNS / run_id / "config.json").read_text())
    model = UNet(cfg["K"] * cfg["subbands"], base=cfg["base"], depth=cfg["depth"])
    model.load_state_dict(torch.load(RUNS / run_id / "best.pt", map_location="cpu"))
    data = SplitData(split, cfg["channels"], cfg["K"], verbose=False)
    dev = torch.device(cfg["device"])
    model = model.to(dev).eval()
    bfs = beamformer_factory(data, cfg["subbands"], dev)
    cols = {k: [] for k in ("pred_az", "pred_el", "has_drone", "az", "el", "range_m")}
    for s, X, _, present, frames in data.batches(64, shuffle=False):
        x = to_maps(bfs(s), X.to(dev), cfg["input_mean"], cfg["input_std"])
        az, el = decode_mask(model(x)[0], mode=mode)
        cols["pred_az"].append(az)
        cols["pred_el"].append(el)
        cols["has_drone"].append(present.numpy())
        for key in ("az", "el", "range_m"):
            cols[key].append(data.meta[s][key][frames])
    return {k: np.concatenate(v) for k, v in cols.items()}


def report(name: str, paper_key: str, split: str, pred: dict) -> None:
    d = pred["has_drone"].astype(bool)
    if "pred_az" in pred:
        err = angular_distance(pred["pred_az"][d], pred["pred_el"][d],
                               pred["az"][d], pred["el"][d])
    else:
        err = angular_errors(pred["el_idx"][d], pred["az_idx"][d], pred["az"][d], pred["el"][d])
    ours = binned_mean(err, np.asarray(pred["range_m"])[d])
    theirs = PAPER[(paper_key, split)]

    print(f"\n{name} on {split}")
    print(f"  {'range':<12}{'paper':>9}{'ours':>9}{'diff':>9}{'frames':>9}")
    for (lo, hi), (mine, n), t in zip(zip(PAPER_BINS[:-1], PAPER_BINS[1:]), ours, theirs):
        diff = f"{mine - t:+.0f}" if np.isfinite(mine) else "—"
        val = f"{mine:.0f}d" if np.isfinite(mine) else "—"
        print(f"  {f'{lo:.0f}-{hi:.0f} m':<12}{t:>8}d{val:>9}{diff:>9}{n:>9}")

    # Their false-positive rule, on the no-drone frames of the same split.
    if (~d).any():
        p_az = pred["pred_az"] if "pred_az" in pred else np.asarray(AZ)[pred["az_idx"]]
        p_el = pred["pred_el"] if "pred_el" in pred else np.asarray(EL)[pred["el_idx"]]
        stable = stability_flags(p_az, p_el)
        fp = 100.0 * stable[~d].mean()
        print(f"  {'stable-direction false positives':<34}{PAPER_FP[paper_key]:>6.1f}% paper"
              f"   {fp:>5.1f}% ours   ({int((~d).sum())} no-drone frames)")


def main(split: str = "test_2") -> int:
    print(__doc__.split("\n\n")[0])
    print(f"\nAll numbers are mean angular error in degrees, drone frames only, "
          f"all 24 microphones (their array).")

    r = baseline.run_split(split, "all24", verbose=False)
    report("DAS beamforming, brightest pixel", "beamforming", split,
           {**r, "has_drone": r["has_drone"]})

    # Both decoders, because this is where we suspected the reimplementation gap lives: they
    # centroid an eroded mask, we were taking the brightest pixel.
    for mode in ("argmax", "centroid"):
        for seed in (0, 1, 2):
            run_id = f"all24_K1_w32_b16_wce_s{seed}"
            if (RUNS / run_id / "best.pt").exists():
                report(f"our U-Net, {mode} decoding (seed {seed})", "unet", split,
                       cnn_predictions(run_id, split, mode=mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "test_2"))
