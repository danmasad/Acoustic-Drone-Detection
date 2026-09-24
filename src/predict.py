"""Per-frame predictions from a finished run, cached to disk.

Everything in `metrics.json` is a summary: one number per run per metric. Several questions we
still owe the report cannot be answered from summaries, only from the frames themselves:

- **Detection against range.** A pooled AUROC hides that the drone is inaudible past 100 m.
- **Temporal integration.** Smoothing a score over a window needs the frames back in time
  order, which means carrying the segment and frame index alongside every prediction.
- **Frame-paired comparisons.** Asking whether K=5 beats K=1 *on the same frame* is far more
  sensitive than comparing two pooled means, and it needs both runs' per-frame output.

So this module runs a checkpoint once and writes every per-frame quantity to
`results/preds/`. The dumps are small (tens of thousands of frames, a dozen float32 columns)
and they are committed, which is what makes every analysis script a pure function of files in
the repository rather than of 4 GB of cache and a GPU.

`cnn_predictions` was lifted out of `utils/check_vs_paper.py`, which now imports it, so there
is one implementation of "run the model and decode a direction" rather than two that can
drift apart.
"""
import argparse
import json

import numpy as np
import torch

from . import baseline, metrics, polar
from .config import AZ, CACHE, EL, MIC_SUBSETS, RESULTS, RUNS, scene_segments
from .data import SplitData, to_maps
from .models import UNet, decode_mask

# Under results/, not cache/: these are committed outputs, and cache/ is the git-ignored
# directory of things a clone is expected to rebuild from audio it may not have.
PREDS = RESULTS / "preds"
# Full mask logits, one map per frame in dump order: 15291 x 23 x 91 float32 is 128 MB a run
# on test 2, far too large to commit. float32 rather than float16 because half precision
# moves near-tied argmax pixels, and the ensemble's gate is that one member reproduces its
# own dump exactly.
HEATMAPS = CACHE / "heatmaps"


def heatmap_path(name: str):
    return HEATMAPS / f"{name}.npy"

# Columns every dump carries, whatever produced it, so downstream analysis never branches on
# provenance. `seg` and `frame` are what make time order and per-scene grouping recoverable.
COMMON = ("score", "el_idx", "az_idx", "pred_az", "pred_el", "has_drone", "az", "el",
          "range_m", "seg", "frame")


def _save(name: str, cols: dict, seg_ids: list[str], meta: dict) -> dict:
    PREDS.mkdir(parents=True, exist_ok=True)
    out = {k: np.asarray(v) for k, v in cols.items()}
    # Compressed because these are committed and most columns are grid indices and repeated
    # labels: 1.17 MB becomes 0.34 MB, and np.load reads either form without being told which.
    np.savez_compressed(PREDS / f"{name}.npz", seg_ids=np.array(seg_ids),
                        meta=json.dumps(meta), **out)
    out["seg_ids"] = np.array(seg_ids)
    out["meta"] = meta
    return out


def load(name: str) -> dict:
    """One cached dump as a dict of arrays, with `meta` decoded back to a dict."""
    d = np.load(PREDS / f"{name}.npz", allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def split_data(split: str, channels, K: int, polar: bool = False) -> SplitData:
    """A dataset split, or `scene_<name>` for one site-1 scene of `config.SCENES`, which is
    what a leave-one-flight-out run reports on."""
    segs = scene_segments([split[len("scene_"):]]) if split.startswith("scene_") else None
    return SplitData(split, channels, K, verbose=False, polar=polar, segs=segs)


@torch.no_grad()
def cnn_predictions(run_id: str, split: str = "test_2", order: str = "causal",
                    seed: int = 0, batch: int = 64, device: str | None = None,
                    overwrite: bool = False, verbose: bool = True,
                    heatmaps: bool = False) -> dict:
    """Every per-frame output of one trained run, cached.

    `order` accepts anything `src.train.reorder` understands, including the `freeze{j}`
    family, so a history-depth sweep is a loop over this function with one set of weights.
    `heatmaps` also writes the full mask logits to `HEATMAPS`, which is what averaging seeds
    before decoding (`utils/ensemble.py`) needs; a cached dump without them is recomputed.
    """
    from . import beamform
    from .train import beamformer_factory, reorder

    name = f"{run_id}_{split}_{order}"
    if (PREDS / f"{name}.npz").exists() and not overwrite and not (
            heatmaps and not heatmap_path(name).exists()):
        if verbose:
            print(f"  cached: {name}")
        return load(name)

    cfg = json.loads((RUNS / run_id / "config.json").read_text())
    dev = beamform.pick_device(device or cfg["device"])
    is_polar = bool(cfg.get("polar", False))
    model = UNet(cfg["K"] * cfg["subbands"], base=cfg.get("base", 32),
                 depth=cfg.get("depth", 3), wrap=not is_polar).to(dev).eval()
    model.load_state_dict(torch.load(RUNS / run_id / "best.pt", map_location=dev))

    data = split_data(split, cfg["channels"], cfg["K"], polar=is_polar)
    frame_norm = cfg.get("protocol", {}).get("input_norm") == "frame"
    # Runs trained before the native disc existed have no polar_mode, and they were all
    # resampled, so that is the right default.
    native = is_polar and cfg.get("polar_mode", "resample") == "native"
    proj = None
    if is_polar:
        proj = polar.NativeDisc(device=dev) if native else polar.Projection(device=dev)
    steer = ({"dirs": proj.steering_dirs, "grid_shape": proj.grid_shape} if native else {})
    bfs = beamformer_factory(data, cfg["subbands"], dev, **steer)
    dirs = proj.dirs if is_polar else None
    valid = proj.valid if is_polar else None
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in COMMON}
    cols["argmax_az"], cols["argmax_el"] = [], []
    maps = []
    for s, X, _, present, frames in data.batches(batch, shuffle=False):
        bf = bfs(s)
        x = to_maps(bf, reorder(X, order, rng).to(dev), cfg["input_mean"], cfg["input_std"],
                    projection=proj, frame_norm=frame_norm)
        mask_logits, presence = model(x)
        k = mask_logits.flatten(1).argmax(1)
        # Both decoders, every time. They disagree by a lot — the centroid is the paper's and
        # is what the report quotes — and running both here costs one extra pooling pass.
        c_az, c_el = decode_mask(mask_logits, mode="centroid", dirs=dirs, valid=valid)
        a_az, a_el = decode_mask(mask_logits, mode="argmax", dirs=dirs, valid=valid)
        # Logits, never sigmoid: float32 sigmoid saturates to exactly 1.0 above a logit of
        # about 17, which once turned a working head into an AUROC of 0.5000 through ties.
        cols["score"].append(presence.cpu().numpy())
        n_cols = mask_logits.shape[-1]
        cols["el_idx"].append((k // n_cols).cpu().numpy())
        cols["az_idx"].append((k % n_cols).cpu().numpy())
        cols["pred_az"].append(c_az)
        cols["pred_el"].append(c_el)
        cols["argmax_az"].append(a_az)
        cols["argmax_el"].append(a_el)
        cols["has_drone"].append(present.numpy())
        for key in ("az", "el", "range_m"):
            cols[key].append(data.meta[s][key][frames])
        cols["seg"].append(np.full(len(frames), s))
        cols["frame"].append(np.asarray(frames))
        if heatmaps:
            maps.append(mask_logits.float().cpu().numpy())

    cols = {k: np.concatenate(v) for k, v in cols.items()}
    if heatmaps:
        HEATMAPS.mkdir(parents=True, exist_ok=True)
        np.save(heatmap_path(name), np.concatenate(maps))
    meta = {"kind": "cnn", "run_id": run_id, "split": split, "order": order,
            "K": cfg["K"], "subset": cfg["subset"], "seed": cfg["seed"],
            "subbands": cfg["subbands"], "polar": is_polar,
            "polar_mode": cfg.get("polar_mode") if is_polar else None,
            "loss": cfg.get("loss", "wce")}
    if verbose:
        print(f"  {name}: {len(cols['score'])} frames")
    return _save(name, cols, [seg.id for seg in data.segs], meta)


def baseline_predictions(split: str = "test_2", subset: str = "all24", device: str = "auto",
                         overwrite: bool = False, verbose: bool = True) -> dict:
    """The classical DAS baseline's per-frame output, in the same layout as a CNN dump.

    `results/baseline.json` only ever stored summaries, so the audibility question — how many
    labelled-positive frames carry no beamformed evidence at all — had no data to run on.
    """
    name = f"baseline_{subset}_{split}"
    if (PREDS / f"{name}.npz").exists() and not overwrite:
        if verbose:
            print(f"  cached: {name}")
        return load(name)

    r = baseline.run_split(split, subset, device=device, verbose=False)
    cols = {k: r[k] for k in ("el_idx", "az_idx", "has_drone", "az", "el", "range_m",
                              "seg", "frame", "peak_db", "contrast_db")}
    # The baseline has no presence head; its detection statistic is the beamformed peak. Named
    # `score` too so every analysis script can treat the two dumps identically.
    cols["score"] = r["peak_db"]
    cols["pred_az"] = np.asarray(AZ)[r["az_idx"]].astype(np.float32)
    cols["pred_el"] = np.asarray(EL)[r["el_idx"]].astype(np.float32)
    meta = {"kind": "baseline", "split": split, "subset": subset,
            "channels": MIC_SUBSETS[subset]}
    if verbose:
        print(f"  {name}: {len(cols['score'])} frames")
    return _save(name, cols, list(r["seg_ids"]), meta)


def verify(pred: dict, run_id: str, split: str = "test_2", block: str | None = None,
           tol: float = 1e-6, polar: bool = False) -> list[str]:
    """Recompute the run's headline metrics from the dump and diff them against metrics.json.

    The gate for A1: if a dump cannot reproduce the numbers the run already reported, every
    analysis built on it is worthless, and the failure should surface here rather than as a
    surprising result three stages later.
    """
    saved = json.loads((RUNS / run_id / "metrics.json").read_text())
    ref = saved[block] if block else saved[split]
    # A polar run's metrics came from the cell's own direction vector, so the check has to
    # score it the same way rather than through the grid-index lookup.
    extra = {"pred_az": pred["argmax_az"], "pred_el": pred["argmax_el"]} if polar else {}
    m = metrics.summarize(pred["score"], pred["has_drone"], pred["el_idx"], pred["az_idx"],
                          pred["az"], pred["el"], pred["range_m"], **extra)
    bad = []
    for key in ("auroc", "auprc", "hit_10deg", "median_angular_deg",
                "specificity_at_95_recall", "n_frames", "n_drone"):
        if key in ref and abs(m[key] - ref[key]) > tol * max(1.0, abs(ref[key])):
            bad.append(f"{key}: dump {m[key]:.6f} vs metrics.json {ref[key]:.6f}")
    return bad


def all_runs(pattern: str = "_w32_b16_wce_s") -> list[str]:
    """Finished runs with a checkpoint, excluding archived and probe directories."""
    return sorted(d.name for d in RUNS.iterdir()
                  if d.is_dir() and not d.name.startswith("_")
                  and pattern in d.name and (d / "best.pt").exists()
                  and (d / "config.json").exists())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dump per-frame predictions for every run.")
    p.add_argument("--split", default="test_2")
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--orders", nargs="*", default=["causal"])
    p.add_argument("--subsets", nargs="*", default=["all24", "device3", "best6"],
                   help="baseline subsets to dump alongside the networks")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--heatmaps", action="store_true",
                   help="also write each run's mask logits to cache/heatmaps/ (128 MB a run)")
    a = p.parse_args()

    print(f"baseline dumps on {a.split}")
    for subset in a.subsets:
        baseline_predictions(a.split, subset, overwrite=a.overwrite)

    runs = a.runs or all_runs()
    print(f"\n{len(runs)} runs on {a.split}, orders {a.orders}")
    failures = []
    for run_id in runs:
        for order in a.orders:
            pred = cnn_predictions(run_id, a.split, order=order, device=a.device,
                                   overwrite=a.overwrite, heatmaps=a.heatmaps)
            if order == "causal":
                bad = verify(pred, run_id, a.split, polar=pred["meta"].get("polar", False))
                if bad:
                    failures.append((run_id, bad))

    print("\nGATE " + ("FAIL" if failures else "PASS"))
    for run_id, bad in failures:
        print(f"  {run_id}: " + "; ".join(bad))
    if not failures:
        print(f"  every causal dump reproduces its own metrics.json to 1e-6")
