"""Train one configuration and write its metrics.

Protocol, identical for every condition because that identity is what makes rows
comparable: model selection on test 2, test 1 sealed, early stopping on test-2 AUPRC with
patience 10 and a hard cap, best checkpoint restored rather than last.
"""
import argparse
import json
import subprocess
import time

import numpy as np
import torch

from . import beamform, metrics
from . import polar as polar_grid
from .config import MIC_SUBSETS, MODEL_BAND, N_SUBBANDS, run_dir
from .data import SplitData, input_stats, to_maps
from .models import Loss, UNet, decode_mask

EARLY = {"patience": 10, "min_delta": 0.001, "min_epochs": 15, "max_epochs": 60}

# Hit rate at the dataset's 10 degree tolerance, not AUPRC: the split is 81% positive so
# AUPRC barely moves, and the claim is about direction. Same monitor for every condition.
# docs/design_notes.md section 29.
MONITOR = "hit_10deg"


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def beamformer_factory(data: SplitData, subbands: int, device, dirs=None, grid_shape=None):
    """One steering tensor per session calibration, memoized.

    `dirs` and `grid_shape` steer at a direction set other than the rectangular grid, which is
    what the natively steered polar disc uses instead of resampling a rectangular map.
    """
    extra = {} if dirs is None else {"dirs": dirs, "grid_shape": grid_shape}

    def get(s: int):
        seg = data.segs[s]
        return beamform.get(seg.align, data.channels, data.freqs, key=seg.session,
                            n_subbands=subbands, band=MODEL_BAND, device=device, **extra)
    return get


# Test-time interventions on the frame stack, same trained weights. `freeze` repeats the
# current frame K times (no history), `shuffle` keeps history but drops order, `reverse`
# flips time. docs/design_notes.md section 13.
ORDERS = ("causal", "freeze", "shuffle", "reverse")


def history_order(j: int, K: int) -> str:
    """The `reorder` name for a history depth of j frames out of K: `freeze{j}`, or `causal`
    when nothing is frozen."""
    return "causal" if j >= K else f"freeze{j}"


def reorder(X: torch.Tensor, order: str, rng) -> torch.Tensor:
    """(B, K, M, F) -> the same frames under a test-time intervention on the K axis.

    `freeze{j}` keeps the last j frames real and fills the earlier slots with the oldest kept
    frame, so j=1 is `freeze` and j=K is `causal`. Section 26 of the notes.
    """
    K = X.shape[1]
    if order == "causal" or K == 1:
        return X
    if order.startswith("freeze"):
        j = min(max(int(order[len("freeze"):] or 1), 1), K)
        if j == K:
            return X
        keep = X[:, K - j:]
        return torch.cat([keep[:, :1].expand(-1, K - j, -1, -1), keep], 1)
    perm = np.arange(K - 1, -1, -1) if order == "reverse" else rng.permutation(K)
    return X[:, torch.from_numpy(np.ascontiguousarray(perm))]


@torch.no_grad()
def evaluate(model, data: SplitData, bfs, stats, loss_fn, batch: int = 64,
             order: str = "causal", seed: int = 0, proj=None) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("score", "el_idx", "az_idx", "has_drone", "az", "el", "range_m",
                            "pred_az", "pred_el")}
    losses = []
    for s, X, mask, present, frames in data.batches(batch, shuffle=False):
        bf = bfs(s)
        x = to_maps(bf, reorder(X, order, rng).to(bf.device), *stats, projection=proj)
        mask, present = mask.to(bf.device), present.to(bf.device)
        mask_logits, presence = model(x)
        total, _, _ = loss_fn(mask_logits, presence, mask, present)
        losses.append(float(total.detach()))
        k = mask_logits.flatten(1).argmax(1)
        # Logits, not sigmoid. Every metric here only needs the ranking, and float32 sigmoid
        # saturates to exactly 1.0 above a logit of about 17: the K=5 probe reported AUROC
        # 0.5000 purely because confident scores had collapsed into one tied value.
        cols["score"].append(presence.cpu().numpy())
        n_cols = mask_logits.shape[-1]
        cols["el_idx"].append((k // n_cols).cpu().numpy())
        cols["az_idx"].append((k % n_cols).cpu().numpy())
        cols["has_drone"].append(present.cpu().numpy())
        if proj is not None:
            # The disc's cells are not an elevation-by-azimuth rectangle, so the direction has
            # to come from the cell's own unit vector. Argmax, not centroid, to keep every
            # number in this table comparable with the rectangular runs.
            az_deg, el_deg = decode_mask(mask_logits, mode="argmax", dirs=proj.dirs,
                                         valid=proj.valid)
            cols["pred_az"].append(az_deg)
            cols["pred_el"].append(el_deg)
        for key in ("az", "el", "range_m"):
            cols[key].append(data.meta[s][key][frames])
    c = {k: np.concatenate(v) for k, v in cols.items() if v}
    m = metrics.summarize(c["score"], c["has_drone"], c["el_idx"], c["az_idx"],
                          c["az"], c["el"], c["range_m"],
                          pred_az=c.get("pred_az"), pred_el=c.get("pred_el"))
    m["val_loss"] = float(np.mean(losses))
    # Spread of the presence logits. Near zero means the head has collapsed to a constant and
    # its AUROC of 0.500 is a dead branch rather than an inseparable problem — worth seeing in
    # the epoch log instead of discovering it in a finished run.
    m["presence_logit_std"] = float(np.std(c["score"]))
    return m


def train(subset: str = "device3", K: int = 1, subbands: int = N_SUBBANDS,
          seed: int = 0, lr: float = 5e-3, loss: str = "wce", batch: int = 32,
          base: int = 32, depth: int = 3, max_epochs: int = EARLY["max_epochs"],
          patience: int = EARLY["patience"], min_epochs: int = EARLY["min_epochs"],
          device: str = "auto", tag: str = "", polar: bool = False,
          disc: str = "resample", verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if disc not in ("resample", "native"):
        raise ValueError(f"disc must be 'resample' or 'native', not {disc!r}")
    # `_polarn` rather than `_polar_native`, and only when it differs: the resampled runs are
    # already on disk under `_polar` and must keep their ids so nothing re-trains.
    suffix = "" if not polar else ("_polarn" if disc == "native" else "_polar")
    run_id = (f"{subset}_K{K}_w{base}_b{subbands}_{loss}" + suffix
              + f"_s{seed}" + (f"_{tag}" if tag else ""))
    out = run_dir(run_id)
    out.mkdir(parents=True, exist_ok=True)

    channels = MIC_SUBSETS[subset]
    train_data = SplitData("train", channels, K, verbose=verbose, polar=polar)
    val_data = SplitData("test_2", channels, K, verbose=verbose, polar=polar)
    dev = beamform.pick_device(device)
    native = polar and disc == "native"
    proj = None
    if polar:
        proj = (polar_grid.NativeDisc(device=dev) if native
                else polar_grid.Projection(device=dev))
    # Native steering changes where the beamformer points; resampling does not, so only the
    # native path hands a direction set to the factory.
    steer = ({"dirs": proj.steering_dirs, "grid_shape": proj.grid_shape} if native else {})
    bfs_train = beamformer_factory(train_data, subbands, dev, **steer)
    bfs_val = beamformer_factory(val_data, subbands, dev, **steer)

    stats = input_stats(train_data, bfs_train, seed=seed,
                        valid=proj.valid if native else None)
    pos_rate = train_data.mask_positive_rate()
    drone_rate = train_data.positive_rate
    # No circular padding on the disc: its horizontal axis is an image coordinate, not an angle.
    model = UNet(K * subbands, base=base, depth=depth, wrap=not polar).to(dev)
    loss_fn = Loss(loss, pos_weight=min(1.0 / max(pos_rate, 1e-6), 100.0),
                   presence_pos_weight=(1 - drone_rate) / max(drone_rate, 1e-6)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    cfg = {"run_id": run_id, "git": git_sha(), "subset": subset, "channels": channels,
           "K": K, "subbands": subbands, "seed": seed, "lr": lr, "loss": loss,
           "polar": polar, "polar_mode": (disc if polar else None),
           "grid_shape": list(train_data.masks[0].shape[1:]),
           "batch": batch, "base": base, "depth": depth, "device": str(dev),
           "input_mean": stats[0], "input_std": stats[1],
           "mask_positive_rate": pos_rate, "pos_weight": float(loss_fn.pw),
           "train_drone_rate": drone_rate, "presence_pos_weight": float(loss_fn.ppw),
           "train_stacks": len(train_data), "val_stacks": len(val_data),
           "val_positive_rate": val_data.positive_rate, "monitor": MONITOR,
           "n_params": model.n_params, "early_stopping": EARLY | {"patience": patience}}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if verbose:
        h, w = train_data.masks[0].shape[1:]
        print(f"{run_id}: {model.n_params / 1e6:.2f} M params, input "
              f"{K * subbands}x{h}x{w}, mask positives {pos_rate * 100:.1f}%, "
              f"pos_weight {float(loss_fn.pw):.0f}, device {dev}")

    best, best_epoch, best_state, history = -np.inf, -1, None, []
    for epoch in range(max_epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for s, X, mask, present, _ in train_data.batches(batch, rng=rng):
            bf = bfs_train(s)
            x = to_maps(bf, X.to(bf.device), *stats, projection=proj)
            mask_logits, presence = model(x)
            total, _, _ = loss_fn(mask_logits, presence, mask.to(bf.device),
                                  present.to(bf.device))
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            tot += float(total.detach()) * len(X)
            n += len(X)
        m = evaluate(model, val_data, bfs_val, stats, loss_fn, proj=proj)
        m |= {"epoch": epoch, "train_loss": tot / max(n, 1),
              "seconds": time.time() - t0}
        history.append(m)
        if verbose:
            print(f"  epoch {epoch:>2}  {m['seconds']:>5.0f}s  train {m['train_loss']:.4f}  "
                  f"val {m['val_loss']:.4f}  AUROC {m['auroc']:.4f} "
                  f"(logit sd {m['presence_logit_std']:.2f})  "
                  f"median {m['median_angular_deg']:.1f} deg  "
                  f"<10d {m['hit_10deg'] * 100:.0f}%")
        if m[MONITOR] > best + EARLY["min_delta"]:
            best, best_epoch = m[MONITOR], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch + 1 >= min_epochs and epoch - best_epoch >= patience:
            if verbose:
                print(f"  early stop: no {MONITOR} gain since epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out / "best.pt")
    final = evaluate(model, val_data, bfs_val, stats, loss_fn, proj=proj)
    result = {"config": cfg, "best_epoch": best_epoch, "epochs_run": len(history),
              "test_2": final, "history": history}
    if K > 1:
        result["order_control"] = {o: evaluate(model, val_data, bfs_val, stats, loss_fn,
                                               order=o, seed=seed, proj=proj)
                                  for o in ORDERS if o != "causal"}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    if verbose:
        print(metrics.format_summary(f"{run_id} (best epoch {best_epoch})", final))
        for o, m in result.get("order_control", {}).items():
            print(metrics.format_summary(f"  test-time {o}", m))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train one configuration.")
    p.add_argument("--subset", default="device3", choices=list(MIC_SUBSETS))
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--subbands", type=int, default=N_SUBBANDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--loss", default="wce", choices=["wce", "tversky"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--base", type=int, default=32,
                   help="base filter count; 32 is 2.3x faster than their 64 and overfits less")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=EARLY["max_epochs"])
    p.add_argument("--patience", type=int, default=EARLY["patience"])
    p.add_argument("--min-epochs", type=int, default=EARLY["min_epochs"])
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="")
    p.add_argument("--polar", action="store_true",
                   help="project the map onto the zenith disc; see src/polar.py")
    p.add_argument("--disc", default="resample", choices=["resample", "native"],
                   help="with --polar: resample the rectangular map, or steer at the disc's "
                        "own cells and skip the interpolation")
    a = p.parse_args()
    train(subset=a.subset, K=a.K, subbands=a.subbands, seed=a.seed, lr=a.lr, loss=a.loss,
          batch=a.batch, base=a.base, depth=a.depth, max_epochs=a.max_epochs,
          patience=a.patience, min_epochs=a.min_epochs, device=a.device, tag=a.tag,
          polar=a.polar, disc=a.disc)
