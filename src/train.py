"""Train one configuration and write its metrics.

Protocol, identical for every condition because that identity is what makes rows
comparable: model selection on test 2, test 1 sealed, early stopping on test-2 AUPRC with
patience 10 and a hard cap, best checkpoint restored rather than last.

The review (docs/review_plan.md, Phase 3) found that protocol selects on the split it reports
and never varies the optimizer. Every change it proposes is a flag in PROTOCOL, and every flag
defaults to the behaviour above, so each of the 81 existing runs is reproduced from HEAD with its
original run id. A run with any non-default flag gets them spelled out in its run id and in
`config.json["protocol"]`:

- `val_split`: `test2` selects on test 2 (the original); `blocks` selects on interleaved 30 s
  blocks of the training scenes and scores test 2 once, as a true test; `lofo` holds out one
  site-1 scene (`fold`, from `config.SCENES`), selects on blocks of the other two and scores the
  held-out scene.
- `schedule` (`plateau` halves the rate after 3 flat epochs of the monitor, `cosine` anneals over
  `max_epochs`), `clip` (gradient-norm cap), `weight_decay` (AdamW when non-zero),
  `encoder_dropout` (channel dropout after each encoder block).
- `input_norm`: `global` training statistics, or `frame` to standardize every map on its own.
- `label_lag`: train on labels moved to sound-emission time; scoring keeps the dataset's labels.
- `far_weight`: per-example loss weight rising linearly from 1 at 60 m to 1 + far_weight at
  120 m and beyond, for the frames the far-field failure lives in.
- `presence_weight`: weight of the presence term against the mask term.
- `augment` (`src/augment.py`), `norm_layer` (`group` for GroupNorm), `target` (`soft` for the
  Gaussian target of `labels.target_mask`) and `nearfield_m` (steering focused at that range)
  are the Phase 3 ablations. `--loss dice` is the fourth, and is a loss, not a protocol flag.
"""
import argparse
import collections
import json
import subprocess
import time

import numpy as np
import torch

from . import augment, beamform, metrics
from . import polar as polar_grid
from .config import AZ, EL, MIC_SUBSETS, MODEL_BAND, N_SUBBANDS, RUNS, SCENES, scene_segments
from .data import SplitData, input_stats, to_maps
from .models import LOSSES, NORM_LAYERS, Loss, UNet, decode_mask

EARLY = {"patience": 10, "min_delta": 0.001, "min_epochs": 15, "max_epochs": 60}

# The plan said monitor test-2 AUPRC. Changed after the first run, on two grounds: the split
# is 81% positive, so AUPRC starts near 0.81 and barely moves, and the headline claim is
# about direction rather than detection. Hit rate at the dataset's own 10 degree tolerance is
# bounded, stable, and the number the claim is actually made in. Everything else is still
# logged every epoch, and the monitor is identical for every condition, which is what
# comparability requires.
MONITOR = "hit_10deg"

PROTOCOL = {"val_split": "test2", "fold": None, "schedule": "none", "clip": 0.0,
            "weight_decay": 0.0, "encoder_dropout": 0.0, "input_norm": "global",
            "label_lag": False, "far_weight": 0.0, "presence_weight": 1.0,
            "augment": False, "norm_layer": "batch", "target": "hard", "nearfield_m": 0.0}
VAL_SPLITS = ("test2", "blocks", "lofo")
TARGETS = ("hard", "soft")
SCHEDULES = ("none", "plateau", "cosine")
PLATEAU = {"factor": 0.5, "patience": 3}
FAR_RAMP_M = (60.0, 120.0)


def protocol_changes(protocol: dict) -> dict:
    """The flags that differ from the original protocol, in PROTOCOL order."""
    unknown = set(protocol) - set(PROTOCOL)
    if unknown:
        raise ValueError(f"unknown protocol flags {sorted(unknown)}")
    p = PROTOCOL | protocol
    if p["val_split"] not in VAL_SPLITS:
        raise ValueError(f"val_split must be one of {VAL_SPLITS}")
    if p["schedule"] not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}")
    if p["input_norm"] not in ("global", "frame"):
        raise ValueError("input_norm must be 'global' or 'frame'")
    if (p["val_split"] == "lofo") != (p["fold"] is not None):
        raise ValueError("fold is required with val_split='lofo' and meaningless without it")
    if p["fold"] is not None and p["fold"] not in SCENES:
        raise ValueError(f"fold must be one of {list(SCENES)}")
    if p["norm_layer"] not in NORM_LAYERS:
        raise ValueError(f"norm_layer must be one of {NORM_LAYERS}")
    if p["target"] not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    return {k: p[k] for k in PROTOCOL if p[k] != PROTOCOL[k]}


def protocol_suffix(changes: dict) -> str:
    """Run-id fragment for the non-default flags; empty for the original protocol."""
    parts = []
    for k, v in changes.items():
        if k == "val_split":
            parts.append(f"v{v}")
        elif k == "fold":
            parts[-1] += v.split("_")[0]
        elif k == "schedule":
            parts.append(v)
        elif k == "input_norm":
            parts.append(f"{v}norm")
        elif k == "label_lag":
            parts.append("lag")
        elif k == "augment":
            parts.append("aug")
        elif k == "norm_layer":
            parts.append("gn")
        elif k == "target":
            parts.append(v)
        else:
            short = {"clip": "clip", "weight_decay": "wd", "encoder_dropout": "edo",
                     "far_weight": "far", "presence_weight": "pw", "nearfield_m": "nf"}[k]
            parts.append(f"{short}{v:g}")
    return "_" + "-".join(parts) if parts else ""


def run_name(subset: str, K: int, base: int, subbands: int, loss: str, polar: bool, disc: str,
             seed: int, tag: str = "", protocol: dict | None = None) -> str:
    # `_polarn` rather than `_polar_native`, and only when it differs: the resampled runs are
    # already on disk under `_polar` and must keep their ids so nothing re-trains.
    suffix = "" if not polar else ("_polarn" if disc == "native" else "_polar")
    return (f"{subset}_K{K}_w{base}_b{subbands}_{loss}" + suffix
            + protocol_suffix(protocol_changes(protocol or {}))
            + f"_s{seed}" + (f"_{tag}" if tag else ""))


def far_weights(range_m: np.ndarray, far_weight: float) -> np.ndarray:
    """Per-example loss weight: 1 inside FAR_RAMP_M[0], rising linearly to 1 + far_weight at
    FAR_RAMP_M[1] and flat beyond; 1 for no-drone frames, whose range is NaN."""
    lo, hi = FAR_RAMP_M
    ramp = np.clip((np.nan_to_num(range_m, nan=0.0) - lo) / (hi - lo), 0.0, 1.0)
    return (1.0 + far_weight * ramp).astype(np.float32)


def protocol_data(channels, K: int, polar: bool, p: dict, verbose: bool = True):
    """(fit, monitor, tests) for a protocol. `tests` maps a result key to a SplitData that is
    scored once at the end and never used for selection, except under the original protocol,
    where test 2 is both, as it always was."""
    from .detector import split_for_early_stopping

    fit_kw = {"label_lag": p["label_lag"], "soft_target": p["target"] == "soft"}
    if p["val_split"] == "test2":
        fit = SplitData("train", channels, K, verbose=verbose, polar=polar, **fit_kw)
        monitor = SplitData("test_2", channels, K, verbose=verbose, polar=polar)
        return fit, monitor, {"test_2": monitor}
    if p["val_split"] == "blocks":
        full = SplitData("train", channels, K, verbose=verbose, polar=polar, **fit_kw)
        tests = {"test_2": SplitData("test_2", channels, K, verbose=verbose, polar=polar)}
    else:
        rest = [n for n in SCENES if n != p["fold"]]
        full = SplitData(f"scenes {'+'.join(rest)}", channels, K, verbose=verbose,
                         segs=scene_segments(rest), polar=polar, **fit_kw)
        held = f"scene_{p['fold']}"
        tests = {held: SplitData(held, channels, K, verbose=verbose,
                                 segs=scene_segments([p["fold"]]), polar=polar)}
    # The monitor is a view of the training scenes, so under label_lag it is scored against
    # lagged labels too, and under soft targets its val_loss is on soft masks; only `tests`
    # are guaranteed the dataset's own.
    fit, monitor = split_for_early_stopping(full)
    return fit, monitor, tests


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def beamformer_factory(data: SplitData, subbands: int, device, dirs=None, grid_shape=None,
                       focus_m: float = 0.0):
    """One steering tensor per session calibration, memoized.

    `dirs` and `grid_shape` steer at a direction set other than the rectangular grid, which is
    what the natively steered polar disc uses instead of resampling a rectangular map.
    `focus_m` is near-field steering (`beamform.Beamformer`).
    """
    extra = {} if dirs is None else {"dirs": dirs, "grid_shape": grid_shape}
    if focus_m > 0:
        extra["focus_m"] = focus_m

    def get(s: int):
        seg = data.segs[s]
        return beamform.get(seg.align, data.channels, data.freqs, key=seg.session,
                            n_subbands=subbands, band=MODEL_BAND, device=device, **extra)
    return get


# Test-time interventions on the frame stack, all applied to the same trained weights. Each
# removes one thing and leaves the rest, so the differences separate mechanisms that "K=5 beats
# K=1" alone cannot:
#   freeze   the current frame repeated K times -> the stack carries no history at all, so any
#            remaining gain over K=1 is extra input channels and parameters, not time
#   shuffle  the same K frames in random order -> removes order, keeps history
#   reverse  time flipped -> removes order in the one way that is still a smooth trajectory
# Notes section 13 predicts persistence dominates beyond ~70 m, where the true peak moves less
# than one grid cell per stack. Persistence survives shuffling, so a small shuffle effect next
# to a large freeze effect is the expected result, not a failure.
ORDERS = ("causal", "freeze", "shuffle", "reverse")


def reorder(X: torch.Tensor, order: str, rng) -> torch.Tensor:
    """(B, K, M, F) -> the same frames under a test-time intervention on the K axis.

    `freeze{j}` generalizes `freeze` to a *history depth*: the last j frames stay real and the
    earlier slots are filled with a copy of the oldest kept frame, so j=1 is exactly `freeze`
    and j=K is exactly `causal`. Sweeping j turns one trained model into a curve of accuracy
    against how far back it is allowed to see, which is the only way to ask whether the
    benefit of history saturates without training a second model at a different K — where
    more input channels and more parameters would confound the answer.
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
             order: str = "causal", seed: int = 0, proj=None, frame_norm: bool = False) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("score", "el_idx", "az_idx", "has_drone", "az", "el", "range_m",
                            "pred_az", "pred_el")}
    losses = []
    for s, X, mask, present, frames in data.batches(batch, shuffle=False):
        bf = bfs(s)
        x = to_maps(bf, reorder(X, order, rng).to(bf.device), *stats, projection=proj,
                    frame_norm=frame_norm)
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
          disc: str = "resample", verbose: bool = True, save_last: int = 0,
          **protocol) -> dict:
    """`save_last` keeps the weights of the final N epochs in `runs/<id>/last/` for
    `utils/average_checkpoints.py`. It changes nothing about training, so it is not part of
    the run id or the protocol."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if disc not in ("resample", "native"):
        raise ValueError(f"disc must be 'resample' or 'native', not {disc!r}")
    changes = protocol_changes(protocol)
    p = PROTOCOL | changes
    if p["input_norm"] == "frame" and polar:
        raise ValueError("input_norm='frame' is untested on the disc, whose dead cells would "
                         "enter every map's statistics")
    if p["augment"] and polar:
        raise ValueError("augment's azimuth roll needs the rectangular grid")
    run_id = run_name(subset, K, base, subbands, loss, polar, disc, seed, tag, changes)
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)

    channels = MIC_SUBSETS[subset]
    # train_data fits, val_data selects the epoch; under the original protocol val_data is
    # test 2 and is also what gets reported, which is the selection the review flagged.
    train_data, val_data, tests = protocol_data(channels, K, polar, p, verbose=verbose)
    frame_norm = p["input_norm"] == "frame"
    dev = beamform.pick_device(device)
    native = polar and disc == "native"
    proj = None
    if polar:
        proj = (polar_grid.NativeDisc(device=dev) if native
                else polar_grid.Projection(device=dev))
    # Native steering changes where the beamformer points; resampling does not, so only the
    # native path hands a direction set to the factory.
    steer = ({"dirs": proj.steering_dirs, "grid_shape": proj.grid_shape} if native else {})
    if p["nearfield_m"] > 0:
        steer["focus_m"] = p["nearfield_m"]
    bfs_train = beamformer_factory(train_data, subbands, dev, **steer)
    bfs_val = beamformer_factory(val_data, subbands, dev, **steer)
    bfs_tests = {k: bfs_val if d is val_data else beamformer_factory(d, subbands, dev, **steer)
                 for k, d in tests.items()}

    stats = ((0.0, 1.0) if frame_norm else
             input_stats(train_data, bfs_train, seed=seed, valid=proj.valid if native else None))
    pos_rate = train_data.mask_positive_rate()
    drone_rate = train_data.positive_rate
    # No circular padding on the disc: its horizontal axis is an image coordinate, not an angle.
    model = UNet(K * subbands, base=base, depth=depth, wrap=not polar,
                 encoder_dropout=p["encoder_dropout"], norm_layer=p["norm_layer"]).to(dev)
    aug_gen = torch.Generator().manual_seed(seed + 1) if p["augment"] else None
    loss_fn = Loss(loss, pos_weight=min(1.0 / max(pos_rate, 1e-6), 100.0),
                   presence_weight=p["presence_weight"],
                   presence_pos_weight=(1 - drone_rate) / max(drone_rate, 1e-6)).to(dev)
    opt = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p["weight_decay"])
           if p["weight_decay"] > 0 else torch.optim.Adam(model.parameters(), lr=lr))
    sched = None
    if p["schedule"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", **PLATEAU)
    elif p["schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

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
    if changes:
        # Only when something differs, so an original-protocol config is key-for-key the
        # config every earlier run wrote.
        cfg["protocol"] = p | {"tests": list(tests)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if verbose:
        h, w = train_data.masks[0].shape[1:]
        print(f"{run_id}: {model.n_params / 1e6:.2f} M params, input "
              f"{K * subbands}x{h}x{w}, mask positives {pos_rate * 100:.1f}%, "
              f"pos_weight {float(loss_fn.pw):.0f}, device {dev}")

    best, best_epoch, best_state, history = -np.inf, -1, None, []
    last = collections.deque(maxlen=save_last) if save_last > 0 else None
    for epoch in range(max_epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for s, X, mask, present, frames in train_data.batches(batch, rng=rng):
            bf = bfs_train(s)
            X = X.to(bf.device)
            if aug_gen is not None:
                X = augment.spectra(X, aug_gen)
            x = to_maps(bf, X, *stats, projection=proj, frame_norm=frame_norm)
            if aug_gen is not None:
                x, mask = augment.roll_azimuth(x, mask.to(bf.device), aug_gen)
            mask_logits, presence = model(x)
            weight = None
            if p["far_weight"] > 0:
                weight = torch.from_numpy(far_weights(train_data.meta[s]["range_m"][frames],
                                                      p["far_weight"])).to(bf.device)
            total, _, _ = loss_fn(mask_logits, presence, mask.to(bf.device),
                                  present.to(bf.device), weight=weight)
            opt.zero_grad(set_to_none=True)
            total.backward()
            if p["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), p["clip"])
            opt.step()
            tot += float(total.detach()) * len(X)
            n += len(X)
        m = evaluate(model, val_data, bfs_val, stats, loss_fn, proj=proj, frame_norm=frame_norm)
        m |= {"epoch": epoch, "train_loss": tot / max(n, 1),
              "seconds": time.time() - t0}
        if sched is not None:
            m["lr"] = opt.param_groups[0]["lr"]
            if p["schedule"] == "plateau":
                sched.step(m[MONITOR])
            else:
                sched.step()
        history.append(m)
        if last is not None:
            last.append((epoch, {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}))
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
    ev = {"stats": stats, "loss_fn": loss_fn, "proj": proj, "frame_norm": frame_norm}
    result = {"config": cfg, "best_epoch": best_epoch, "epochs_run": len(history)}
    if last is not None:
        (out / "last").mkdir(exist_ok=True)
        for e, state in last:
            torch.save(state, out / "last" / f"epoch{e:02d}.pt")
        result["saved_epochs"] = [e for e, _ in last]
    if changes:
        result["monitor"] = evaluate(model, val_data, bfs_val, **ev)
    for key, data in tests.items():
        result[key] = evaluate(model, data, bfs_tests[key], **ev)
    result["history"] = history
    primary = next(iter(tests))
    final = result[primary]
    if K > 1:
        result["order_control"] = {o: evaluate(model, tests[primary], bfs_tests[primary],
                                               order=o, seed=seed, **ev)
                                  for o in ORDERS if o != "causal"}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    if verbose:
        print(metrics.format_summary(f"{run_id} (best epoch {best_epoch}, {primary})", final))
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
    p.add_argument("--loss", default="wce", choices=list(LOSSES))
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--base", type=int, default=32,
                   help="base filter count; 32 is 2.3x faster than their 64 and overfits less")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=EARLY["max_epochs"])
    p.add_argument("--patience", type=int, default=EARLY["patience"])
    p.add_argument("--min-epochs", type=int, default=EARLY["min_epochs"])
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="")
    p.add_argument("--save-last", type=int, default=0,
                   help="keep the final N epochs' weights for utils/average_checkpoints.py")
    p.add_argument("--polar", action="store_true",
                   help="project the map onto the zenith disc; see src/polar.py")
    p.add_argument("--disc", default="resample", choices=["resample", "native"],
                   help="with --polar: resample the rectangular map, or steer at the disc's "
                        "own cells and skip the interpolation")
    g = p.add_argument_group("protocol (defaults reproduce every earlier run; see module doc)")
    g.add_argument("--val-split", default=PROTOCOL["val_split"], choices=VAL_SPLITS)
    g.add_argument("--fold", default=PROTOCOL["fold"], choices=list(SCENES),
                   help="held-out scene, with --val-split lofo")
    g.add_argument("--schedule", default=PROTOCOL["schedule"], choices=SCHEDULES)
    g.add_argument("--clip", type=float, default=PROTOCOL["clip"],
                   help="gradient-norm cap; 0 is off")
    g.add_argument("--weight-decay", type=float, default=PROTOCOL["weight_decay"],
                   help="AdamW weight decay; 0 keeps plain Adam")
    g.add_argument("--encoder-dropout", type=float, default=PROTOCOL["encoder_dropout"])
    g.add_argument("--input-norm", default=PROTOCOL["input_norm"], choices=["global", "frame"])
    g.add_argument("--label-lag", action="store_true",
                   help="train on labels at sound-emission time")
    g.add_argument("--far-weight", type=float, default=PROTOCOL["far_weight"])
    g.add_argument("--presence-weight", type=float, default=PROTOCOL["presence_weight"])
    g.add_argument("--augment", action="store_true",
                   help="gain jitter, channel dropout, additive noise, azimuth roll (src/augment.py)")
    g.add_argument("--norm-layer", default=PROTOCOL["norm_layer"], choices=list(NORM_LAYERS))
    g.add_argument("--target", default=PROTOCOL["target"], choices=list(TARGETS))
    g.add_argument("--nearfield-m", type=float, default=PROTOCOL["nearfield_m"],
                   help="steer at this range with spherical wavefronts; 0 is plane waves")
    a = p.parse_args()
    train(subset=a.subset, K=a.K, subbands=a.subbands, seed=a.seed, lr=a.lr, loss=a.loss,
          batch=a.batch, base=a.base, depth=a.depth, max_epochs=a.max_epochs,
          patience=a.patience, min_epochs=a.min_epochs, device=a.device, tag=a.tag,
          polar=a.polar, disc=a.disc, save_last=a.save_last, val_split=a.val_split, fold=a.fold,
          schedule=a.schedule, clip=a.clip, weight_decay=a.weight_decay,
          encoder_dropout=a.encoder_dropout, input_norm=a.input_norm, label_lag=a.label_lag,
          far_weight=a.far_weight, presence_weight=a.presence_weight, augment=a.augment,
          norm_layer=a.norm_layer, target=a.target, nearfield_m=a.nearfield_m)
