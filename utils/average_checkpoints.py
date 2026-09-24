"""Checkpoint averaging: the mean of a run's final epochs' weights, scored like `best.pt`.

Early stopping keeps the single epoch with the best monitor, which on a monitor as noisy as a
hit rate over a few thousand frames is partly the luckiest epoch rather than the best model.
Averaging the weights of several consecutive epochs (Izmailov et al.'s stochastic weight
averaging, without the cyclic schedule) lands nearer the centre of the basin those epochs are
wandering around, and is usually flatter, which is the property that transfers across scenes.

Needs a run trained with `python -m src.train --save-last N`, which writes the final N epochs to
`runs/<id>/last/`. Floating-point tensors are averaged; BatchNorm's batch counters are taken
from the last epoch. The averaged weights' BatchNorm running statistics belong to none of the
members, so they are recomputed with one pass over the run's own training data (the SWA
`update_bn` step) before anything is scored; `--no-bn-update` skips it, for the gate only.

Everything the run reported is re-scored the same way: its monitor split and every test split
of its protocol. Written to `runs/<id>/averaged.pt` (ignored by git) and
`results/checkpoint_averaging.json`.

Gate (`--gate`): averaging `best.pt` alone with no BatchNorm pass reproduces every test in the
run's own `metrics.json` to 1e-6. `--self-test` checks the arithmetic and the BatchNorm pass on
synthetic tensors, without data.

Run: .venv/bin/python utils/average_checkpoints.py <run_id> ... [--epochs 40 41 42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import beamform                                      # noqa: E402
from src import polar as polar_grid                           # noqa: E402
from src.config import RESULTS, RUNS                          # noqa: E402
from src.data import to_maps                                  # noqa: E402
from src.models import Loss, UNet                             # noqa: E402
from src.train import (PROTOCOL, beamformer_factory, evaluate,  # noqa: E402
                       protocol_data)

REPORTED = ("auroc", "specificity_at_95_recall", "hit_10deg", "median_angular_deg")
TOL = 1e-6


def average(states: list[dict]) -> dict:
    """Uniform mean of state dicts. Integer buffers (BatchNorm's `num_batches_tracked`) are
    counters, not parameters, and come from the last state."""
    out = {}
    for k, v in states[-1].items():
        if v.is_floating_point():
            out[k] = torch.stack([s[k].double() for s in states]).mean(0).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


@torch.no_grad()
def update_bn(model: nn.Module, inputs) -> None:
    """Recompute every BatchNorm's running mean and variance as the plain average over
    `inputs`, an iterable of model input batches."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    for x in inputs:
        model(x)
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.train(was_training)


class Run:
    """A finished run rebuilt from its config: model, data, beamformers, loss."""

    def __init__(self, run_id: str, device: str | None = None):
        self.id, self.dir = run_id, RUNS / run_id
        cfg = json.loads((self.dir / "config.json").read_text())
        self.cfg = cfg
        stored = dict(cfg.get("protocol", {}))
        stored.pop("tests", None)
        self.p = PROTOCOL | stored
        self.dev = beamform.pick_device(device or cfg["device"])
        is_polar = bool(cfg.get("polar", False))
        native = is_polar and cfg.get("polar_mode", "resample") == "native"
        self.proj = None
        if is_polar:
            self.proj = (polar_grid.NativeDisc(device=self.dev) if native
                         else polar_grid.Projection(device=self.dev))
        steer = ({"dirs": self.proj.steering_dirs, "grid_shape": self.proj.grid_shape}
                 if native else {})
        steer["focus_m"] = self.p["nearfield_m"]
        self.fit, self.monitor, self.tests = protocol_data(cfg["channels"], cfg["K"], is_polar,
                                                           self.p, verbose=False)
        self.bfs_fit = beamformer_factory(self.fit, cfg["subbands"], self.dev, **steer)
        self.bfs_monitor = beamformer_factory(self.monitor, cfg["subbands"], self.dev, **steer)
        self.bfs_tests = {k: self.bfs_monitor if d is self.monitor
                          else beamformer_factory(d, cfg["subbands"], self.dev, **steer)
                          for k, d in self.tests.items()}
        self.stats = (cfg["input_mean"], cfg["input_std"])
        self.frame_norm = self.p["input_norm"] == "frame"
        self.loss_fn = Loss(cfg.get("loss", "wce"), pos_weight=cfg["pos_weight"],
                            presence_weight=self.p["presence_weight"],
                            presence_pos_weight=cfg["presence_pos_weight"]).to(self.dev)

    def model(self, state: dict) -> UNet:
        c = self.cfg
        m = UNet(c["K"] * c["subbands"], base=c.get("base", 32), depth=c.get("depth", 3),
                 wrap=not c.get("polar", False),
                 encoder_dropout=self.p["encoder_dropout"],
                 norm_layer=self.p["norm_layer"]).to(self.dev)
        m.load_state_dict(state)
        return m

    def train_inputs(self, batch: int = 64):
        for s, X, _, _, _ in self.fit.batches(batch, shuffle=False):
            bf = self.bfs_fit(s)
            yield to_maps(bf, X.to(bf.device), *self.stats, projection=self.proj,
                          frame_norm=self.frame_norm)

    def score(self, model: UNet) -> dict:
        ev = {"stats": self.stats, "loss_fn": self.loss_fn, "proj": self.proj,
              "frame_norm": self.frame_norm}
        out = {"monitor": evaluate(model, self.monitor, self.bfs_monitor, **ev)}
        for k, d in self.tests.items():
            out[k] = evaluate(model, d, self.bfs_tests[k], **ev)
        return out


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def gate(run: Run) -> list[str]:
    ref = json.loads((run.dir / "metrics.json").read_text())
    got = run.score(run.model(average([load(run.dir / "best.pt")])))
    bad = []
    for split in run.tests:
        for k in REPORTED:
            a, b = got[split][k], ref[split][k]
            if abs(a - b) > TOL * max(1.0, abs(b)):
                bad.append(f"{run.id} {split} {k}: {a:.6f} vs metrics.json {b:.6f}")
    return bad


def self_test() -> int:
    torch.manual_seed(0)
    bad = []
    a = UNet(16, base=8, depth=2).state_dict()
    b = {k: (v + 1.0 if v.is_floating_point() else v + 5) for k, v in a.items()}
    one = average([a])
    if any(not torch.equal(one[k], a[k]) for k in a):
        bad.append("average of one state is not that state")
    two = average([a, b])
    for k, v in a.items():
        want = v + 0.5 if v.is_floating_point() else b[k]
        if not torch.allclose(two[k].double(), want.double(), atol=1e-6):
            bad.append(f"average of two states wrong at {k}")
            break

    bn = nn.Sequential(nn.BatchNorm2d(3)).eval()
    xs = [torch.randn(8, 3, 4, 5) * 2 + 1 for _ in range(4)]
    update_bn(bn, xs)
    full = torch.cat(xs)
    mean = full.mean((0, 2, 3))
    # Running variance is the unbiased batch variance, averaged over batches.
    var = torch.stack([x.var((0, 2, 3), unbiased=True) for x in xs]).mean(0)
    if not (torch.allclose(bn[0].running_mean, mean, atol=1e-5)
            and torch.allclose(bn[0].running_var, var, atol=1e-5)):
        bad.append("update_bn does not give the plain average of batch statistics")
    if bn[0].momentum != 0.1 or bn.training:
        bad.append("update_bn did not restore momentum and mode")

    print("GATE " + ("FAIL" if bad else "PASS") + " (self-test on synthetic tensors)")
    for x in bad:
        print(f"  {x}")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="*")
    p.add_argument("--epochs", nargs="*", type=int, default=None,
                   help="which saved epochs to average; default every file in last/")
    p.add_argument("--no-bn-update", action="store_true")
    p.add_argument("--gate", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not a.runs:
        p.error("give at least one run id")

    out_path = RESULTS / "checkpoint_averaging.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    failures = []
    for run_id in a.runs:
        run = Run(run_id, a.device)
        if a.gate:
            failures += gate(run)
            continue
        files = sorted((run.dir / "last").glob("epoch*.pt"))
        if a.epochs is not None:
            files = [f for f in files if int(f.stem[len("epoch"):]) in a.epochs]
        if not files:
            raise SystemExit(f"{run_id}: no saved epochs; train with --save-last N")
        epochs = [int(f.stem[len("epoch"):]) for f in files]
        model = run.model(average([load(f) for f in files]))
        if not a.no_bn_update:
            update_bn(model, run.train_inputs())
        torch.save(model.state_dict(), run.dir / "averaged.pt")
        got = run.score(model)
        best = run.score(run.model(load(run.dir / "best.pt")))
        ref = json.loads((run.dir / "metrics.json").read_text())
        report[run_id] = {"epochs": epochs, "best_epoch": ref["best_epoch"],
                          "bn_update": not a.no_bn_update,
                          "averaged": {k: {m: v[m] for m in REPORTED} for k, v in got.items()},
                          "best": {k: {m: v[m] for m in REPORTED} for k, v in best.items()}}
        print(f"\n{run_id}: epochs {epochs} averaged, best epoch {ref['best_epoch']}")
        for split in got:
            print(f"  {split:14s} hit@10  best {best[split]['hit_10deg']:.4f}  "
                  f"averaged {got[split]['hit_10deg']:.4f}")

    if a.gate:
        print("GATE " + ("FAIL" if failures else "PASS"))
        for f in failures:
            print(f"  {f}")
        if not failures:
            print("  best.pt, averaged alone, reproduces every run's metrics.json")
        return 1 if failures else 0
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
