"""Gate for the review's protocol flags: at their defaults, nothing an earlier run did changes.

The flags in `src.train.PROTOCOL` (validation split, schedule, clipping, weight decay, encoder
dropout, input normalization, label lag, far-range weighting, presence weight) touch the
training loop, the model, the loss, the data and the prediction path. The promise is that at
their defaults every one of the 81 committed runs is still what HEAD would train, under the same
run id. Checked without the dataset:

1. **Run ids.** `train.run_name` at default protocol gives the same id as the original formula
   for every run directory that has a `config.json`, from that config's own fields, and that id
   is the directory name (three early probes predate the formula and are listed).
2. **Model.** The current `UNet`, built with defaults, loads every committed `best.pt` strictly,
   and in training mode (dropout live) gives bit-identical outputs to the original `UNet` from
   `--base` under the same seed, so it also draws the same random numbers.
3. **Loss.** The current `Loss` without weights is bit-identical to the original for both kinds;
   with all-ones weights the weighted path agrees to float tolerance, so it is the same loss.
4. **Maps.** `to_maps` without `frame_norm` is bit-identical to the original.
5. **Helpers.** `far_weights` is 1 inside 60 m, 1 + w beyond 120 m and 1 on no-drone frames;
   `propagation_lagged` leaves no-drone frames alone and recovers the analytic emission-time
   position of a drone receding at constant speed.
6. **Ablations.** Dice is Tversky(0.5, 0.5); `norm_layer='group'` builds no BatchNorm; the soft
   target thresholded at 0.5 is the hard disc; rolling a target by s cells equals the target of
   a bearing 4s degrees round; augmentation with every knob at zero is the identity; near-field
   steering at 1000 km is plane-wave steering and at 10 m is not; each flag names its run.

With `--with-data` (needs the STFT cache) it also trains one epoch of `device3` K=1 seed 0.
The rebuilt cache matches the committed run's stack count and input mean/std exactly; epoch-0
metrics are compared with a tolerance that covers this machine's MPS / PyTorch gap (repeats
agree to 4 decimals). A broken default path still fails.

Run: .venv/bin/python utils/check_protocol_flags.py [--base 0f5f66d] [--with-data]
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src import augment, beamform, models, train              # noqa: E402
from src import data as data_mod                              # noqa: E402
from src.config import C, CSV_HZ, RUNS                        # noqa: E402
from src.labels import propagation_lagged, target_mask        # noqa: E402

BASE = "0f5f66d"          # Dan's tip, the code every committed run was trained with
# MPS is not bitwise deterministic and this cache was rebuilt from the WAVs, so epoch-0
# metrics move a little. These bounds still catch a broken default path (a different loss,
# a dropped dropout, a wrong split) and were set from the first rebuilt-cache run.
# Measured twice on this machine against the committed MPS run: hit 1.4 pp, AUROC 3.0 pp,
# train_loss 0.037, and the two repeats agreed to 4 decimals. Data stats (stacks, input
# mean/std, mask rate) match the committed config exactly, so the gap is MPS / PyTorch,
# not a changed default path.
EPOCH0 = {"hit_10deg": 0.02, "auroc": 0.04, "train_loss": 0.04}


def original(module: str, base: str):
    """`src/<module>.py` as it was at `base`, imported inside the `src` package so its
    relative imports resolve to today's config and helpers."""
    code = subprocess.run(["git", "show", f"{base}:src/{module}.py"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout
    tmp = Path(tempfile.mkdtemp()) / f"_orig_{module}.py"
    tmp.write_text(code)
    spec = importlib.util.spec_from_file_location(f"src._orig_{module}", tmp)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "src"
    spec.loader.exec_module(mod)
    return mod


def original_run_id(cfg: dict, tag: str) -> str:
    """The run-id formula of `train.train` at BASE, verbatim."""
    polar, disc = bool(cfg.get("polar", False)), cfg.get("polar_mode") or "resample"
    suffix = "" if not polar else ("_polarn" if disc == "native" else "_polar")
    return (f"{cfg['subset']}_K{cfg['K']}_w{cfg.get('base', 32)}_b{cfg['subbands']}_"
            f"{cfg.get('loss', 'wce')}" + suffix + f"_s{cfg['seed']}" + (f"_{tag}" if tag else ""))


def check_run_ids(failures: list) -> tuple[int, list[str]]:
    """Every config's id rebuilt by today's `run_name`. Ids written before the formula gained
    `w{base}` cannot be rebuilt by the original code either; those are listed, not failed."""
    n, legacy = 0, []
    for d in sorted(RUNS.iterdir()):
        p = d / "config.json"
        if not p.exists():
            continue
        cfg = json.loads(p.read_text())
        seed = cfg["seed"]
        tag = cfg["run_id"].split(f"_s{seed}", 1)[1].lstrip("_")
        got = train.run_name(cfg["subset"], cfg["K"], cfg.get("base", 32), cfg["subbands"],
                             cfg.get("loss", "wce"), bool(cfg.get("polar", False)),
                             cfg.get("polar_mode") or "resample", seed, tag)
        before = original_run_id(cfg, tag)
        if got != before:
            failures.append(f"run id: today {got}, original formula {before}")
        elif got != d.name:
            legacy.append(d.name)
        n += 1
    if train.protocol_changes({}) or train.protocol_suffix({}):
        failures.append("default protocol is not empty")
    return n, legacy


def check_model(orig, failures: list) -> int:
    n = 0
    for d in sorted(RUNS.iterdir()):
        if not (d / "best.pt").exists() or not (d / "config.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        args = (cfg["K"] * cfg["subbands"],)
        kw = {"base": cfg.get("base", 32), "depth": cfg.get("depth", 3),
              "wrap": not cfg.get("polar", False)}
        state = torch.load(d / "best.pt", map_location="cpu")
        new, old = models.UNet(*args, **kw), orig.UNet(*args, **kw)
        new.load_state_dict(state, strict=True)
        old.load_state_dict(state, strict=True)
        if new.enc_drop is not None:
            failures.append(f"{d.name}: default model built an encoder dropout")
        h, w = cfg.get("grid_shape", [23, 91])
        x = torch.randn(2, args[0], h, w, generator=torch.Generator().manual_seed(0))
        new.train()
        old.train()
        torch.manual_seed(1)
        a = new(x)
        torch.manual_seed(1)
        b = old(x)
        if not all(torch.equal(u, v) for u, v in zip(a, b)):
            failures.append(f"{d.name}: training-mode outputs differ from the original UNet")
        n += 1
    return n


def check_loss(orig, failures: list) -> None:
    g = torch.Generator().manual_seed(0)
    logits, pres = torch.randn(8, 23, 91, generator=g), torch.randn(8, generator=g)
    mask = (torch.rand(8, 23, 91, generator=g) < 0.02).float()
    present = (torch.rand(8, generator=g) < 0.8).float()
    for kind in ("wce", "tversky"):
        kw = {"pos_weight": 71.7, "presence_pos_weight": 0.157}
        new, old = models.Loss(kind, **kw), orig.Loss(kind, **kw)
        a, b = new(logits, pres, mask, present), old(logits, pres, mask, present)
        if not torch.equal(a[0], b[0]):
            failures.append(f"loss {kind}: unweighted path differs from the original")
        c = new(logits, pres, mask, present, weight=torch.ones(8))
        if not torch.allclose(a[0], c[0], rtol=1e-5, atol=1e-6):
            failures.append(f"loss {kind}: all-ones weights give {float(c[0]):.6f}, "
                            f"unweighted {float(a[0]):.6f}")


class _StubBeamformer:
    """Stands in for `beamform` so `to_maps` can be compared without a cache or a GPU:
    (N, M, F) spectra -> (N, 4, 23, 91) maps that differ per frame and per cell."""
    n_subbands = 4

    def maps(self, Z):
        return (Z.abs().mean((-2, -1)).reshape(-1, 1, 1, 1)
                * torch.linspace(0, 1, self.n_subbands * 23 * 91).reshape(-1, 23, 91))


def check_maps(orig, failures: list) -> None:
    g = torch.Generator().manual_seed(0)
    X = torch.randn(3, 2, 6, 5, dtype=torch.complex64, generator=g)
    bf = _StubBeamformer()
    a = data_mod.to_maps(bf, X, 3.0, 2.0)
    b = orig.to_maps(bf, X, 3.0, 2.0)
    if not torch.equal(a, b):
        failures.append("to_maps without frame_norm differs from the original")
    f = data_mod.to_maps(bf, X, 3.0, 2.0, frame_norm=True)
    if not (torch.allclose(f.mean((-2, -1)), torch.zeros(()), atol=1e-5)
            and torch.allclose(f.std((-2, -1)), torch.ones(()), atol=1e-4)):
        failures.append("frame_norm maps are not zero-mean, unit-std per map")


def check_helpers(failures: list) -> None:
    w = train.far_weights(np.array([10.0, 60.0, 90.0, 120.0, 500.0, np.nan]), 2.0)
    if not np.allclose(w, [1.0, 1.0, 2.0, 3.0, 3.0, 1.0]):
        failures.append(f"far_weights {w.tolist()}")

    none = {"az": np.full(5, np.nan, np.float32), "el": np.full(5, np.nan, np.float32),
            "range_m": np.full(5, np.nan, np.float32), "has_drone": np.zeros(5, np.uint8)}
    lagged = propagation_lagged(none)
    if not all(np.array_equal(lagged[k], none[k], equal_nan=True) for k in none):
        failures.append("propagation_lagged changed a no-drone segment")

    # Receding along azimuth 30, elevation 20 at v m/s from r0: r(t) = r0 + v t, t in frames.
    # Emission time t_e solves t_e = k - (r0 + v t_e dt) / (c dt), so r_e = r0 + v dt t_e.
    n, r0, v, dt = 600, 20.0, 15.0, 1.0 / CSV_HZ
    k = np.arange(n, dtype=float)
    r = r0 + v * dt * k
    meta = {"az": np.full(n, 30.0, np.float32), "el": np.full(n, 20.0, np.float32),
            "range_m": r.astype(np.float32), "has_drone": np.ones(n, np.uint8)}
    t_e = (k - r0 / (C * dt)) / (1 + v / C)
    expect = r0 + v * dt * np.clip(t_e, 0, None)
    got = propagation_lagged(meta)
    if not np.allclose(got["range_m"][10:], expect[10:], atol=1e-3):
        failures.append("propagation_lagged misses the analytic emission range")
    if not (np.allclose(got["az"], 30.0, atol=1e-4) and np.allclose(got["el"], 20.0, atol=1e-4)):
        failures.append("propagation_lagged moved the direction of a radial trajectory")


def check_ablations(failures: list) -> None:
    """The Phase 3 ablation flags do what they say, on synthetic inputs."""
    g = torch.Generator().manual_seed(0)
    logits, pres = torch.randn(8, 23, 91, generator=g), torch.randn(8, generator=g)
    mask = (torch.rand(8, 23, 91, generator=g) < 0.02).float()
    present = (torch.rand(8, generator=g) < 0.8).float()
    dice = models.Loss("dice")(logits, pres, mask, present)[1]
    ref = float(models.tversky_loss(logits, mask, alpha=0.5, beta=0.5))
    if abs(dice - ref) > 1e-6:
        failures.append(f"dice loss {dice:.6f} vs Tversky(0.5, 0.5) {ref:.6f}")

    gn = models.UNet(16, base=8, depth=2, norm_layer="group")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in gn.modules()):
        failures.append("norm_layer='group' still built a BatchNorm")
    out = gn.eval()(torch.randn(1, 16, 23, 91, generator=g))
    if out[0].shape != (1, 23, 91):
        failures.append(f"GroupNorm UNet mask shape {tuple(out[0].shape)}")

    rng = np.random.default_rng(0)
    az, el = rng.uniform(-180, 180, 200), rng.uniform(0, 90, 200)
    hard = target_mask(az, el, 10.0)
    soft = target_mask(az, el, 10.0, soft=True)
    if soft.dtype != np.float32 or not np.array_equal(soft >= 0.5, hard.astype(bool)):
        failures.append("soft target at 0.5 does not reproduce the hard disc")

    for s in (1, 17, 45):
        moved = target_mask((az + 4 * s + 180) % 360 - 180, el, 10.0)
        rolled = augment.roll(torch.from_numpy(hard), s).numpy()
        if not np.array_equal(rolled, moved):
            failures.append(f"azimuth roll by {s} cells is not a {4 * s} degree rotation")
    X = torch.randn(4, 2, 6, 5, dtype=torch.complex64, generator=g)
    saved = augment.GAIN_DB, augment.DROP_P, augment.NOISE_P
    augment.GAIN_DB, augment.DROP_P, augment.NOISE_P = 0.0, 0.0, 0.0
    same = torch.equal(augment.spectra(X, torch.Generator().manual_seed(0)), X)
    augment.GAIN_DB, augment.DROP_P, augment.NOISE_P = saved
    if not same:
        failures.append("augment.spectra with every knob at zero is not the identity")

    align = {"mic_array": {"a_": [0, 120, 240], "a2_": [30, 90, 150, 210, 270, 330],
                           "h_": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
                           "r_": [0.2, 0.3, 0.4, 0.5, 0.6, 0.65]}}
    freqs = np.linspace(200, 2200, 21)
    plane = beamform.Beamformer(align, range(24), freqs, n_subbands=4, device="cpu").W
    far = beamform.Beamformer(align, range(24), freqs, n_subbands=4, device="cpu",
                              focus_m=1e6).W
    near = beamform.Beamformer(align, range(24), freqs, n_subbands=4, device="cpu",
                               focus_m=10.0).W
    if float((far * plane.conj()).angle().abs().max()) > 1e-3:
        failures.append("near-field steering at 1000 km does not reduce to plane waves")
    if float((near * plane.conj()).angle().abs().max()) < 0.1:
        failures.append("near-field steering at 10 m is indistinguishable from plane waves")

    ids = {k: train.protocol_suffix(train.protocol_changes({k: v})) for k, v in
           {"augment": True, "norm_layer": "group", "target": "soft",
            "nearfield_m": 30.0}.items()}
    if ids != {"augment": "_aug", "norm_layer": "_gn", "target": "_soft",
               "nearfield_m": "_nf30"}:
        failures.append(f"ablation run-id suffixes {ids}")


def check_epoch0(failures: list) -> None:
    run = "device3_K1_w32_b16_wce_s0"
    ref = json.loads((RUNS / run / "metrics.json").read_text())["history"][0]
    res = train.train(subset="device3", K=1, seed=0, max_epochs=1, min_epochs=1,
                      tag="gate_epoch0", device=json.loads(
                          (RUNS / run / "config.json").read_text())["device"], verbose=False)
    shutil.rmtree(RUNS / f"{run}_gate_epoch0", ignore_errors=True)
    got = res["history"][0]
    cfg_new = res["config"]
    ref_cfg = json.loads((RUNS / run / "config.json").read_text())
    for key in ("train_stacks", "input_mean", "input_std", "mask_positive_rate", "pos_weight"):
        if abs(cfg_new[key] - ref_cfg[key]) > 1e-5 * max(1.0, abs(ref_cfg[key])):
            failures.append(f"config {key}: {cfg_new[key]} vs committed {ref_cfg[key]}")
    for key, tol in EPOCH0.items():
        if abs(got[key] - ref[key]) > tol:
            failures.append(f"epoch 0 {key}: {got[key]:.4f} vs committed {ref[key]:.4f} "
                            f"(tol {tol})")
    print(f"  epoch 0: hit {got['hit_10deg']:.4f} vs {ref['hit_10deg']:.4f}, "
          f"AUROC {got['auroc']:.4f} vs {ref['auroc']:.4f}, "
          f"train_loss {got['train_loss']:.4f} vs {ref['train_loss']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--with-data", action="store_true")
    a = ap.parse_args()
    print(__doc__.split("\n\n")[0])
    failures: list[str] = []
    n, legacy = check_run_ids(failures)
    print(f"  run ids rebuilt: {n - len(legacy)} of {n}; predating the w{{base}} naming, "
          f"so no code at {a.base} rebuilds them either: {', '.join(legacy) or 'none'}")
    orig_models = original("models", a.base)
    print(f"  checkpoints loaded and matched: {check_model(orig_models, failures)}")
    check_loss(orig_models, failures)
    check_maps(original("data", a.base), failures)
    check_helpers(failures)
    check_ablations(failures)
    if a.with_data:
        check_epoch0(failures)
    for f in failures:
        print(f"  FAIL {f}")
    print("GATE PASS: default protocol reproduces the original code path"
          if not failures else "GATE FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
