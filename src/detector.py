"""Per-channel spectral detector: is a drone audible in this one microphone?

Separate from the map-based model on purpose. Detection is a question about timbre — a
drone is a set of harmonics from the blade-pass frequency — and answering it per channel has
four consequences:

- No cross-channel structure to exploit, so injected mono audio (ESC-50, Al-Emadi) can be
  mixed in later without the synthetic-spatial shortcut that plane-wave rendering creates.
- 24 training samples per frame instead of one, which is what the starved detection head
  needs: 4,911 negative frames become 117,864 negative examples.
- Detection becomes independent of microphone count by construction, so the 6-versus-24
  comparison isolates to localization, where the physics actually differs.
- It costs minutes, not hours: the model is ~100k parameters over a 381-bin spectrum.

Frame-level decisions come from averaging per-channel logits, which also lets us report how
many microphones a stable decision needs.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import metrics
from .config import CACHE_BAND, MIC_SUBSETS, RUNS
from .data import SplitData
from .esc50 import HARD_CLASSES


def log_filterbank(freqs: np.ndarray, n_bins: int = 64,
                   band=(200.0, 4000.0)) -> np.ndarray:
    """(n_bins, F) triangular filters with log-spaced centres. A fixed matrix, not a layer.

    Why the frequency axis should be logarithmic here, in one paragraph, because it is the
    only physics-motivated design choice in the detector:

    A multirotor's sound is a harmonic comb — energy at the blade-pass frequency (rotor RPM
    over 60, times the number of blades) and at every integer multiple of it. Throttle up to
    climb or fight wind and the RPM rises, which scales *every* harmonic by the same factor.
    On a linear axis that is a different shift in hertz for each harmonic, since the tenth
    moves ten times as far as the first: the comb stretches. Convolution is invariant to
    sliding, not to stretching, so a linear-axis network has to learn the low-RPM comb, the
    mid-RPM comb and the high-RPM comb as separate patterns. On a log axis multiplication
    becomes addition, the comb slides rigidly, and one learned template covers every throttle
    setting. Nothing is added to the model; a variation it had to memorize is removed.

    Bin count is capped by the cache, not chosen freely. The spectra have 10 Hz spacing, and
    at the bottom of the band 10 Hz is 5% of 200 Hz, so log bins narrower than about 5% would
    interpolate rather than measure. Over 200-4000 Hz that allows roughly 64 bins, which also
    shrinks the input from 381 values to 64.

    Two costs, reported either way: the top of the band is coarsened, so high harmonics blur
    together (acceptable — the low harmonics carry the distinctiveness), and if the detector is
    keying on broadband energy rather than comb structure then this changes nothing, in which
    case the null result says something about what it learned.
    """
    edges = np.geomspace(band[0], band[1], n_bins + 2)
    bank = np.zeros((n_bins, len(freqs)), np.float32)
    for i in range(n_bins):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs >= lo) & (freqs <= mid)
        dn = (freqs > mid) & (freqs <= hi)
        bank[i, up] = (freqs[up] - lo) / max(mid - lo, 1e-9)
        bank[i, dn] = (hi - freqs[dn]) / max(hi - mid, 1e-9)
        s = bank[i].sum()
        if s > 0:
            bank[i] /= s                   # unit area, so a flat spectrum stays flat
    return bank


class SpectralDetector(nn.Module):
    """(B, K, F) log-magnitude spectra -> one logit per channel-frame."""

    def __init__(self, in_frames: int = 1, width: int = 32, dropout: float = 0.2):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv1d(in_frames, c, 7, stride=2, padding=3), nn.BatchNorm1d(c), nn.ReLU(True),
            nn.Conv1d(c, c * 2, 5, stride=2, padding=2), nn.BatchNorm1d(c * 2), nn.ReLU(True),
            nn.Conv1d(c * 2, c * 4, 3, stride=2, padding=1), nn.BatchNorm1d(c * 4),
            nn.ReLU(True),
            nn.Conv1d(c * 4, c * 4, 3, stride=2, padding=1), nn.BatchNorm1d(c * 4),
            nn.ReLU(True))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c * 8, 1))

    def forward(self, x):
        h = self.net(x)
        pooled = torch.cat([h.mean(-1), h.amax(-1)], -1)
        return self.head(pooled).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def spectra(X: torch.Tensor, mean: float = 0.0, std: float = 1.0,
            bank: torch.Tensor | None = None) -> torch.Tensor:
    """(B, K, M, F) complex -> (B*M, K, F') standardized log-magnitude, channels as samples.

    With a filterbank the pooling happens on linear magnitude before the decibel conversion,
    because summing energy and then taking the logarithm is what a filter does; taking the
    logarithm first would average decibels, which is a geometric mean and would let one dead
    bin drag a whole band down.
    """
    mag = X.abs()
    if bank is not None:
        mag = mag @ bank.T
    mag = 20.0 * torch.log10(mag + 1e-9)
    mag = (mag - mean) / std
    return mag.permute(0, 2, 1, 3).flatten(0, 1)


@torch.no_grad()
def evaluate(model, data: SplitData, stats, device, batch: int = 32,
             channel_counts=(1, 2, 6, 24), bank=None) -> dict:
    """Frame-level metrics from averaged per-channel logits, plus a channel-count sweep."""
    model.eval()
    logits, labels = [], []
    for s, X, _, present, _ in data.batches(batch, shuffle=False):
        x = spectra(X.to(device), *stats, bank=bank)
        out = model(x).reshape(len(present), -1)                  # (B, M)
        logits.append(out.cpu().numpy())
        labels.append(present.numpy())
    L, y = np.concatenate(logits), np.concatenate(labels)
    out = {"auroc": metrics.auroc(L.mean(1), y),
           "auprc": metrics.auprc(L.mean(1), y),
           "specificity_at_95_recall": metrics.specificity_at_recall(L.mean(1), y),
           "per_channel_auroc": float(np.mean([metrics.auroc(L[:, m], y)
                                               for m in range(L.shape[1])])),
           "by_n_channels": {}}
    f1 = metrics.best_f1(L.mean(1), y)
    out |= {f"f1_{k}": v for k, v in f1.items()}
    out["false_alarms_per_hour"] = metrics.false_alarms_per_hour(L.mean(1), y, f1["threshold"])
    for n in channel_counts:
        if n <= L.shape[1]:
            out["by_n_channels"][str(n)] = metrics.auroc(L[:, :n].mean(1), y)
    # The operating point every confuser number is quoted at: catch 95% of drone frames.
    out["threshold_at_95_recall"] = float(np.quantile(L.mean(1)[y.astype(bool)], 0.05))
    out["n_frames"] = int(len(y))
    return out


@torch.no_grad()
def confuser_rejection(model, data: SplitData, stats, device, conf, threshold: float,
                       batch: int = 32, seed: int = 0, bank=None) -> dict:
    """How often a held-out environmental sound, added to our own ambient, reads as a drone.

    Only no-drone frames are used, and the confuser is added on top of them, so every example
    keeps our array's real noise floor and the label stays honest: still no drone, now with a
    chainsaw. The rate is reported per class, because a mean over fifty classes averages a
    helicopter together with a ticking clock and hides the only cases that matter.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    fired: dict[str, list] = {}
    clean_fired = []
    for s, X, _, present, _ in data.batches(batch, shuffle=False, rng=rng):
        keep = ~present.bool().numpy()
        if not keep.any():
            continue
        Xn = X[torch.from_numpy(keep)]
        clean = model(spectra(Xn.to(device), *stats, bank=bank))
        clean_fired.append((clean.reshape(len(Xn), -1).mean(1) >= threshold).cpu().numpy())
        mixed, flag, cats = conf.mix_into(Xn, rng, prob=1.0)
        out = model(spectra(mixed.to(device), *stats, bank=bank)).reshape(len(Xn), -1).mean(1)
        hit = (out >= threshold).cpu().numpy()
        for i, c in enumerate(cats):
            if flag[i]:
                fired.setdefault(c, []).append(bool(hit[i]))
    per_class = {c: {"n": len(v), "fire_rate": float(np.mean(v))}
                 for c, v in sorted(fired.items())}
    hard = [per_class[c]["fire_rate"] for c in HARD_CLASSES if c in per_class]
    return {"clean_fire_rate": float(np.concatenate(clean_fired).mean()),
            "mean_fire_rate": float(np.mean([v["fire_rate"] for v in per_class.values()])),
            "hard_class_fire_rate": float(np.mean(hard)) if hard else float("nan"),
            "worst_class": max(per_class, key=lambda c: per_class[c]["fire_rate"]),
            "per_class": per_class}


MONITOR_BLOCK = 300        # 30 s at 10 Hz


def split_for_early_stopping(data: SplitData, fraction: float = 0.2,
                             block: int = MONITOR_BLOCK):
    """Split one SplitData into (fit, monitor) by holding out interleaved blocks of frames.

    Why this exists: with a held-out *scene* as the test set, early stopping on that scene is
    selection on the test set, and here it is not a small effect. The detector's AUROC on an
    unseen scene swings between 0.36 and 0.80 from one epoch to the next, so "best epoch on the
    held-out scene" would report a maximum over twelve draws that a rerun could not reproduce.
    The monitor set therefore has to come out of the training scenes.

    Why blocks rather than a contiguous tail, which is what this did first: the tail of a flight
    segment is the end of the flight, where the drone is far away and quiet, and the tail of a
    pre-takeoff segment is the moment just before the motors start. A tail split therefore pairs
    the hardest positives against the most drone-adjacent negatives, and it showed: one fold's
    monitor AUROC sat below 0.4 for most of training, which says more about the split than about
    the model. Interleaving 30-second blocks keeps the monitor set representative in range,
    level and class balance.

    Why not random frames: frames 100 ms apart are near-duplicates, so a random holdout puts a
    frame's own neighbours on the other side of the split and measures memorization. With
    30-second blocks only the two frames at each block boundary have a neighbour across it.

    This monitors convergence, not generalization — it is the same scene. Generalization is what
    the held-out scene measures, once, at the end. The copy is shallow on purpose: the two views
    share the spectra, which are gigabytes.
    """
    import copy

    every = max(int(round(1.0 / max(fraction, 1e-6))), 2)
    fit_rows, monitor_rows = [], []
    for s in range(len(data.segs)):
        rows = np.where(data.index[:, 0] == s)[0]
        if len(rows) == 0:
            continue
        # Short segments get proportionally shorter blocks, so that every segment contributes
        # to the monitor set — our 75-second pre-takeoff negatives are too scarce to skip. The
        # held block rotates with the segment index so it is never systematically the tail.
        size = max(1, min(block, len(rows) // every))
        held = ((data.index[rows, 1] // size) % every) == (s % every)
        if held.all() or not held.any():
            fit_rows.append(rows)
            continue
        fit_rows.append(rows[~held])
        monitor_rows.append(rows[held])
    fit, monitor = copy.copy(data), copy.copy(data)
    fit.index = data.index[np.concatenate(fit_rows)]
    monitor.index = data.index[np.concatenate(monitor_rows)]
    return fit, monitor


def load_run(run_id: str, device: str = "auto"):
    """A finished detector, ready to score anything: (model, config, stats, bank, device)."""
    from .beamform import pick_device

    saved = json.loads((RUNS / run_id / "metrics.json").read_text())
    cfg = saved["config"]
    dev = pick_device(device)
    model = SpectralDetector(cfg["K"], cfg["width"]).to(dev).eval()
    model.load_state_dict(torch.load(RUNS / run_id / "best.pt", map_location=dev))
    bank = None
    if cfg.get("log_bins"):
        freqs = np.fft.rfftfreq(4800, 1 / 48000)
        freqs = freqs[(freqs >= cfg["band"][0]) & (freqs <= cfg["band"][1])]
        bank = torch.from_numpy(log_filterbank(freqs, cfg["log_bins"])).to(dev)
    return model, saved, (cfg["input_mean"], cfg["input_std"]), bank, dev


def ambient_reference_energy(data: SplitData) -> float:
    """Median in-band energy of one channel-frame of our own *training* no-drone audio.

    This is the level DEMAND scenes get matched to before they are used as training negatives.
    It has to come from the training split: `utils/demand_false_alarms.py` takes the same
    statistic from test 2, which is right for an evaluation that must be in the units the
    reported threshold lives in, and would be a leak here.

    Taken straight off the labelled frames rather than by sampling batches, because batches are
    bucketed by segment: unshuffled, the first several hundred are all from the first flight and
    contain no negatives at all, and a shuffled sample would make the reference level depend on
    the seed.
    """
    got = []
    for X, meta in zip(data.X, data.meta):
        quiet = ~meta["has_drone"].astype(bool)
        if quiet.any():
            got.append((np.abs(X[quiet]) ** 2).sum(-1).reshape(-1))
    if not got:
        raise ValueError("no no-drone frames in the training data to level-match against")
    return float(np.median(np.concatenate(got)))


def ambient_pool(groups, data: SplitData, K: int, verbose: bool = True):
    """(n, K, 1, F) complex64 stacks of DEMAND ambient from `groups`, in our own units.

    Single channel, because the detector reads one microphone at a time. Replicating a mono
    recording across 24 channels would hand the map model a spatial pattern no source produces,
    which is why this path exists only for the per-channel detector.
    """
    from . import demand

    ref = ambient_reference_energy(data)
    amb = demand.Ambient(reference_energy=ref, verbose=verbose)
    scenes = [s for g in groups for s in demand.GROUPS[g] if s in amb.scenes]
    pool = np.concatenate([amb.frames(s, K=K, channels=1) for s in scenes])
    if verbose:
        print(f"  ambient negatives: {len(pool)} frames "
              f"({len(pool) * 0.1 / 60:.1f} min) from {len(scenes)} scenes in "
              f"{', '.join(groups)}, level-matched to our training ambient ({ref:.3e})")
    return pool, scenes, ref


def train(K: int = 1, seed: int = 0, lr: float = 1e-3, batch: int = 32, width: int = 32,
          epochs: int = 12, patience: int = 4, device: str = "auto", tag: str = "",
          verbose: bool = True, train_segs=None, val_segs=None, confusers: bool = False,
          mix_prob: float = 0.5, log_bins: int = 0, monitor_fraction: float = 0.0,
          ambient_groups=None, ambient_frac: float = 0.3) -> dict:
    """One detector. `confusers` mixes ESC-50 into training frames, `log_bins` switches the
    frequency axis to a log-spaced filterbank of that many bins (0 keeps the linear axis).

    `monitor_fraction > 0` moves early stopping onto the tail of the *training* scenes, which is
    required whenever the validation set is a held-out scene whose number will be reported.

    `ambient_groups` names DEMAND scene groups to draw extra negatives from, `ambient_frac` how
    many relative to the channel-frames in each batch. This is addition, not mixing: the frames
    are real continuous ambient labelled no-drone, which is the thing the detector was firing on
    at 24,881 alarms per hour. Groups not named here are what the false-alarm rate is then
    measured on, so the question stays "does hearing real ambient generalize to other ambient".
    """
    from .beamform import pick_device

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(device)
    run_id = (f"detector_K{K}_w{width}" + ("_conf" if confusers else "")
              + (f"_log{log_bins}" if log_bins else "")
              + ("_amb" if ambient_groups else "") + f"_s{seed}"
              + (f"_{tag}" if tag else ""))
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)

    channels = MIC_SUBSETS["all24"]
    train_data = SplitData("train", channels, K, band=CACHE_BAND, verbose=verbose,
                           segs=train_segs)
    val_data = SplitData("test_2", channels, K, band=CACHE_BAND, verbose=verbose,
                         segs=val_segs)
    monitor_data = val_data
    if monitor_fraction > 0:
        train_data, monitor_data = split_for_early_stopping(train_data, monitor_fraction)
        if verbose:
            print(f"  early stopping on {monitor_fraction * 100:.0f}% of the training scenes, "
                  f"held out as interleaved {MONITOR_BLOCK // 10}s blocks "
                  f"({len(monitor_data)} frames), never on the held-out scene")

    bank = None
    if log_bins:
        bank = torch.from_numpy(log_filterbank(train_data.freqs, log_bins)).to(dev)

    # Mixing confusers into *both* classes on purpose: if only negatives were mixed, "contains
    # an added sound" would become a perfect cue for the negative class and the model would
    # learn that instead of learning what a drone sounds like.
    # The held-out fold is loaded whether or not we train on confusers, because the interesting
    # number is the difference: what mixing buys against a model that never saw one.
    conf_train = conf_test = None
    try:
        from .esc50 import TEST_FOLD, Confusers
        conf_test = Confusers(folds=(TEST_FOLD,), verbose=verbose)
        if confusers:
            conf_train = Confusers(folds=tuple(f for f in range(1, 6) if f != TEST_FOLD),
                                   verbose=verbose)
    except FileNotFoundError:
        if confusers:
            raise
        if verbose:
            print("  ESC-50 cache absent, skipping the confuser-rejection measurement")

    sample = next(train_data.batches(64, rng=rng))[1]
    mag = sample.abs() if bank is None else sample.abs() @ bank.cpu().T
    m = 20.0 * torch.log10(mag + 1e-9)
    stats = (float(m.mean()), float(m.std()))

    pool = pool_scenes = None
    if ambient_groups:
        pool, pool_scenes, _ = ambient_pool(ambient_groups, train_data, K, verbose=verbose)

    drone_rate = train_data.positive_rate
    # Ambient negatives change the balance the loss has to be weighted against: every added
    # frame is a negative, so the effective positive rate falls by 1/(1+ambient_frac) and a
    # pos_weight computed from our own frames alone would over-weight the drone class.
    effective_rate = drone_rate / (1.0 + (ambient_frac if ambient_groups else 0.0))
    model = SpectralDetector(K, width).to(dev)
    pw = torch.tensor((1 - effective_rate) / max(effective_rate, 1e-6), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if verbose:
        n_bins = log_bins or len(train_data.freqs)
        print(f"{run_id}: {model.n_params / 1e3:.0f}k params, {n_bins} bins"
              + (f" (log, from {len(train_data.freqs)})" if bank is not None else "")
              + f", {len(train_data) * len(channels):,} channel-frames per epoch, device {dev}")

    best, best_epoch, best_state, history = -np.inf, -1, None, []
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for s, X, _, present, _ in train_data.batches(batch, rng=rng):
            if conf_train is not None:
                X, _, _ = conf_train.mix_into(X, rng, prob=mix_prob)
            x = spectra(X.to(dev), *stats, bank=bank)
            # Every channel of a frame inherits the frame's label.
            y = present.to(dev).unsqueeze(1).expand(-1, len(channels)).reshape(-1)
            if pool is not None:
                # Sampled with replacement each batch rather than iterated: the pool is 45
                # minutes of ambient against 60 of ours, and an epoch defined by our frames
                # should see a fresh draw of it rather than a fixed pairing.
                pick = rng.integers(0, len(pool), max(1, int(round(ambient_frac * len(y)))))
                xa = spectra(torch.from_numpy(pool[pick]).to(dev), *stats, bank=bank)
                x = torch.cat([x, xa])
                y = torch.cat([y, torch.zeros(len(xa), device=dev)])
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
            n += len(y)
        m_ = evaluate(model, monitor_data, stats, dev, bank=bank)
        m_ |= {"epoch": epoch, "train_loss": tot / max(n, 1), "seconds": time.time() - t0}
        history.append(m_)
        if verbose:
            print(f"  epoch {epoch:>2}  {m_['seconds']:>5.0f}s  train {m_['train_loss']:.4f}  "
                  f"AUROC {m_['auroc']:.4f} (1 mic {m_['by_n_channels']['1']:.4f})  "
                  f"reject@95 {m_['specificity_at_95_recall'] * 100:.0f}%"
                  + ("  [monitor]" if monitor_data is not val_data else ""))
        if m_["auroc"] > best + 1e-3:
            best, best_epoch = m_["auroc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"  early stop: no AUROC gain since epoch {best_epoch}")
            break

    # Score the held-out data twice: with the checkpoint early stopping chose, and with the
    # weights training happened to end on. When those two disagree by more than the effect being
    # measured — and on an unseen scene they do — the gap is the result, not a detail. Reporting
    # only the selected checkpoint would quote the maximum of a noisy sequence.
    last = evaluate(model, val_data, stats, dev, bank=bank)
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out / "best.pt")
    final = evaluate(model, val_data, stats, dev, bank=bank)
    result = {"config": {"run_id": run_id, "K": K, "width": width, "seed": seed, "lr": lr,
                         "batch": batch, "channels": channels, "band": list(CACHE_BAND),
                         "input_mean": stats[0], "input_std": stats[1],
                         "n_params": model.n_params, "train_drone_rate": drone_rate,
                         "effective_drone_rate": effective_rate,
                         "ambient_groups": list(ambient_groups) if ambient_groups else [],
                         "ambient_scenes": pool_scenes or [],
                         "ambient_frac": ambient_frac if ambient_groups else 0.0,
                         "confusers": confusers, "mix_prob": mix_prob,
                         "log_bins": log_bins, "monitor_fraction": monitor_fraction,
                         "early_stopping_on": ("interleaved blocks of the training scenes"
                                               if monitor_fraction > 0 else "validation split"),
                         "train_segments": [s.id for s in train_data.segs],
                         "val_segments": [s.id for s in val_data.segs]},
              "best_epoch": best_epoch, "test_2": final, "last_epoch": last,
              "history": history}
    if conf_test is not None:
        result["confuser_rejection"] = confuser_rejection(
            model, val_data, stats, dev, conf_test, final["threshold_at_95_recall"],
            seed=seed, bank=bank)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    if verbose:
        print(f"\n{run_id} best epoch {best_epoch}: AUROC {final['auroc']:.3f}, "
              f"AUPRC {final['auprc']:.3f}, rejects "
              f"{final['specificity_at_95_recall'] * 100:.0f}% of negatives at 95% recall"
              f"  (last epoch would have given AUROC {last['auroc']:.3f})")
        print("  channels needed: " + "  ".join(
            f"{n} mic {v:.3f}" for n, v in final["by_n_channels"].items()))
        if "confuser_rejection" in result:
            c = result["confuser_rejection"]
            print(f"  at 95% recall, ESC-50 fold 5 fires {c['mean_fire_rate'] * 100:.0f}% of "
                  f"the time on average, {c['hard_class_fire_rate'] * 100:.0f}% on the "
                  f"drone-like classes, worst is {c['worst_class']}; our own clean ambient "
                  f"fires {c['clean_fire_rate'] * 100:.0f}%")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the per-channel spectral detector.")
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="")
    p.add_argument("--confusers", action="store_true",
                   help="mix ESC-50 folds 1-4 into training frames")
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--log-bins", type=int, default=0,
                   help="log-spaced filterbank of this many bins; 0 keeps the linear axis")
    p.add_argument("--ambient-groups", nargs="*", default=None,
                   help="DEMAND scene groups to draw extra training negatives from; the rest "
                        "are what the false-alarm rate is then measured on")
    p.add_argument("--ambient-frac", type=float, default=0.3)
    a = p.parse_args()
    train(K=a.K, width=a.width, seed=a.seed, lr=a.lr, batch=a.batch, epochs=a.epochs,
          device=a.device, tag=a.tag, confusers=a.confusers, mix_prob=a.mix_prob,
          log_bins=a.log_bins, ambient_groups=a.ambient_groups,
          ambient_frac=a.ambient_frac)
