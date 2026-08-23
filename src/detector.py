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


def spectra(X: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """(B, K, M, F) complex -> (B*M, K, F) standardized log-magnitude, channels as samples."""
    mag = 20.0 * torch.log10(X.abs() + 1e-9)
    mag = (mag - mean) / std
    return mag.permute(0, 2, 1, 3).flatten(0, 1)


@torch.no_grad()
def evaluate(model, data: SplitData, stats, device, batch: int = 32,
             channel_counts=(1, 2, 6, 24)) -> dict:
    """Frame-level metrics from averaged per-channel logits, plus a channel-count sweep."""
    model.eval()
    logits, labels = [], []
    for s, X, _, present, _ in data.batches(batch, shuffle=False):
        x = spectra(X.to(device), *stats)
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
    return out


def train(K: int = 1, seed: int = 0, lr: float = 1e-3, batch: int = 32, width: int = 32,
          epochs: int = 12, patience: int = 4, device: str = "auto", tag: str = "",
          verbose: bool = True) -> dict:
    from .beamform import pick_device

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(device)
    run_id = f"detector_K{K}_w{width}_s{seed}" + (f"_{tag}" if tag else "")
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)

    channels = MIC_SUBSETS["all24"]
    train_data = SplitData("train", channels, K, band=CACHE_BAND, verbose=verbose)
    val_data = SplitData("test_2", channels, K, band=CACHE_BAND, verbose=verbose)

    sample = next(train_data.batches(64, rng=rng))[1]
    m = 20.0 * torch.log10(sample.abs() + 1e-9)
    stats = (float(m.mean()), float(m.std()))

    drone_rate = train_data.positive_rate
    model = SpectralDetector(K, width).to(dev)
    pw = torch.tensor((1 - drone_rate) / max(drone_rate, 1e-6), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if verbose:
        print(f"{run_id}: {model.n_params / 1e3:.0f}k params, {len(train_data.freqs)} bins, "
              f"{len(train_data) * len(channels):,} channel-frames per epoch, device {dev}")

    best, best_epoch, best_state, history = -np.inf, -1, None, []
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for s, X, _, present, _ in train_data.batches(batch, rng=rng):
            x = spectra(X.to(dev), *stats)
            # Every channel of a frame inherits the frame's label.
            y = present.to(dev).unsqueeze(1).expand(-1, len(channels)).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(y)
            n += len(y)
        m_ = evaluate(model, val_data, stats, dev)
        m_ |= {"epoch": epoch, "train_loss": tot / max(n, 1), "seconds": time.time() - t0}
        history.append(m_)
        if verbose:
            print(f"  epoch {epoch:>2}  {m_['seconds']:>5.0f}s  train {m_['train_loss']:.4f}  "
                  f"AUROC {m_['auroc']:.4f} (1 mic {m_['by_n_channels']['1']:.4f})  "
                  f"reject@95 {m_['specificity_at_95_recall'] * 100:.0f}%")
        if m_["auroc"] > best + 1e-3:
            best, best_epoch = m_["auroc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"  early stop: no AUROC gain since epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, out / "best.pt")
    final = evaluate(model, val_data, stats, dev)
    result = {"config": {"run_id": run_id, "K": K, "width": width, "seed": seed, "lr": lr,
                         "batch": batch, "channels": channels, "band": list(CACHE_BAND),
                         "input_mean": stats[0], "input_std": stats[1],
                         "n_params": model.n_params, "train_drone_rate": drone_rate},
              "best_epoch": best_epoch, "test_2": final, "history": history}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    if verbose:
        print(f"\n{run_id} best epoch {best_epoch}: AUROC {final['auroc']:.3f}, "
              f"AUPRC {final['auprc']:.3f}, rejects "
              f"{final['specificity_at_95_recall'] * 100:.0f}% of negatives at 95% recall")
        print("  channels needed: " + "  ".join(
            f"{n} mic {v:.3f}" for n, v in final["by_n_channels"].items()))
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
    a = p.parse_args()
    train(K=a.K, width=a.width, seed=a.seed, lr=a.lr, batch=a.batch, epochs=a.epochs,
          device=a.device, tag=a.tag)
