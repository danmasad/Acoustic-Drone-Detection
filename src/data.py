"""Batching over the cached spectra, with frame stacking.

Two constraints shape this module:

- **Batches never mix sessions.** The array was recalibrated between sessions, so steering
  differs, and a batch has to share one steering tensor. Indices are shuffled globally and
  then bucketed by segment, so batches stay random without mixing geometries.
- **Stacks never cross a segment boundary.** A stack is the K frames ending at the labelled
  frame, so it is causal by construction, and the first K-1 frames of each segment are
  dropped rather than padded.
"""
import numpy as np
import torch

from .config import (ANGULAR_THRESHOLD, GRID_SHAPE, MODEL_BAND, N_SUBBANDS, segments)
from .labels import propagation_lagged, target_mask
from .stft_cache import load_cache


class SplitData:
    """Every segment of one split, pre-selected down to the microphones and bins in use.

    `segs` overrides the split lookup with an explicit segment list, which is what
    leave-one-scene-out needs: folds there are defined by recording session and flight rather
    than by the dataset's own train/test roles, so they cannot be expressed as a split name.
    """

    def __init__(self, split: str, channels, K: int = 1, band=MODEL_BAND,
                 radius: float = ANGULAR_THRESHOLD, verbose: bool = True, segs=None,
                 polar: bool = False, label_lag: bool = False):
        """`label_lag` moves every label, and therefore every target mask, to the drone's
        position at sound-emission time (`labels.propagation_lagged`). For training targets
        only: scoring always uses the dataset's own labels, so runs stay comparable."""
        from . import polar as polar_grid

        self.split, self.channels, self.K, self.band = split, list(channels), K, band
        self.polar = polar
        make_mask = polar_grid.target_mask if polar else target_mask
        self.segs, self.X, self.masks, self.meta = [], [], [], []
        self.index = []
        for seg in (segments(split) if segs is None else list(segs)):
            stft, meta = load_cache(seg)
            if label_lag:
                meta = propagation_lagged(meta)
            sel = (meta["freqs"] >= band[0]) & (meta["freqs"] <= band[1])
            X = np.ascontiguousarray(np.asarray(stft)[:, self.channels][:, :, sel])
            self.freqs = meta["freqs"][sel]
            s = len(self.segs)
            self.segs.append(seg)
            self.X.append(X)
            self.masks.append(make_mask(meta["az"], meta["el"], radius))
            self.meta.append(meta)
            self.index += [(s, f) for f in range(K - 1, len(X))]
        self.index = np.array(self.index, dtype=np.int64)
        if verbose:
            gib = sum(x.nbytes for x in self.X) / 2**30
            print(f"{split}: {len(self.index)} stacks over {len(self.segs)} segments, "
                  f"{len(self.channels)} mics x {len(self.freqs)} bins, {gib:.2f} GiB in RAM")

    def __len__(self) -> int:
        return len(self.index)

    @property
    def positive_rate(self) -> float:
        """Fraction of frames with a drone. The detection bar to beat is this number."""
        d = np.concatenate([m["has_drone"] for m in self.meta])
        return float(d.mean())

    def mask_positive_rate(self) -> float:
        """Fraction of pixels that are inside the target disc, over drone frames only."""
        tot = sum(m.sum() for m in self.masks)
        n_drone = sum(int(m["has_drone"].sum()) for m in self.meta)
        cells = int(np.prod(self.masks[0].shape[1:])) if self.masks else \
            GRID_SHAPE[0] * GRID_SHAPE[1]
        return float(tot / max(n_drone, 1) / cells)

    def batches(self, batch: int = 32, shuffle: bool = True, rng=None):
        """Yields (segment index, X, mask, present, frame indices).

        X is (B, K, M, F) complex64: the caller beamforms, because that is the step whose
        cost depends on the microphone subset and which belongs on the GPU.
        """
        order = np.arange(len(self.index))
        if shuffle:
            (rng or np.random.default_rng()).shuffle(order)
        buckets: dict[int, list[int]] = {}
        for k in order:
            s, f = self.index[k]
            buckets.setdefault(int(s), []).append(int(f))
            if len(buckets[int(s)]) == batch:
                yield self._make(int(s), buckets.pop(int(s)))
        for s, frames in list(buckets.items()):
            if frames:
                yield self._make(s, frames)

    def _make(self, s: int, frames: list[int]):
        f = np.asarray(frames)
        offs = np.arange(-(self.K - 1), 1)                       # current frame last
        idx = f[:, None] + offs[None, :]                         # (B, K)
        X = self.X[s][idx]                                       # (B, K, M, F)
        mask = self.masks[s][f]
        present = self.meta[s]["has_drone"][f]
        return (s, torch.from_numpy(X),
                torch.from_numpy(mask.astype(np.float32)),
                torch.from_numpy(present.astype(np.float32)), f)


def to_maps(bf, X: torch.Tensor, mean: float = 0.0, std: float = 1.0,
            projection=None, frame_norm: bool = False) -> torch.Tensor:
    """(B, K, M, F) spectra -> (B, K*S, 23, 91) standardized decibel maps.

    With a `projection` (see `src.polar`) the maps come back as (B, K*S, 46, 46) on the disc.
    Standardization happens first so the projection's zero fill for the dead corners is the
    mean of the distribution rather than an arbitrary dark value.

    `frame_norm` standardizes each map on its own (per example and channel, over its cells)
    instead of with the global training statistics, so a level shift between sites cannot move
    the input; `mean` and `std` are then ignored. It erases absolute level, which is the
    presence head's cue, so it is a localization protocol, not a detection one.
    """
    B, K = X.shape[:2]
    maps = bf.maps(X.reshape(B * K, *X.shape[2:]))
    maps = maps.reshape(B, K * bf.n_subbands, *maps.shape[-2:])
    if frame_norm:
        mu = maps.mean((-2, -1), keepdim=True)
        maps = (maps - mu) / maps.std((-2, -1), keepdim=True).clamp_min(1e-6)
    else:
        maps = (maps - mean) / std
    return maps if projection is None else projection(maps)


def input_stats(data: SplitData, beamformers, n_batches: int = 16, batch: int = 32,
                seed: int = 0, valid=None) -> tuple[float, float]:
    """Global mean and standard deviation of the decibel maps, from a sample of training
    batches. Global rather than per-frame: per-frame normalization would erase the loudness
    difference between a drone and silence, which is the detection cue.

    `valid` is a flat boolean mask over the map's cells, for a layout where some cells describe
    no direction. It matters for the natively steered polar disc: its dead corners all carry the
    zenith placeholder direction, so leaving them in would let one real direction's energy
    contribute 21.5% of the statistics and pull the standardization toward the zenith.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for i, (s, X, *_ ) in enumerate(data.batches(batch, rng=rng)):
        bf = beamformers(s)
        m = bf.maps(X.reshape(-1, *X.shape[2:]).to(bf.device))
        if valid is not None:
            m = m.reshape(*m.shape[:2], -1)[..., valid.to(m.device)]
        vals.append(m.flatten().to("cpu"))
        if i + 1 >= n_batches:
            break
    v = torch.cat(vals)
    return float(v.mean()), float(v.std())
