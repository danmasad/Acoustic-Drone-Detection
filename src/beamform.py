"""Turn cached spectra into energy maps, on the GPU, inside the training loop.

Steering is a matmul: a delay is a phase rotation, so the map for a batch of frames is the
cached spectra multiplied by a precomputed steering tensor. It costs about 2% of a training
step, which is why the cache stores spectra and the maps are never written to disk. The
microphone subset, the band, the number of sub-bands and the direction grid are therefore
runtime flags.
"""
import numpy as np
import torch

from .config import GRID_SHAPE, MODEL_BAND, N_SUBBANDS
from .geometry import directions, mic_positions

_CACHE: dict = {}


def pick_device(prefer: str = "auto") -> torch.device:
    """MPS when it is really available. Note it is *not* available inside a sandboxed
    shell, where torch reports a misleading 'macOS 14.0+' error."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def subband_matrix(freqs: np.ndarray, n_subbands: int = N_SUBBANDS,
                   band=MODEL_BAND) -> tuple[np.ndarray, np.ndarray]:
    """(selection mask over freqs, (n_subbands, n_selected) averaging weights).

    Equal-width sub-bands, each the mean of its native bins. Averaging rather than picking
    sparse bins: adjacent 10 Hz bins are nearly identical, and a drone's harmonics slide
    with RPM, so a 125 Hz band is stable frame to frame where a single bin is not.
    """
    sel = (freqs >= band[0]) & (freqs <= band[1])
    f = freqs[sel]
    edges = np.linspace(band[0], band[1], n_subbands + 1)
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_subbands - 1)
    W = np.zeros((n_subbands, len(f)), dtype=np.float32)
    W[idx, np.arange(len(f))] = 1.0
    counts = W.sum(1, keepdims=True)
    if (counts == 0).any():
        raise ValueError(f"empty sub-band with n={n_subbands} over {band} Hz")
    return sel, W / counts


class Beamformer:
    """Steering for one (session geometry, microphone subset, band, sub-band count).

    Tensors are built once and memoized, because the grid steering tensor is ~40 MB and
    rebuilding it per batch would cost more than the beamforming itself.
    """

    def __init__(self, align: dict, channels, freqs: np.ndarray,
                 n_subbands: int = N_SUBBANDS, band=MODEL_BAND,
                 device: torch.device | str = "auto", dirs: np.ndarray | None = None,
                 grid_shape=GRID_SHAPE):
        self.channels = list(channels)
        self.device = pick_device(device) if isinstance(device, str) else device
        self.grid_shape = grid_shape
        self.n_subbands = n_subbands

        sel, sub = subband_matrix(freqs, n_subbands, band)
        self.sel = sel
        pos = mic_positions(align)[self.channels]
        dirs = directions() if dirs is None else dirs
        tau = dirs @ pos.T / 343.0                                   # (D, M)
        W = np.exp(-2j * np.pi * freqs[sel][:, None, None] * tau[None])

        # (F, M, D) so steering is a batched matmul over the frequency axis.
        self.W = torch.from_numpy(W.transpose(0, 2, 1).astype(np.complex64)).to(self.device)
        self.sub = torch.from_numpy(sub).to(self.device)
        self.norm = 1.0 / len(self.channels)                         # keep scale comparable

    @property
    def n_freqs(self) -> int:
        return int(self.sel.sum())

    def maps(self, X: torch.Tensor, log: bool = True, eps: float = 1e-12) -> torch.Tensor:
        """(B, n_subbands) + grid_shape from (B, M, F) complex spectra already on device.

        Returns decibels by default: energy spans orders of magnitude with distance, and a
        log scale is what makes a fixed global standardization usable across sessions.
        """
        Y = torch.bmm(X.permute(2, 0, 1), self.W)                    # (F, B, D)
        e = (Y.real ** 2 + Y.imag ** 2) * (self.norm ** 2)
        e = torch.einsum("sf,fbd->bsd", self.sub, e)                 # (B, S, D)
        if log:
            e = 10.0 * torch.log10(e + eps)
        return e.reshape(e.shape[0], self.n_subbands, *self.grid_shape)


def get(align: dict, channels, freqs: np.ndarray, key: str = "", **kw) -> Beamformer:
    """Memoized Beamformer. `key` should identify the session calibration."""
    ck = (key, tuple(channels), kw.get("n_subbands", N_SUBBANDS),
          kw.get("band", MODEL_BAND), str(kw.get("device", "auto")))
    if ck not in _CACHE:
        _CACHE[ck] = Beamformer(align, channels, freqs, **kw)
    return _CACHE[ck]


def peak_direction(maps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Brightest pixel of each map, summed over sub-bands. Returns (el index, az index)."""
    flat = maps.sum(1).flatten(1)
    k = flat.argmax(1)
    return k // maps.shape[-1], k % maps.shape[-1]
