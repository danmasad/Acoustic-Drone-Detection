"""The network: an encoder-decoder over the energy map, with two heads.

Depth 3 and 3x3 kernels follow the reference paper. The base width is 32 against their 64:
2.3x faster, overfits less, about 2 M parameters.

Two departures, both deliberate:

- **Circular padding on the azimuth axis.** The map's left and right edges are the same
  direction, so zero padding would invent a discontinuity at +/-180 degrees and make the
  model worse at exactly one bearing.
- **A presence head.** Their model outputs only a per-pixel map, which is why their
  detection number is a proxy. A pooled bottleneck classifier gives a real drone/no-drone
  decision, so precision-recall and false alarms per hour become measurable.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import AZ, EL
from .geometry import directions, to_angles


def decode_mask(logits: torch.Tensor, mode: str = "centroid", threshold: float = 0.5,
                dirs: torch.Tensor | None = None,
                valid: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mask logits -> one direction per frame, in degrees.

    `argmax` is the brightest pixel, quantized to the grid. `centroid` is the paper's rule:
    threshold at 0.5, erode 3x3, fall back to the unthresholded mask if that empties it, then
    average unit vectors (not azimuth degrees, which wrap). `dirs` gives one unit vector per
    cell for a non-rectangular layout (the disc in `src.polar`); `valid` marks its dead cells.
    """
    p = torch.sigmoid(logits)
    B, n_el, n_az = p.shape
    wrap = dirs is None
    if mode == "argmax":
        # The disc's dead cells carry the zenith as a placeholder direction, so an unmasked
        # argmax can answer "straight up" for a frame whose peak was actually in a corner. The
        # target is zero there and a trained model learns to keep them dark, but nothing
        # enforces it, and -1 is below any sigmoid.
        if valid is not None:
            p = p.masked_fill(~valid.reshape(1, n_el, n_az), -1.0)
        k = p.flatten(1).argmax(1)
        if wrap:
            return AZ[(k % n_az).cpu().numpy()], EL[(k // n_az).cpu().numpy()]
        return to_angles(dirs[k].cpu().numpy())

    binary = (p > threshold).float()
    if valid is not None:
        binary = binary * valid.reshape(1, n_el, n_az).float()
    # Erosion = every pixel in the 3x3 neighbourhood is active. On the rectangular grid azimuth
    # wraps and elevation does not, so the two axes are padded differently, exactly as the
    # convolutions do. On the disc there is no seam to preserve and no axis wraps.
    padded = (F.pad(binary.unsqueeze(1), (1, 1, 0, 0), mode="circular") if wrap
              else F.pad(binary.unsqueeze(1), (1, 1, 0, 0), mode="constant", value=0.0))
    padded = F.pad(padded, (0, 0, 1, 1), mode="constant", value=0.0)
    eroded = (-F.max_pool2d(-padded, 3, stride=1)).squeeze(1)
    weights = torch.where(eroded.flatten(1).sum(1, keepdim=True) > 0,
                          eroded.flatten(1), binary.flatten(1))
    # If nothing survives the threshold either, fall back to the peak so every frame answers.
    empty = weights.sum(1) == 0
    if empty.any():
        weights = weights.clone()
        weights[empty] = F.one_hot(p.flatten(1)[empty].argmax(1), n_el * n_az).float()

    v = torch.from_numpy(directions().astype(np.float32)).to(p.device) if wrap else dirs
    mean = (weights @ v) / weights.sum(1, keepdim=True).clamp(min=1e-9)
    mean = mean / mean.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return to_angles(mean.cpu().numpy())


class Conv(nn.Module):
    """3x3 convolution, batch norm, ReLU, wrapping in azimuth and zero-padding in elevation.

    `wrap=False` zero-pads both axes, which is what the polar disc needs: its horizontal axis
    is a Cartesian coordinate of the image, not an angle, so wrapping it would join the disc's
    left rim to its right rim — two directions 180 degrees apart.
    """

    def __init__(self, cin: int, cout: int, wrap: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(cout)
        self.wrap = wrap

    def forward(self, x):
        x = (F.pad(x, (1, 1, 0, 0), mode="circular") if self.wrap
             else F.pad(x, (1, 1, 0, 0), mode="constant", value=0.0))
        x = F.pad(x, (0, 0, 1, 1), mode="constant", value=0.0)
        return F.relu(self.norm(self.conv(x)), inplace=True)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, wrap: bool = True):
        super().__init__()
        self.a, self.b = Conv(cin, cout, wrap), Conv(cout, cout, wrap)

    def forward(self, x):
        return self.b(self.a(x))


class UNet(nn.Module):
    """(B, C, H, W) maps -> (B, H, W) per-pixel logits, plus one presence logit per map.

    H x W is the 23 x 91 rectangle, or the 46 x 46 disc with `wrap=False`.
    """

    def __init__(self, in_channels: int, base: int = 32, depth: int = 3,
                 dropout: float = 0.1, wrap: bool = True):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        c = in_channels
        for ch in chans:
            self.down.append(Block(c, ch, wrap))
            c = ch
        self.bottleneck = Block(c, c * 2, wrap)
        self.up = nn.ModuleList()
        c = c * 2
        for ch in reversed(chans):
            self.up.append(Block(c + ch, ch, wrap))
            c = ch
        self.drop = nn.Dropout2d(dropout)
        self.mask_head = nn.Conv2d(c, 1, 1)
        # Mean and max pooling: the max keeps the single sharp peak that separates a drone from
        # diffuse wind. LeakyReLU keeps a gradient for negative inputs, so the head cannot die
        # into a constant output.
        self.presence_head = nn.Sequential(nn.Flatten(), nn.Linear(chans[-1] * 4, 128),
                                           nn.LeakyReLU(0.01, inplace=True),
                                           nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        pooled = torch.cat([F.adaptive_avg_pool2d(x, 1), F.adaptive_max_pool2d(x, 1)], 1)
        presence = self.presence_head(pooled).squeeze(-1)
        for block, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat([x, skip], 1))
        return self.mask_head(self.drop(x)).squeeze(1), presence

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def tversky_loss(logits, target, alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0):
    """The paper's loss. Beta > alpha makes a missed drone cell cost more than a false one;
    alpha = beta = 0.5 is the Dice loss."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum((-2, -1))
    fp = (p * (1 - target)).sum((-2, -1))
    fn = ((1 - p) * target).sum((-2, -1))
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


class Loss(nn.Module):
    """Per-pixel localization loss plus presence classification, weighted equally.

    `kind='wce'` (default) is class-weighted cross-entropy: about 1.4% of cells are positive,
    so unweighted it is minimized by predicting empty sky everywhere. `kind='tversky'` is the
    paper's loss, run as its own experiment.
    """

    def __init__(self, kind: str = "wce", pos_weight: float = 40.0,
                 presence_weight: float = 1.0, presence_pos_weight: float = 1.0):
        super().__init__()
        self.kind = kind
        self.register_buffer("pw", torch.tensor(float(pos_weight)))
        # Most training frames contain a drone; the only negatives are the ambient tails and the
        # pre-takeoff windows. Unweighted, the presence head is minimized by always answering yes.
        self.register_buffer("ppw", torch.tensor(float(presence_pos_weight)))
        self.presence_weight = presence_weight

    def forward(self, mask_logits, presence_logit, mask, present):
        if self.kind == "wce":
            loc = F.binary_cross_entropy_with_logits(mask_logits, mask, pos_weight=self.pw)
        elif self.kind == "tversky":
            loc = tversky_loss(mask_logits, mask)
        else:
            raise ValueError(f"unknown loss {self.kind}")
        det = F.binary_cross_entropy_with_logits(presence_logit, present, pos_weight=self.ppw)
        return (loc + self.presence_weight * det, loc.detach().item(), det.detach().item())
