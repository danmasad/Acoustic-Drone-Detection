"""The network: an encoder-decoder over the energy map, with two heads.

Hyperparameters are adopted from the reference paper rather than searched (64 base
filters, depth 3, 3x3 kernels), because that is where their compute went and re-searching
per condition would confound the comparison we care about.

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


def decode_mask(logits: torch.Tensor, mode: str = "centroid", threshold: float = 0.5,
                dirs: torch.Tensor | None = None,
                valid: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mask logits -> one direction per frame, in degrees.

    `dirs` supplies one unit vector per map cell and switches the decoder to a layout other
    than the rectangular elevation-by-azimuth grid — the polar disc of `src.polar`. Without it
    the rectangular grid is assumed, and the azimuth axis is treated as circular. `valid` marks
    cells that describe no direction at all, which only the disc has.

    Two decoders, and the choice turns out to matter more than most architecture decisions:

    - `argmax`: the brightest pixel. Simple, but quantized to the 4 degree grid and decided by
      a single pixel, so one bright artefact anywhere in the map wins outright.
    - `centroid`: the paper's post-processing (their section 3) — threshold the sigmoid at 0.5,
      erode with a 3x3 kernel, fall back to the unthresholded mask if erosion empties it, then
      take the centroid of what remains. Averaging over the activated region is robust to
      speckle, and the answer is continuous rather than grid-quantized.

    The centroid is taken over unit vectors rather than over azimuth degrees, because azimuth
    wraps: the mean of 179 and -179 is 180, not 0.
    """
    return decode_probs(torch.sigmoid(logits), mode, threshold, dirs, valid)


def decode_probs(p: torch.Tensor, mode: str = "centroid", threshold: float = 0.5,
                 dirs: torch.Tensor | None = None,
                 valid: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`decode_mask` on per-pixel probabilities rather than logits, which is what an ensemble
    has after averaging its members' sigmoids."""
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
        v = dirs[k].cpu().numpy()
        return (np.degrees(np.arctan2(v[:, 1], v[:, 0])),
                np.degrees(np.arcsin(np.clip(v[:, 2], -1.0, 1.0))))

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

    if wrap:
        az = torch.from_numpy(np.radians(AZ).astype(np.float32)).to(p.device)
        el = torch.from_numpy(np.radians(EL).astype(np.float32)).to(p.device)
        grid_el, grid_az = torch.meshgrid(el, az, indexing="ij")
        v = torch.stack([torch.cos(grid_el) * torch.cos(grid_az),
                         torch.cos(grid_el) * torch.sin(grid_az),
                         torch.sin(grid_el)], -1).reshape(-1, 3)
    else:
        v = dirs
    mean = (weights @ v) / weights.sum(1, keepdim=True).clamp(min=1e-9)
    mean = mean / mean.norm(dim=1, keepdim=True).clamp(min=1e-9)
    m = mean.cpu().numpy()
    return (np.degrees(np.arctan2(m[:, 1], m[:, 0])),
            np.degrees(np.arcsin(np.clip(m[:, 2], -1.0, 1.0))))


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
    """(B, C, 23, 91) maps -> per-pixel logits of the same size, plus one presence logit."""

    def __init__(self, in_channels: int, base: int = 64, depth: int = 3,
                 dropout: float = 0.1, wrap: bool = True, encoder_dropout: float = 0.0):
        super().__init__()
        chans = [base * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList()
        c = in_channels
        for ch in chans:
            self.down.append(Block(c, ch, wrap))
            c = ch
        # Channel dropout after every encoder block. Built only when asked for, so a default
        # model draws the same random numbers as every run trained before the option existed.
        self.enc_drop = nn.Dropout2d(encoder_dropout) if encoder_dropout > 0 else None
        self.bottleneck = Block(c, c * 2, wrap)
        self.up = nn.ModuleList()
        c = c * 2
        for ch in reversed(chans):
            self.up.append(Block(c + ch, ch, wrap))
            c = ch
        self.drop = nn.Dropout2d(dropout)
        self.mask_head = nn.Conv2d(c, 1, 1)
        # Mean *and* max pooling: a drone is a focused peak, and averaging alone throws away
        # exactly the peakiness that distinguishes it from diffuse wind noise.
        #
        # LeakyReLU, not ReLU. At lr 5e-3 the plain-ReLU version of this head died outright in
        # the all24 seed-0 run: every one of the 128 units sat at a negative pre-activation for
        # every input (mean -7.6, max -0.50), so the logit was frozen at the output bias and
        # AUROC read exactly 0.500 from epoch 0. A dead ReLU layer passes zero gradient and can
        # never recover, and a single hidden layer gives it no second path. A small negative
        # slope keeps gradient alive at no cost to the mask branch.
        self.presence_head = nn.Sequential(nn.Flatten(), nn.Linear(chans[-1] * 4, 128),
                                           nn.LeakyReLU(0.01, inplace=True),
                                           nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            if self.enc_drop is not None:
                x = self.enc_drop(x)
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


def tversky_loss(logits, target, alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0,
                 reduce: bool = True):
    """Their loss. Beta > alpha punishes misses harder than false positives, which is what
    a sparse target needs; it is the smooth generalization of the Dice coefficient."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum((-2, -1))
    fp = (p * (1 - target)).sum((-2, -1))
    fn = ((1 - p) * target).sum((-2, -1))
    per_example = 1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return per_example.mean() if reduce else per_example


class Loss(nn.Module):
    """Per-pixel localization loss plus presence classification.

    `kind='wce'` is class-weighted cross-entropy, the default: about 2% of pixels are
    positive, so an unweighted loss is minimized by predicting empty sky everywhere.
    `kind='tversky'` is the paper's choice, kept as the one alternative the step 4 sanity
    check compares.
    """

    def __init__(self, kind: str = "wce", pos_weight: float = 40.0,
                 presence_weight: float = 1.0, presence_pos_weight: float = 1.0):
        super().__init__()
        self.kind = kind
        self.register_buffer("pw", torch.tensor(float(pos_weight)))
        # 92% of training frames contain a drone, because the dataset's only negatives are
        # one 5-minute ambient tail. Unweighted, the presence head is minimized by always
        # answering yes, which is what it did.
        self.register_buffer("ppw", torch.tensor(float(presence_pos_weight)))
        self.presence_weight = presence_weight

    def forward(self, mask_logits, presence_logit, mask, present, weight=None):
        """`weight` is an optional (B,) per-example weight; the loss is then the weighted mean
        of per-example losses. Without it the code path is the original one, operation for
        operation, so default runs reproduce bit for bit."""
        if weight is not None:
            return self._weighted(mask_logits, presence_logit, mask, present, weight)
        if self.kind == "wce":
            loc = F.binary_cross_entropy_with_logits(mask_logits, mask, pos_weight=self.pw)
        elif self.kind == "tversky":
            loc = tversky_loss(mask_logits, mask)
        else:
            raise ValueError(f"unknown loss {self.kind}")
        det = F.binary_cross_entropy_with_logits(presence_logit, present, pos_weight=self.ppw)
        return (loc + self.presence_weight * det, loc.detach().item(), det.detach().item())

    def _weighted(self, mask_logits, presence_logit, mask, present, weight):
        if self.kind == "wce":
            loc = F.binary_cross_entropy_with_logits(mask_logits, mask, pos_weight=self.pw,
                                                     reduction="none").mean((-2, -1))
        elif self.kind == "tversky":
            loc = tversky_loss(mask_logits, mask, reduce=False)
        else:
            raise ValueError(f"unknown loss {self.kind}")
        det = F.binary_cross_entropy_with_logits(presence_logit, present, pos_weight=self.ppw,
                                                 reduction="none")
        w = weight / weight.sum()
        loc, det = (w * loc).sum(), (w * det).sum()
        return (loc + self.presence_weight * det, loc.detach().item(), det.detach().item())
