"""Training-time augmentation, for the `augment` protocol flag of `src.train`.

The model has two site-1 scenes to learn from and is scored on a third, so anything that is
constant within a scene and different between them — a recorder's gain, a dead channel, the
ambient noise floor, the bearing the drone happened to fly on — is a feature it can memorize.
Each augmentation breaks one of those:

- **Gain jitter**, per example and microphone, constant over frames and frequency: a
  channel's level offset that is fixed within a session is something the model can key on.
- **Channel dropout**: one microphone zeroed, so no single channel becomes load-bearing.
- **Additive noise**: spatially white complex noise at a random level below the example's
  own power, a different ambient floor per example.
- **Azimuth roll**: the maps and target rotated together about the vertical. The array is not
  rotationally symmetric, so a rolled map is not one the array could have produced — its
  sidelobes rotate with the source — but the bearing distribution of two scenes is narrow
  enough that the network can otherwise learn "drones are north-east".

Random numbers come from a CPU generator the caller owns, so turning augmentation on does not
move the draws of the shuffling or dropout generators that a default run uses.
"""
import torch

GAIN_DB = 1.0             # standard deviation
DROP_P = 0.1              # probability an example loses one microphone
NOISE_P = 0.5             # probability an example gets additive noise
NOISE_DB = (-20.0, -10.0)  # noise power relative to the example's mean power
ROLL = True


def spectra(X: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """(B, K, M, F) complex spectra -> the same with gain jitter, dropout and noise."""
    B, K, M, F = X.shape
    dev = X.device
    gain_db = torch.randn(B, 1, M, 1, generator=gen) * GAIN_DB
    g = (10.0 ** (gain_db / 20.0)).to(dev)
    drop = torch.rand(B, generator=gen) < DROP_P
    ch = torch.randint(0, M, (B,), generator=gen)
    keep = torch.ones(B, 1, M, 1)
    keep[drop, 0, ch[drop], 0] = 0.0
    X = X * (g * keep.to(dev))

    noisy = (torch.rand(B, generator=gen) < NOISE_P).to(dev)
    lo, hi = NOISE_DB
    rel_db = torch.rand(B, generator=gen) * (hi - lo) + lo
    power = (X.real ** 2 + X.imag ** 2).mean((1, 2, 3))
    scale = torch.sqrt(power * (10.0 ** (rel_db.to(dev) / 10.0)) / 2.0) * noisy
    n = torch.complex(torch.randn(B, K, M, F, generator=gen),
                      torch.randn(B, K, M, F, generator=gen)).to(dev)
    return X + n * scale.reshape(B, 1, 1, 1).to(X.real.dtype)


def roll_azimuth(maps: torch.Tensor, mask: torch.Tensor, gen: torch.Generator):
    """Rotate (B, C, n_el, n_az) maps and their (B, n_el, n_az) targets by the same random
    whole number of azimuth cells.

    The grid's first and last columns are the same bearing (-180 and +180), so the roll acts on
    the unique columns and the last is copied back from the first; rolling all 91 would leave a
    duplicated bearing in the middle of the map."""
    if not ROLL:
        return maps, mask
    shifts = torch.randint(0, maps.shape[-1] - 1, (maps.shape[0],), generator=gen).tolist()
    return (torch.stack([roll(m, s) for m, s in zip(maps, shifts)]),
            torch.stack([roll(t, s) for t, s in zip(mask, shifts)]))


def roll(t: torch.Tensor, s: int) -> torch.Tensor:
    """`t` rotated by `s` azimuth cells, i.e. every bearing moved 4 * s degrees east."""
    n = t.shape[-1] - 1
    out = torch.empty_like(t)
    out[..., :n] = torch.roll(t[..., :n], s, dims=-1)
    out[..., n] = out[..., 0]
    return out
