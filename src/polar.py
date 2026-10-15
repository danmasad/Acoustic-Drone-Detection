"""The sky as a disc instead of a rectangle.

Zenith at the centre of an N x N image, horizon on the inscribed circle:

    r = (90 - elevation) / 90        theta = azimuth

One pixel is then roughly the same angle everywhere, and azimuth has no seam. The corners
(21.5% of pixels) describe no direction and are held at zero. N = 46 puts 23 radial cells
across 90 degrees, matching the rectangular grid's 4 degree spacing. The target is the same
spherical cap rasterized onto the disc's own direction vectors, never a resampled rectangle.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import AZ, EL
from .geometry import unit_vectors

N = 46
DEAD_FRACTION = 1.0 - np.pi / 4


def cell_angles(n: int = N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(azimuth, elevation, valid) for every cell of the n x n disc image, in degrees.

    Cell centres, not edges, so no cell straddles the zenith. Elevation goes negative outside
    the inscribed circle, which is what marks the dead corners: an impossible direction.
    """
    t = (2.0 * np.arange(n) + 1.0) / n - 1.0          # cell centres in [-1, 1]
    v, u = np.meshgrid(t, t, indexing="ij")
    r = np.hypot(u, v)
    return np.degrees(np.arctan2(v, u)), 90.0 * (1.0 - r), r <= 1.0


def cell_directions(n: int = N) -> tuple[np.ndarray, np.ndarray]:
    """((n*n, 3) unit vectors, (n*n,) validity). Dead cells get the zenith as a placeholder."""
    az, el, valid = cell_angles(n)
    dirs = unit_vectors(az, np.where(valid, el, 90.0))
    return dirs.reshape(-1, 3), valid.reshape(-1)


def sampling_grid(n: int = N) -> torch.Tensor:
    """(1, n, n, 2) grid for `F.grid_sample` on a (B, C, len(EL), len(AZ)) map.

    Each disc cell points at its (azimuth, elevation) on the rectangle. Dead corners fall below
    the horizon; the ones just past the rim still interpolate energy from the horizon row, so
    `Projection` clears them with the validity mask.
    """
    az, el, _ = cell_angles(n)
    x = (az - AZ[0]) / (AZ[-1] - AZ[0]) * 2.0 - 1.0
    y = (el - EL[0]) / (EL[-1] - EL[0]) * 2.0 - 1.0
    return torch.from_numpy(np.stack([x, y], -1).astype(np.float32)).unsqueeze(0)


class Projection:
    """Cached tensors for one device, since the grid is fixed for the whole run."""

    def __init__(self, n: int = N, device=None):
        self.grid = sampling_grid(n).to(device)
        dirs, valid = cell_directions(n)
        self.dirs = torch.from_numpy(dirs.astype(np.float32)).to(device)
        self.valid = torch.from_numpy(valid).to(device)
        self.keep = self.valid.reshape(1, 1, n, n).float()

    def __call__(self, maps: torch.Tensor) -> torch.Tensor:
        """(B, C, 23, 91) -> (B, C, n, n). Bilinear, because the map is a smooth energy field.

        Zero is the right fill for the dead corners because the maps arriving here are already
        standardized, so zero is the mean of the input distribution rather than an artificial
        dark spot the first convolution would treat as an edge.
        """
        grid = self.grid.expand(maps.shape[0], -1, -1, -1)
        out = F.grid_sample(maps, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=True)
        return out * self.keep.to(out.dtype)


class NativeDisc:
    """The disc steered directly: the beamformer is aimed at the disc's own cells.

    The map is sharp at disc resolution, with no interpolation from the rectangle. It has the
    same `dirs`, `valid`, `keep` and call as `Projection`, so it fits the `projection=` slot of
    `data.to_maps`. Cost is 2116 directions against the rectangle's 2093.
    """

    def __init__(self, n: int = N, device=None):
        self.n = n
        dirs, valid = cell_directions(n)
        # float32 for the model, float64 for the steering tensor: `Beamformer` computes phase
        # from these, and rounding a direction before computing a delay is a wavelength-scale
        # error at 4 kHz.
        self.steering_dirs = dirs
        self.dirs = torch.from_numpy(dirs.astype(np.float32)).to(device)
        self.valid = torch.from_numpy(valid).to(device)
        self.keep = self.valid.reshape(1, 1, n, n).float()

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def __call__(self, maps: torch.Tensor) -> torch.Tensor:
        """(B, C, n, n) -> the same, with the dead corners zeroed.

        The corners carry the zenith placeholder direction, so without this they would repeat
        one real direction's energy across 21.5% of the image.
        """
        return maps * self.keep.to(maps.dtype)


def target_mask(az, el, radius_deg: float, n: int = N, chunk: int = 2048) -> np.ndarray:
    """(b, n, n) uint8: the same spherical cap as `labels.target_mask`, on the disc's cells.

    Drawn from the cap on the disc's own directions. Dead cells carry the zenith as a placeholder,
    which falls inside the cap whenever the drone is overhead, so they are masked out.
    """
    dirs, valid = cell_directions(n)
    az, el = np.asarray(az, float), np.asarray(el, float)
    out = np.zeros((len(az), n, n), dtype=np.uint8)
    for i in range(0, len(az), chunk):
        a, e = az[i:i + chunk], el[i:i + chunk]
        ok = ~np.isnan(a)
        if not ok.any():
            continue
        src = unit_vectors(a, e)
        src[~ok] = 0.0
        ang = np.degrees(np.arccos(np.clip(dirs @ src.T, -1.0, 1.0)))          # (n*n, b)
        m = (ang.T <= radius_deg) & valid[None, :]
        m[~ok] = False
        out[i:i + chunk] = m.reshape((-1, n, n))
    return out
