"""The sky as a disc instead of a rectangle.

The rectangular map has two defects, and we only ever fixed one. Circular padding solved the
azimuth seam. The pole was never addressed: at 88 degrees elevation the 91 azimuth columns
describe directions 0.14 degrees apart, while at the horizon the same columns are 4 degrees
apart. A 3x3 convolution therefore covers a 12 degree patch of sky near the horizon and a 0.4
degree sliver near the zenith, so a filter that has learned what a drone looks like in one place
is looking at something else entirely in the other.

That matters here specifically because our one reproduction gap is beyond 100 m and is almost
entirely in azimuth — 53 degrees against 11 in elevation — and azimuth is the axis the
rectangular layout distorts.

The fix is the projection a fisheye lens makes: look straight up, put the zenith at the centre
of a square image and the horizon on the inscribed circle. Radius is elevation, angle around the
disc is azimuth:

    r = (90 - elevation) / 90        theta = azimuth

Now a step of one pixel is a step of roughly the same angle everywhere, and there is no seam
anywhere because azimuth wraps around the disc's interior rather than off its edge.

Two costs, both accepted deliberately:

- The corners of the square are outside the unit disc and describe no direction at all. They are
  1 - pi/4 = 21.5% of the pixels, and they are held at zero.
- Azimuth resolution near the zenith is now genuinely coarse rather than fictitiously fine. That
  is the correct trade: near the zenith azimuth barely changes the direction, which is why the
  rectangular grid's precision there was never real.

`N = 46` is not a free choice. The radial axis then holds 23 cells across 90 degrees of
elevation, exactly matching the 4 degree spacing of the rectangular grid, so the comparison
between layouts is not confounded by resolution.

One function defines the projection and everything else derives from it, which is what keeps the
input, the target and the decoder from drifting apart: the network's input is the rectangular
map resampled onto the disc, and the target is the *same spherical cap* rasterized directly onto
the disc's own direction vectors — never a resampled copy of the rectangular target, which would
blur a hard label through two interpolations.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import AZ, EL

N = 46
SHAPE = (N, N)
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
    el = np.where(valid, el, 90.0)
    a, e = np.radians(az), np.radians(el)
    dirs = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)
    return dirs.reshape(-1, 3), valid.reshape(-1)


def sampling_grid(n: int = N) -> torch.Tensor:
    """(1, n, n, 2) grid for `F.grid_sample` on a (B, C, len(EL), len(AZ)) map.

    Dead corners map to negative elevation, i.e. outside the source, so zero padding handles the
    far ones. It does not handle the near ones: a cell at radius 1.02 sits four tenths of a cell
    past the horizon row, and bilinear interpolation there returns most of that row's value
    rather than zero. The validity mask in `Projection` is what actually clears them, and the
    gate in `utils/check_polar.py` exists because this was wrong the first time.
    """
    az, el, _ = cell_angles(n)
    x = (az - AZ[0]) / (AZ[-1] - AZ[0]) * 2.0 - 1.0
    y = (el - EL[0]) / (EL[-1] - EL[0]) * 2.0 - 1.0
    return torch.from_numpy(np.stack([x, y], -1).astype(np.float32)).unsqueeze(0)


class Projection:
    """Cached tensors for one device, since the grid is fixed for the whole run."""

    def __init__(self, n: int = N, device=None):
        self.n = n
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
    """The disc without the resampling step: steer the beamformer at the disc's own cells.

    `Projection` builds the disc from the finished rectangular map, so its input is a bilinear
    interpolation of cells 4 degrees apart in azimuth painted onto cells that are 2.5 degrees
    apart at the rim — the map is smoothed on the way in while the target stays a hard cap
    rasterized at full disc precision. A far drone is a low-contrast peak, which is exactly what
    that smoothing costs the most, and the far bin is where the resampled disc lost: 82 degrees
    of mean azimuth error against the rectangle's 57.

    Steering natively removes the interpolation entirely. `cell_directions` already gives one
    unit vector per cell and `beamform.Beamformer` already accepts an arbitrary direction set,
    so the beamformer returns an (n, n) disc directly and this class only has to clear the dead
    corners. Cost is unchanged: 2116 disc directions against the rectangle's 2093.

    Interchangeable with `Projection` from the caller's side — same `dirs`, `valid`, `keep`, and
    the same one-argument call — so it drops into the `projection=` slot of `data.to_maps`.
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

    Rasterized from the cap rather than resampled from the rectangular mask. Dead cells can
    never be positive because their placeholder direction is the zenith, which is inside the cap
    only when the drone is overhead — so they are masked out explicitly.
    """
    dirs, valid = cell_directions(n)
    az, el = np.asarray(az, float), np.asarray(el, float)
    out = np.zeros((len(az), n, n), dtype=np.uint8)
    for i in range(0, len(az), chunk):
        a, e = az[i:i + chunk], el[i:i + chunk]
        ok = ~np.isnan(a)
        if not ok.any():
            continue
        src = np.stack([np.cos(np.radians(e)) * np.cos(np.radians(a)),
                        np.cos(np.radians(e)) * np.sin(np.radians(a)),
                        np.sin(np.radians(e))], -1)
        src[~ok] = 0.0
        ang = np.degrees(np.arccos(np.clip(dirs @ src.T, -1.0, 1.0)))          # (n*n, b)
        m = (ang.T <= radius_deg) & valid[None, :]
        m[~ok] = False
        out[i:i + chunk] = m.reshape((-1, n, n))
    return out
