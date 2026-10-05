"""Microphone positions and steering vectors.

Ported from utils/validate/validate_dataset.py. The array was recalibrated between sessions, so
positions are always read from that session's own alignment_params.json and never
hardcoded.
"""
import itertools

import numpy as np

from .config import AZ, C, EL, GRID_SHAPE


def mic_positions(align: dict) -> np.ndarray:
    """(24, 3) metres. Six 3-microphone rings up the tripod legs, then a flat ring of six.

    Accepts either the whole alignment_params dict or just its "mic_array" entry.
    """
    cfg = align.get("mic_array", align)
    a, a2, h, r = cfg["a_"], cfg["a2_"], cfg["h_"], cfg["r_"]
    pos = [[r[ri] * np.cos(np.radians(a[ai])), r[ri] * np.sin(np.radians(a[ai])), h[ri]]
           for ri in range(6) for ai in range(3)]
    pos += [[r[2] * np.cos(np.radians(a2[ai])), r[2] * np.sin(np.radians(a2[ai])), h[2]]
            for ai in range(6)]
    return np.array(pos, dtype=np.float64)


def directions(az_deg=AZ, el_deg=EL) -> np.ndarray:
    """(D, 3) unit vectors, flattened so a reshape to GRID_SHAPE gives (elevation, azimuth)."""
    el, az = np.meshgrid(np.radians(el_deg), np.radians(az_deg), indexing="ij")
    return np.stack([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], axis=-1).reshape(-1, 3)


def steering(pos: np.ndarray, freqs: np.ndarray, dirs: np.ndarray | None = None) -> np.ndarray:
    """(F, D, M) complex64 phase ramps.

    A delay is a frequency-dependent phase rotation, which is why the map has to be formed
    per frequency bin. Working in the frequency domain also gives fractional delays for
    free, so this is more accurate than the integer sample shifts create_dataset.py uses.
    """
    if dirs is None:
        dirs = directions()
    tau = dirs @ pos.T / C                                       # (D, M)
    return np.exp(-2j * np.pi * freqs[:, None, None] * tau[None]).astype(np.complex64)


def energy_map(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Delay-and-sum energy for one frame. X is (F, M), W is (F, D, M) -> (F,) + GRID_SHAPE."""
    Y = np.einsum("fdm,fm->fd", W, X, optimize=True)
    return (np.abs(Y) ** 2).reshape((-1,) + GRID_SHAPE)


def angular_distance(az1, el1, az2, el2) -> np.ndarray:
    """Great-circle angle in degrees between two directions given in degrees."""
    a1, e1, a2, e2 = map(np.radians, (az1, el1, az2, el2))
    dot = (np.cos(e1) * np.cos(e2) * np.cos(a1 - a2) + np.sin(e1) * np.sin(e2))
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def spacings(pos: np.ndarray) -> tuple[float, float]:
    """(minimum spacing, aperture). Minimum spacing sets the spatial aliasing limit,
    aperture sets the beamwidth; thinning an array trades one against the other."""
    d = [np.linalg.norm(pos[i] - pos[j])
         for i, j in itertools.combinations(range(len(pos)), 2)]
    return min(d), max(d)


def coplanar_thickness(pos: np.ndarray) -> float:
    """Spread perpendicular to the best-fit plane. ~0 means elevation is unresolvable."""
    return float(np.linalg.svd(pos - pos.mean(0), compute_uv=False)[-1])


def alias_free_hz(pos: np.ndarray) -> float:
    """Highest frequency with no grating lobes: spacing must stay under half a wavelength."""
    return C / (2 * spacings(pos)[0])
