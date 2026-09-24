"""Per-frame ground truth from the GPS logs.

Ported from utils/validate_dataset.py, generalized to every segment. The drone's position
comes from its own flight log, projected into UTM and taken relative to the array, whose
position comes from a separate stationary reference log. The array's compass heading was
never recorded, so each session ships one hand-measured `initial_azimuth_deg` and every
azimuth is an offset from it.

GPS rows are 10 Hz and audio frames are 100 ms, so row k of the log is frame k of the
segment. That identity is the whole alignment scheme; it is asserted, not assumed.
"""
import numpy as np
import pandas as pd
from pyproj import Transformer

from .config import CSV_HZ, FEET_TO_M, MPH_TO_MPS, Segment, UTM_EPSG
from .geometry import angular_distance, directions

_FLIGHT_COLS = ["latitude", "longitude", "altitude_above_seaLevel(feet)",
                "time(millisecond)", " xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"]


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _array_origin(session_dir, align) -> tuple[float, float, float, Transformer]:
    """Array position: mean of the stationary reference log, in UTM metres plus altitude."""
    ref = _read((session_dir / align["ref_csv"]).resolve())
    tr = Transformer.from_crs("epsg:4326", UTM_EPSG, always_xy=True)
    rx, ry = tr.transform(ref["longitude"].dropna().astype(float).mean(),
                          ref["latitude"].dropna().astype(float).mean())
    alt0 = float(ref["altitude_above_seaLevel(feet)"].dropna().iloc[0]) * FEET_TO_M
    return rx, ry, alt0, tr


def frame_labels(seg: Segment, n_frames: int | None = None) -> dict:
    """Per-frame labels for one segment.

    Returns az, el, range_m, speed_mps (NaN wherever no drone is present) and has_drone,
    each of length n_frames. For no-drone segments the length comes from the declared
    segment bounds; for drone segments it comes from the flight log, which is what limits
    how much audio is usable.
    """
    if not seg.has_drone:
        n = int(round((seg.skip_seconds_end - seg.skip_seconds) * CSV_HZ))
        n = n if n_frames is None else min(n, n_frames)
        nan = np.full(n, np.nan, dtype=np.float32)
        return {"az": nan, "el": nan.copy(), "range_m": nan.copy(),
                "speed_mps": nan.copy(),
                "has_drone": np.zeros(n, dtype=np.uint8)}

    rx, ry, alt0, tr = _array_origin(seg.session_dir, seg.align)
    fl = (_read(seg.session_dir / seg.flight_csv)
          .iloc[seg.start_index_csv:]
          .reset_index(drop=True)[_FLIGHT_COLS]
          .dropna())

    x, y = tr.transform(fl["longitude"].to_numpy(float), fl["latitude"].to_numpy(float))
    bearing = np.degrees(np.arctan2(y - ry, x - rx))
    az_off = seg.initial_azimuth_deg - bearing[0]
    az = -(((bearing + az_off + 180) % 360) - 180)

    horiz = np.hypot(x - rx, y - ry)
    rel_alt = fl["altitude_above_seaLevel(feet)"].to_numpy(float) * FEET_TO_M - alt0
    el = np.degrees(np.arctan2(rel_alt, horiz))
    rng = np.hypot(horiz, rel_alt)
    speed = np.linalg.norm(
        fl[[" xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"]].to_numpy(float),
        axis=1) * MPH_TO_MPS

    n = len(az) if n_frames is None else min(len(az), n_frames)
    if seg.skip_seconds_end is not None:
        n = min(n, int(round((seg.skip_seconds_end - seg.skip_seconds) * CSV_HZ)))
    return {"az": az[:n].astype(np.float32), "el": el[:n].astype(np.float32),
            "range_m": rng[:n].astype(np.float32),
            "speed_mps": speed[:n].astype(np.float32),
            "has_drone": np.ones(n, dtype=np.uint8)}


def propagation_lagged(meta: dict, iterations: int = 3) -> dict:
    """Labels moved to where the drone was when the sound in each frame left it.

    Row k of the GPS log is where the drone *is* at frame k, but the sound heard in frame k was
    emitted range/c earlier: 0.29 s, about three frames, at 100 m. So each drone frame's label
    is replaced by the trajectory interpolated at emission time k - r/(c dt), in Cartesian
    coordinates so azimuth never wraps mid-interpolation. The emission range depends on the
    emission time, so the delay is iterated to a fixed point; three rounds settle it well below
    a millisecond at these speeds. Emission times before the segment's first row clamp to it.
    No-drone frames are returned unchanged.
    """
    from .config import C

    out = dict(meta)
    d = meta["has_drone"].astype(bool)
    if not d.any():
        return out
    az, el, r = (np.radians(meta["az"][d].astype(float)), np.radians(meta["el"][d].astype(float)),
                 meta["range_m"][d].astype(float))
    xyz = np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], 1)
    k = np.arange(len(r), dtype=float)
    lag = r / (C / CSV_HZ)
    for _ in range(iterations):
        t = np.clip(k - lag, 0.0, None)
        p = np.stack([np.interp(t, k, xyz[:, i]) for i in range(3)], 1)
        lag = np.linalg.norm(p, axis=1) / (C / CSV_HZ)
    rng = np.linalg.norm(p, axis=1)
    lagged = {"az": np.degrees(np.arctan2(p[:, 1], p[:, 0])),
              "el": np.degrees(np.arcsin(np.clip(p[:, 2] / np.maximum(rng, 1e-9), -1, 1))),
              "range_m": rng}
    for key, v in lagged.items():
        col = np.array(meta[key], copy=True)
        col[d] = v.astype(col.dtype)
        out[key] = col
    return out


def target_mask(az, el, radius_deg: float, chunk: int = 2048, soft: bool = False) -> np.ndarray:
    """(n, 23, 91) uint8: 1 within radius_deg of the labelled direction.

    A disc on the sphere rather than a rectangle on the grid, so the target does not
    stretch as elevation rises and azimuth cells crowd together. Frames with no drone get
    an all-zero mask.

    `soft` gives float32 instead (float16 rounds pixels just outside the radius up to exactly
    0.5, so the disc would no longer be recoverable): a Gaussian in angular distance scaled to be exactly 0.5 at
    `radius_deg`, so the centroid decoder's 0.5 threshold on a perfect prediction lights the
    same disc the hard target draws, while pixels near the edge stop being all-or-nothing.
    """
    from .config import GRID_SHAPE

    dirs = directions()                                          # (D, 3)
    az, el = np.asarray(az, float), np.asarray(el, float)
    out = np.zeros((len(az),) + GRID_SHAPE, dtype=np.float32 if soft else np.uint8)
    sigma = radius_deg / np.sqrt(2.0 * np.log(2.0))
    for i in range(0, len(az), chunk):
        a, e = az[i:i + chunk], el[i:i + chunk]
        ok = ~np.isnan(a)
        if not ok.any():
            continue
        src = np.stack([np.cos(np.radians(e)) * np.cos(np.radians(a)),
                        np.cos(np.radians(e)) * np.sin(np.radians(a)),
                        np.sin(np.radians(e))], axis=-1)          # (b, 3)
        src[~ok] = 0.0
        ang = np.degrees(np.arccos(np.clip(dirs @ src.T, -1.0, 1.0)))   # (D, b)
        if soft:
            m = np.exp(-0.5 * (ang.T / sigma) ** 2).reshape((-1,) + GRID_SHAPE)
        else:
            m = (ang.T <= radius_deg).reshape((-1,) + GRID_SHAPE)
        m[~ok] = 0
        out[i:i + chunk] = m
    return out


def grid_error(peak_az, peak_el, az, el) -> np.ndarray:
    """Angular error in degrees between predicted and true direction, NaN where no drone."""
    return angular_distance(peak_az, peak_el, az, el)
