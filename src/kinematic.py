"""The kinematic filter on decoded directions.

The filter keeps a stored direction and takes a new decode at once when it lies within a
fixed 10° of the stored one. A larger jump is taken only when the last five raw directions
all agree with it. The cap is the same angle at every range; GPS range is never an input to
the decision. It runs after the network, on one segment's frames in time order, and needs no
retraining.
"""
import numpy as np

from .geometry import to_angles, unit_vectors

# One sky-angle cap at every range. Five frames is the half-second a jump must hold.
CAP_DEG = 10.0
K_AGREE = 5


def apply_filter(az: np.ndarray, el: np.ndarray):
    """One segment. The stored direction starts at frame 0 and resets with the caller.

    The cap is a fixed sky angle. Range is not read here.
    """
    vec = unit_vectors(az, el)
    out = np.empty_like(vec)
    stored = vec[0].copy()
    out[0] = stored
    limit = float(np.cos(np.radians(CAP_DEG)))
    for t in range(1, len(vec)):
        new = vec[t]
        if float(np.dot(stored, new)) >= limit:
            stored = new.copy()
        else:
            start = t - K_AGREE + 1
            if start >= 0 and bool(np.all(vec[start:t + 1] @ new >= limit)):
                stored = new.copy()
        out[t] = stored
    return to_angles(out)
