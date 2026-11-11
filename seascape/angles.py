"""Compass angles in degrees: a bearing clockwise from north, an elevation up from
the horizon.
"""

import math

import numpy as np
import numpy.typing as npt


def wrapped_deg(angle_deg: float) -> float:
    """The same direction in [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def bearing_deg(east: float, north: float) -> float:
    """The compass bearing towards a point `east` and `north`, in (-180, 180]."""
    return math.degrees(math.atan2(east, north))


def direction(bearing_deg: npt.ArrayLike, elevation_deg: npt.ArrayLike) -> np.ndarray:
    """Unit (east, north, up) towards a compass bearing and elevation, on the last
    axis. A direction, so the bearing is not negated.
    """
    b, e = np.radians(bearing_deg), np.radians(elevation_deg)
    return np.stack(
        np.broadcast_arrays(np.cos(e) * np.sin(b), np.cos(e) * np.cos(b), np.sin(e)), -1
    )
