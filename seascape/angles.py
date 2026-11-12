"""Compass angles in degrees: a bearing clockwise from north, an elevation up from
the horizon.
"""

import math

import numpy as np
import numpy.typing as npt


def wrapped_deg(angle_deg: float) -> float:
    """The same direction in [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def yaw(bearing_deg: float) -> float:
    """Bearing to a rotation about +Z, in radians: the one negation, as +Z turns to
    port.
    """
    return -math.radians(bearing_deg)


def turned(bearing_deg: float, elevation_deg: float) -> np.ndarray:
    """(3, 3) rotation taking +Y to `direction(bearing_deg, elevation_deg)`: a pitch
    about +X, then a turn about +Z, so +X stays level.
    """
    turn, pitch = yaw(bearing_deg), math.radians(elevation_deg)
    ct, st, cp, sp = math.cos(turn), math.sin(turn), math.cos(pitch), math.sin(pitch)
    about_z = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    return about_z @ about_x


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
