"""Compass angles in degrees, a bearing clockwise from north and an elevation up from
the horizon, and the rotations and frames they turn.
"""

import math
from typing import overload

import numpy as np
import numpy.typing as npt

# A +X right, +Y ahead, +Z up frame's axes as OpenCV's: +X right, +Y down, +Z ahead.
TO_CV = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


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


@overload
def bearing_deg(east: float, north: float) -> float: ...
@overload
def bearing_deg(east: np.ndarray, north: np.ndarray) -> np.ndarray: ...
def bearing_deg(
    east: float | np.ndarray, north: float | np.ndarray
) -> float | np.ndarray:
    """The compass bearing towards points `east` and `north`, in (-180, 180]."""
    bearing = np.degrees(np.arctan2(east, north))
    return float(bearing) if np.ndim(bearing) == 0 else bearing


def elevation_deg(up: npt.ArrayLike) -> np.ndarray:
    """The elevation of unit rays whose up component is `up`; float noise past 1 is
    clamped.
    """
    return np.degrees(np.arcsin(np.clip(up, -1.0, 1.0)))


def direction(bearing_deg: npt.ArrayLike, elevation_deg: npt.ArrayLike) -> np.ndarray:
    """Unit (east, north, up) towards a compass bearing and elevation, on the last
    axis. A direction, so the bearing is not negated.
    """
    b, e = np.radians(bearing_deg), np.radians(elevation_deg)
    return np.stack(
        np.broadcast_arrays(np.cos(e) * np.sin(b), np.cos(e) * np.cos(b), np.sin(e)), -1
    )


def attitude_deg(R_frame_body: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw of a +X starboard, +Y bow, +Z up body, yaw then pitch then
    roll: roll positive starboard down, pitch bow up, yaw clockwise in (-180, 180].
    """
    x, y, z = R_frame_body[:3, :3].T
    roll = math.atan2(-x[2], z[2])
    return (
        math.degrees(roll),
        float(elevation_deg(y[2])),
        bearing_deg(float(y[0]), float(y[1])),
    )


def pitch_to_deg(ray: np.ndarray, elevation_deg: float) -> float:
    """The pitch about +X, as `turned` applies it, that raises `ray`, a unit (3,), to
    `elevation_deg`: the pitch turns its (y, z) in their plane, so an off-axis ray
    climbs less than the pitch.

    Raises:
        ValueError: No pitch reaches it, or the ray faces aft of the beam.
    """
    _, y, z = ray
    # Past the beam a pitch lowers the ray, and asin's branch no longer holds.
    if y <= 0.0:
        raise ValueError("a ray facing aft of the beam has no pitch")
    sine = math.sin(math.radians(elevation_deg)) / math.hypot(y, z)
    if abs(sine) > 1.0:
        raise ValueError(f"no rig pitch raises a ray to {elevation_deg:.1f} deg")
    return math.degrees(math.asin(sine) - math.atan2(z, y))
