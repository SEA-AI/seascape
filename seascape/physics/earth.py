"""The earth under the sea: its pull, its horizon, and where a bearing points."""

import math

import numpy as np
import numpy.typing as npt

# Standard gravity is 9.80665 (3rd CGPM, 1901); the wave models quote 9.81.
GRAVITY_MS2 = 9.81

# Mean radius, IUGG.
EARTH_RADIUS_M = 6_371_000.0


def earth_radius_m(refraction_k: float) -> float:
    """Effective radius, R / (1 - k).

    Surveying's standard refraction treatment: a bent ray over R is straight over R'.
    """
    return EARTH_RADIUS_M / (1.0 - refraction_k)


def sea_z_m(east_m: float, north_m: float, radius_m: float) -> float:
    """Height of the sea at a point, relative to the tangent plane at the origin.

    The parabola that osculates the sphere; one definition, so hull and mesh share it.
    """
    return -(east_m * east_m + north_m * north_m) / (2.0 * radius_m)


def horizon_m(height_m: float, refraction_k: float) -> float:
    """Distance to the horizon from `height_m`, tangent to the effective sphere."""
    return math.sqrt(2.0 * earth_radius_m(refraction_k) * height_m)


def direction(bearing_deg: npt.ArrayLike, elevation_deg: npt.ArrayLike) -> np.ndarray:
    """Unit (east, north, up) towards a compass bearing and elevation, on the last
    axis. A direction, so the bearing is not negated.
    """
    b, e = np.radians(bearing_deg), np.radians(elevation_deg)
    return np.stack(
        np.broadcast_arrays(np.cos(e) * np.sin(b), np.cos(e) * np.cos(b), np.sin(e)), -1
    )
