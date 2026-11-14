"""The earth under the sea: its pull and its horizon."""

import math

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


def sea_run_m(
    origin_m: tuple[float, float, float],
    toward: tuple[float, float, float],
    radius_m: float,
) -> float | None:
    """The horizontal distance a ray from `origin_m` runs before it meets the sea
    `sea_z_m` defines, or None if it passes over the horizon or starts under the sea.

    Args:
        origin_m: World east, north and up of the ray's start.
        toward: East, north and up of the ray's direction.
        radius_m: The effective radius, `earth_radius_m`.
    """
    east, north, up = origin_m
    flat = math.hypot(toward[0], toward[1])
    if flat == 0.0:
        return 0.0 if toward[2] < 0.0 else None
    along_e, along_n = toward[0] / flat, toward[1] / flat
    # Height over the sea at a run s: c + b s + s^2 / (2 R), a root where it meets.
    b = toward[2] / flat + (east * along_e + north * along_n) / radius_m
    c = up - sea_z_m(east, north, radius_m)
    disc = b * b - 2.0 * c / radius_m
    if c < 0.0 or b >= 0.0 or disc < 0.0:
        return None
    # The nearer root, in the form that keeps its digits at a small curvature.
    return 2.0 * c / (-b + math.sqrt(disc))


def horizon_m(height_m: float, refraction_k: float) -> float:
    """Distance to the horizon from `height_m`, tangent to the effective sphere."""
    return math.sqrt(2.0 * earth_radius_m(refraction_k) * height_m)
