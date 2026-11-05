"""A fisheye lens in OpenCV's model: image radius as a polynomial of the field angle.

With theta the angle from the optical axis, theta_d = theta (1 + k1 theta^2 +
k2 theta^4 + k3 theta^6 + k4 theta^8) and the image radius is fx theta_d
(https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html). All k are zero for
an equidistant lens. Blender's polynomial camera takes the inverse, the angle as a
polynomial of the sensor radius.
"""

import math

import numpy as np

type K = tuple[float, float, float, float]

# Numerics: angles the inverse is tabulated at, and radii the Blender polynomial is
# fitted at.
ANGLES = 20001
RADII = 512
# The Blender polynomial may miss the model by this many pixels.
FIT_TOLERANCE_PX = 1.0


def radius_px(theta: np.ndarray, k: K, fx: float) -> np.ndarray:
    """Image radius, in pixels, of the field angle `theta` in radians."""
    return (
        fx
        * theta
        * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8)
    )


def focal_px(hfov_rad: float, width_px: int, k: K) -> float:
    """The `fx` that puts the field angle `hfov_rad / 2` at the image's left and right
    edges."""
    return width_px / 2 / float(radius_px(np.array(hfov_rad / 2), k, 1.0))


def theta(radius: np.ndarray, k: K, fx: float, reach_px: float) -> np.ndarray:
    """Field angle, in radians, of image radii up to `reach_px`; the lens must put each
    radius at one angle, so the model rising out to `reach_px` is checked."""
    angles = np.linspace(0.0, math.pi, ANGLES)
    radii = radius_px(angles, k, fx)
    falling = np.flatnonzero(np.diff(radii) <= 0.0)
    top = int(falling[0]) if falling.size else ANGLES - 1
    if radii[top] < reach_px:
        raise ValueError(
            f"the fisheye model stops rising at {math.degrees(angles[top]):.1f} deg, "
            f"short of the image radius {reach_px:.0f} px"
        )
    return np.interp(radius, radii[: top + 1], angles[: top + 1])


def direction(x_px: np.ndarray, y_px: np.ndarray, k: K, fx: float) -> np.ndarray:
    """Unit rays, (..., 3), through pixels `x_px` right and `y_px` up of the principal
    point, in a camera that looks down -Z with +X right and +Y up, as Blender's does."""
    radius = np.hypot(x_px, y_px)
    angle = theta(radius, k, fx, float(radius.max()))
    safe = np.where(radius > 0.0, radius, 1.0)
    along = np.where(radius > 0.0, np.sin(angle) / safe, 0.0)
    return np.stack([x_px * along, y_px * along, -np.cos(angle)], axis=-1)


def polynomial(
    k: K, fx: float, width_px: int, height_px: int, pitch_mm: float
) -> tuple[float, float, float, float, float]:
    """Blender's `fisheye_polynomial_k0..k4` for a frame of `pitch_mm` pixels: the field
    angle in radians as a polynomial of the sensor radius in mm, fitted out to the
    frame's corner."""
    reach = math.hypot(width_px, height_px) / 2
    radius = np.linspace(0.0, reach, RADII)
    angle = theta(radius, k, fx, reach)
    series = np.polynomial.polynomial
    coefficients = series.polyfit(radius * pitch_mm, angle, 4)
    # The angle's error as the pixels it moves a ray by, where the lens is flattest.
    slope = np.gradient(radius, angle)
    miss = np.abs(series.polyval(radius * pitch_mm, coefficients) - angle) * slope
    if miss.max() > FIT_TOLERANCE_PX:
        raise ValueError(
            f"Blender's fourth-degree polynomial misses the fisheye model by "
            f"{miss.max():.2f} px"
        )
    return tuple(float(c) for c in coefficients)  # ty: ignore[invalid-return-type]
