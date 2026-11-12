"""A fisheye lens in OpenCV's model (`cv2.fisheye`), and the polynomial Blender takes.

With theta the angle from the optical axis, theta_d = theta (1 + k1 theta^2 +
k2 theta^4 + k3 theta^6 + k4 theta^8) and the image radius is fx theta_d
(https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html). All k are zero for
an equidistant lens. Blender's polynomial camera takes the inverse, the angle as a
polynomial of the sensor radius.

Limitation: OpenCV takes a ray as its point on the plane one unit ahead, so a lens
sees under 90 deg off its axis.
"""

import math

import cv2
import numpy as np

type K = tuple[float, float, float, float]

# The Blender polynomial may miss the model by this many pixels.
FIT_TOLERANCE_PX = 1.0


def radius_px(theta: np.ndarray, k: K, fx: float) -> np.ndarray:
    """Image radius, in pixels, of a field angle.

    Args:
        theta: Angle from the optical axis, in radians, under 90 deg.
        k: OpenCV's k1..k4, all 0 for an equidistant lens.
        fx: Image radius per radian of the distorted angle, in pixels.
    """
    ahead = np.stack([np.tan(theta), np.zeros_like(theta)], axis=-1)
    matrix = np.diag([fx, fx, 1.0])
    image = cv2.fisheye.distortPoints(ahead.reshape(-1, 1, 2), matrix, np.array(k))
    return image[..., 0].reshape(np.shape(theta))


def focal_px(hfov_rad: float, width_px: int, k: K) -> float:
    """The `fx` that puts half the field angle at the image's left and right edges.

    Args:
        hfov_rad: Field angle across the image width, edge to edge.
        width_px: Image width; its edges sit half of it from the principal point.
        k: OpenCV's k1..k4, all 0 for an equidistant lens.
    """
    return width_px / 2 / float(radius_px(np.array(hfov_rad / 2), k, 1.0))


def theta(radius: np.ndarray, k: K, fx: float) -> np.ndarray:
    """Field angle, in radians, of image radii.

    Args:
        radius: Distances from the principal point, in pixels.
        k: OpenCV's k1..k4, all 0 for an equidistant lens.
        fx: Image radius per radian of the distorted angle, in pixels.
    """
    return np.arccos(direction(radius, np.zeros_like(radius), k, fx)[..., 2])


def direction(x_px: np.ndarray, y_px: np.ndarray, k: K, fx: float) -> np.ndarray:
    """Unit rays, (..., 3), in OpenCV's camera axes: +X right, +Y down, +Z ahead.

    Args:
        x_px: Pixels right of the principal point.
        y_px: Pixels down from the principal point, along the image's rows.
        k: OpenCV's k1..k4, all 0 for an equidistant lens.
        fx: Image radius per radian of the distorted angle, in pixels.
    """
    pixels = np.stack([x_px, y_px], axis=-1).astype(np.float64)
    matrix = np.diag([fx, fx, 1.0])
    # Each ray's point on the plane one unit ahead. The model is radial, so it keeps
    # the pixel's side of each axis.
    ahead = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), matrix, np.array(k))
    rays = np.concatenate(
        [ahead.reshape(pixels.shape), np.ones_like(pixels[..., :1])], axis=-1
    )
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def polynomial(
    k: K, fx: float, width_px: int, height_px: int, pitch_mm: float
) -> tuple[float, float, float, float, float]:
    """Blender's `fisheye_polynomial_k0..k4`: the field angle in radians as a
    polynomial of the sensor radius in mm.

    Raises if the fourth degree misses the model by more than `FIT_TOLERANCE_PX` out
    to the frame's corner.

    Args:
        k: OpenCV's k1..k4, all 0 for an equidistant lens.
        fx: Image radius per radian of the distorted angle, in pixels.
        width_px: Image width; with the height, the corner the fit reaches.
        height_px: Image height.
        pitch_mm: A pixel's width on the sensor. Only the scale of the coefficients
            depends on it, not the angle a pixel sees.
    """
    radius = np.linspace(0.0, math.hypot(width_px, height_px) / 2, 512)
    series = np.polynomial.polynomial
    coefficients = series.polyfit(radius * pitch_mm, theta(radius, k, fx), 4)
    fitted = series.polyval(radius * pitch_mm, coefficients)
    # A lens that folds or reaches 90 deg short of the corner misses here too.
    miss = np.abs(radius_px(fitted, k, fx) - radius).max()
    if miss > FIT_TOLERANCE_PX:
        raise ValueError(
            f"Blender's fourth-degree polynomial misses the fisheye model by "
            f"{miss:.2f} px within the frame"
        )
    return tuple(float(c) for c in coefficients)  # ty: ignore[invalid-return-type]
