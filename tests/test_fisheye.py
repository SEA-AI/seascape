"""A fisheye lens in OpenCV's model, and the polynomial Blender takes for it."""

import math

import numpy as np
import pytest

from seascape.physics import fisheye

K = (-0.0793, -0.0461, 0.0331, -0.0065)


def test_an_equidistant_lens_puts_the_radius_at_fx_times_the_angle() -> None:
    theta = np.linspace(0.0, 1.2, 7)
    assert fisheye.radius_px(theta, (0.0, 0.0, 0.0, 0.0), 900.0) == pytest.approx(
        900.0 * theta
    )


def test_the_focal_length_puts_half_the_field_at_the_image_edge() -> None:
    hfov = math.radians(130.0)
    fx = fisheye.focal_px(hfov, 3840, K)
    assert fisheye.radius_px(np.array(hfov / 2), K, fx) == pytest.approx(1920.0)


def test_the_angle_of_a_radius_undoes_the_radius_of_an_angle() -> None:
    theta = np.linspace(0.0, 1.2, 25)
    radius = fisheye.radius_px(theta, K, 1900.0)
    assert fisheye.theta(radius, K, 1900.0, radius.max()) == pytest.approx(
        theta, abs=1e-6
    )


def test_a_pixel_off_the_axis_looks_that_far_off_it() -> None:
    fx, theta = 1900.0, math.radians(40.0)
    x = fisheye.radius_px(np.array([0.0, theta]), K, fx)
    ray = fisheye.direction(x, np.zeros(2), K, fx)
    assert ray[0] == pytest.approx((0.0, 0.0, -1.0))
    assert ray[1] == pytest.approx((math.sin(theta), 0.0, -math.cos(theta)))


@pytest.mark.parametrize("size", [(3840, 2160), (3600, 1080), (640, 360)])
def test_blenders_polynomial_follows_the_model_within_a_pixel(
    size: tuple[int, int],
) -> None:
    width, height = size
    fx = fisheye.focal_px(math.radians(130.0), width, K)
    pitch_mm = 0.002
    coefficients = fisheye.polynomial(K, fx, width, height, pitch_mm)
    radius = np.linspace(0.0, math.hypot(width, height) / 2, 64)
    theta = np.polynomial.polynomial.polyval(radius * pitch_mm, coefficients)
    assert fisheye.radius_px(theta, K, fx) == pytest.approx(radius, abs=1.0)


def test_a_lens_that_stops_rising_short_of_the_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="stops rising"):
        fisheye.polynomial((-1.0, 0.0, 0.0, 0.0), 1000.0, 3840, 2160, 0.002)
