"""Compass angles."""

import math

import numpy as np
import pytest

from seascape import angles


@pytest.mark.parametrize(
    ("bearing_deg", "elevation_deg", "toward"),
    [
        (0.0, 0.0, (0.0, 1.0, 0.0)),
        (90.0, 0.0, (1.0, 0.0, 0.0)),
        (33.0, 90.0, (0, 0, 1)),
    ],
)
def test_a_direction_points_along_its_compass_bearing(
    bearing_deg: float, elevation_deg: float, toward: tuple[float, float, float]
) -> None:
    got = angles.direction(bearing_deg, elevation_deg)
    np.testing.assert_allclose(got, toward, atol=1e-12)


@pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 135.0, -90.0, 180.0])
def test_a_bearing_reads_back_its_direction(bearing_deg: float) -> None:
    east, north, _ = angles.direction(bearing_deg, 0.0)
    assert angles.bearing_deg(east, north) == pytest.approx(bearing_deg)


@pytest.mark.parametrize(("bearing_deg", "elevation_deg"), [(0.0, 0.0), (30.0, -12.0)])
def test_a_turn_takes_ahead_to_its_direction_and_keeps_starboard_level(
    bearing_deg: float, elevation_deg: float
) -> None:
    turn = angles.turned(bearing_deg, elevation_deg)
    np.testing.assert_allclose(
        turn[:, 1], angles.direction(bearing_deg, elevation_deg), atol=1e-12
    )
    assert turn[2, 0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("pitch_deg", [-10.0, 5.0])
@pytest.mark.parametrize("bearing_deg", [0.0, 40.0])
def test_a_pitch_is_found_back_from_the_ray_it_raised(
    pitch_deg: float, bearing_deg: float
) -> None:
    """Round trip through `turned`: an off-axis ray climbs less than the pitch."""
    ray = angles.direction(bearing_deg, 0.0)
    raised = angles.turned(0.0, pitch_deg) @ ray
    climbed = math.degrees(math.asin(raised[2]))
    assert angles.pitch_to_deg(ray, climbed) == pytest.approx(pitch_deg)


def test_an_elevation_clamps_float_noise_past_the_zenith() -> None:
    assert angles.elevation_deg(1.0 + 1e-12) == pytest.approx(90.0)


def _rolled(roll_deg: float) -> np.ndarray:
    """About +Y, the bow: positive puts starboard down."""
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@pytest.mark.parametrize(
    ("roll_deg", "pitch_deg", "yaw_deg"), [(-8.0, 12.0, 30.0), (20.0, -5.0, -120.0)]
)
def test_an_attitude_reads_back_the_roll_pitch_and_yaw_that_built_it(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> None:
    body = angles.turned(yaw_deg, pitch_deg) @ _rolled(roll_deg)
    assert angles.attitude_deg(body) == pytest.approx((roll_deg, pitch_deg, yaw_deg))


@pytest.mark.parametrize(
    ("angle_deg", "wrapped"),
    [(180.0, 180.0), (-180.0, 180.0), (190.0, -170.0), (-190.0, 170.0)],
)
def test_an_angle_wraps_into_the_range_of_a_bearing(
    angle_deg: float, wrapped: float
) -> None:
    assert angles.wrapped_deg(angle_deg) == pytest.approx(wrapped)


def test_due_south_and_due_north_carry_no_negative_zero() -> None:
    assert angles.bearing_deg(-0.0, -1.0) == 180.0
    assert math.copysign(1.0, angles.bearing_deg(-0.0, 1.0)) == 1.0
    assert math.copysign(1.0, angles.wrapped_deg(0.0)) == 1.0
