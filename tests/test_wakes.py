import math

import pytest

from seascape import wakes


def _hull(froude: float, length_m: float = 20.0) -> wakes.Wake:
    speed_mps = froude * math.sqrt(9.81 * length_m)
    return wakes.Wake(speed_mps=speed_mps, length_m=length_m, beam_m=length_m / 4)


def test_kelvin_s_wedge_holds_until_the_wake_narrows() -> None:
    assert _hull(0.3).peak_angle_rad == wakes.KELVIN_HALF_ANGLE_RAD
    assert math.degrees(wakes.KELVIN_HALF_ANGLE_RAD) == pytest.approx(19.47, abs=0.01)
    # Continuous through the narrowing.
    just_past = _hull(wakes.NARROWING_FROUDE * 1.0001).peak_angle_rad
    assert just_past == pytest.approx(wakes.KELVIN_HALF_ANGLE_RAD, rel=1e-3)


def test_a_fast_hull_s_wake_narrows_as_one_over_froude() -> None:
    """Rabaud & Moisy eq. 4, alpha ~ 1 / (2 sqrt(2 pi) Fr), the large-Froude limit."""
    for froude in (3.0, 6.0):
        limit = 1 / (2 * math.sqrt(2 * math.pi) * froude)
        assert _hull(froude).peak_angle_rad == pytest.approx(limit, rel=0.03)


def test_no_waves_below_kriebel_and_seelig_s_threshold() -> None:
    assert _hull(wakes.WAVELESS_FROUDE).height_scale_m == 0.0
    assert _hull(0.3).height_scale_m > 0.0


def test_a_planing_hull_s_waves_stop_growing_past_the_hump() -> None:
    hump = _hull(wakes.HUMP_FROUDE).height_scale_m
    assert _hull(0.4).height_scale_m < hump
    assert _hull(1.2).height_scale_m == pytest.approx(hump)


def test_kriebel_and_seelig_s_height() -> None:
    """beta at L / Le = 3, against their eq. at F = 0.3, y = L."""
    hull = _hull(0.3, length_m=100.0)
    beta = 1 + 8 * math.tanh(0.45) ** 3
    expected = beta * hull.speed_mps**2 / 9.81 * 0.2**2
    assert hull.height_scale_m == pytest.approx(expected, rel=1e-3)


def test_no_arm_is_steeper_than_michell_s_wave() -> None:
    assert _hull(0.55, length_m=5.0).steepest_arm <= math.pi * wakes.MAX_STEEPNESS
