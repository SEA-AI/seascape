"""A slew from rest to rest. No Blender."""

import pytest

from seascape.physics import slew


def test_a_short_slew_never_reaches_its_rate() -> None:
    # Up and down at 2 deg/s^2 over 2 deg peaks at 2 deg/s, in 2 s.
    assert slew.duration_s(2.0, 6.0, 2.0) == pytest.approx(2.0)
    assert slew.covered_deg(2.0, 6.0, 2.0, 1.0) == pytest.approx(1.0)


def test_an_instant_ramp_slews_at_its_rate() -> None:
    assert slew.duration_s(-12.0, 6.0, None) == pytest.approx(2.0)
    assert slew.covered_deg(-12.0, 6.0, None, 0.5) == pytest.approx(-3.0)
