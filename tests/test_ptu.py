"""The pan-tilt unit's tilt loop."""

import math

import numpy as np
import pytest

from seascape.physics import ptu


@pytest.mark.parametrize("target_deg", [0.0, -2.5])
def test_a_swell_leaves_the_first_order_high_pass_of_the_pitch_from_the_start(
    target_deg: float,
) -> None:
    omega = 2.0 * math.pi / 4.0
    times = np.arange(0.0, 8.0, 1.0 / 30.0)
    tilt = ptu.tilt_deg(lambda t: 4.0 * math.sin(omega * t), list(times), target_deg)

    off = 4.0 * np.sin(omega * times) + tilt - target_deg
    # |H| of e' = -K e + p' at omega: omega / sqrt(omega^2 + K^2).
    gain = omega / math.hypot(omega, ptu.GAIN_PER_S)
    assert np.abs(off).max() == pytest.approx(4.0 * gain, rel=0.05)


def test_the_head_waits_until_its_error_asks_for_more_than_the_dead_band() -> None:
    times = np.arange(0.0, 10.0, 0.1)
    tilt = ptu.tilt_deg(lambda t: 0.05 * t, list(times))

    off = 0.05 * times + tilt
    # It moves once K times the error reaches the dead band, and then keeps up.
    waits_deg = ptu.DEAD_BAND_DEG_S / ptu.GAIN_PER_S
    assert np.abs(off).max() == pytest.approx(waits_deg, rel=0.05)


def test_a_pitch_step_turns_the_head_no_faster_than_the_unit_can() -> None:
    times = np.arange(0.0, 1.0, 0.01)
    tilt = ptu.tilt_deg(lambda t: 30.0 if t > 0.0 else 0.0, list(times))

    speed = np.diff(tilt) / np.diff(times)
    assert np.abs(speed).max() == pytest.approx(ptu.MAX_SPEED_DEG_S, rel=0.02)


@pytest.mark.parametrize(
    ("pan_deg", "pitch_deg"), [(0.0, 3.0), (90.0, -5.0), (-90.0, 5.0), (180.0, -3.0)]
)
def test_a_panned_head_pitches_with_the_base_tilt_it_faces(
    pan_deg: float, pitch_deg: float
) -> None:
    # Bow up 3, starboard down 5.
    assert ptu.head_pitch_deg(3.0, 5.0, pan_deg) == pytest.approx(pitch_deg)
