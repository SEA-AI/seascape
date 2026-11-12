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
    tilt = ptu.tilt_deg(
        lambda t, held: 4.0 * math.sin(omega * t) + held, list(times), target_deg
    )

    off = 4.0 * np.sin(omega * times) + tilt - target_deg
    # |H| of e' = -K e + p' at omega: omega / sqrt(omega^2 + K^2).
    gain = omega / math.hypot(omega, ptu.GAIN_PER_S)
    # Judgement: 5% for the 15 ms tick and the dead band, which the linear gain omits.
    assert np.abs(off).max() == pytest.approx(4.0 * gain, rel=0.05)


def test_the_head_waits_until_its_error_asks_for_more_than_the_dead_band() -> None:
    times = np.arange(0.0, 10.0, 0.1)
    tilt = ptu.tilt_deg(lambda t, held: 0.05 * t + held, list(times), 0.0)

    off = 0.05 * times + tilt
    # It moves once K times the error reaches the dead band, and then keeps up.
    waits_deg = ptu.DEAD_BAND_DEG_S / ptu.GAIN_PER_S
    # Judgement: 5% for the 15 ms tick, by which the error overshoots the dead band.
    assert np.abs(off).max() == pytest.approx(waits_deg, rel=0.05)


def test_a_pitch_step_turns_the_head_no_faster_than_the_unit_can() -> None:
    times = np.arange(0.0, 1.0, 0.01)
    tilt = ptu.tilt_deg(
        lambda t, held: (30.0 if t > 0.0 else 0.0) + held, list(times), 0.0
    )

    speed = np.diff(tilt) / np.diff(times)
    assert np.abs(speed).max() == pytest.approx(ptu.MAX_SPEED_DEG_S, rel=0.02)


def test_a_head_started_past_the_zenith_starts_on_target() -> None:
    # Deck and mount pitched 165 between them: the IMU reads the pitch folded.
    def head_deg(_t_s: float, held: float) -> float:
        up = math.radians(165.0 + held)
        return math.degrees(math.asin(math.sin(up)))

    tilt = ptu.tilt_deg(head_deg, [0.0], 0.0)

    assert head_deg(0.0, tilt[0]) == pytest.approx(0.0, abs=0.1)
