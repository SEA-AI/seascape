"""A pan-tilt unit holding its head's pitch: the tilt loop of a FLIR PTU-5 as
SEA.AI's PTU firmware drives it.
"""

from collections.abc import Callable, Sequence

import numpy as np

# `PGain`: deg/s of tilt speed per degree the head is off its target; `DGain` and
# `IGain` are 0.
GAIN_PER_S = 10.0
# The clamp on `tilt_speed_des`.
MAX_SPEED_DEG_S = 50.0
# `tilt_speed_des`'s dead band filter.
DEAD_BAND_DEG_S = 1.0
# The stabilisation thread's period, STAB_UPDATE_TIME_NS.
TICK_S = 0.015
# Judgement: ten of the loop's time constants, so the head starts in its steady lag.
SETTLE_S = 10.0 / GAIN_PER_S


def tilt_deg(
    head_pitch_deg: Callable[[float, float], float],
    times_s: Sequence[float],
    elevation_deg: float,
) -> np.ndarray:
    """The tilt the loop holds at each of `times_s`, from the head on target
    `SETTLE_S` before 0.

    Args:
        head_pitch_deg: The head's pitch, up, that the IMU on it reads, given a time
            from `-SETTLE_S` and the unit's tilt.
        times_s: Ascending, from 0.
        elevation_deg: The pitch the loop holds the head at, up from level.
    """
    ticks = np.arange(-SETTLE_S, times_s[-1] + TICK_S, TICK_S)
    tilt = np.empty_like(ticks)
    held = elevation_deg - head_pitch_deg(-SETTLE_S, 0.0)
    for i, t_s in enumerate(ticks):
        tilt[i] = held
        speed = GAIN_PER_S * (elevation_deg - head_pitch_deg(float(t_s), held))
        speed = float(np.clip(speed, -MAX_SPEED_DEG_S, MAX_SPEED_DEG_S))
        if abs(speed) < DEAD_BAND_DEG_S:
            speed = 0.0
        held += speed * TICK_S
    return np.interp(times_s, ticks, tilt)
