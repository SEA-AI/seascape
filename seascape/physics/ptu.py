"""A pan-tilt unit holding its head's pitch: the tilt loop of a FLIR PTU-5 as
SEA.AI's firmware drives it (POWER_Stabilization, `ptu_control_c`, `stabilize_ptu`).
"""

import math
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
# Ten of the loop's time constants: the head starts in its steady lag, as a unit
# that has run for a while.
SETTLE_S = 10.0 / GAIN_PER_S


def tilt_deg(
    pitch_deg: Callable[[float], float],
    times_s: Sequence[float],
    target_deg: float = 0.0,
) -> np.ndarray:
    """The tilt the loop holds at each of `times_s`, from the head on target
    `SETTLE_S` before 0.

    Args:
        pitch_deg: The base's pitch under the head, up, from `-SETTLE_S`.
        times_s: Ascending, from 0.
        target_deg: The head's pitch the loop holds, up; `target_inclination`.
    """
    ticks = np.arange(-SETTLE_S, times_s[-1] + TICK_S, TICK_S)
    tilt = np.empty_like(ticks)
    held = target_deg - pitch_deg(-SETTLE_S)
    for i, t_s in enumerate(ticks):
        tilt[i] = held
        speed = GAIN_PER_S * (target_deg - pitch_deg(float(t_s)) - held)
        speed = float(np.clip(speed, -MAX_SPEED_DEG_S, MAX_SPEED_DEG_S))
        if abs(speed) < DEAD_BAND_DEG_S:
            speed = 0.0
        held += speed * TICK_S
    return np.interp(times_s, ticks, tilt)


def head_pitch_deg(pitch_deg: float, roll_deg: float, pan_deg: float) -> float:
    """The base's tilt under a head panned `pan_deg` clockwise from the bow: bow up
    ahead, starboard up abeam to starboard. Small angles, so pitch and roll add as
    the components of one tilt vector.

    Args:
        pitch_deg: Bow up.
        roll_deg: Starboard down.
        pan_deg: Clockwise from the bow.
    """
    pan = math.radians(pan_deg)
    return pitch_deg * math.cos(pan) - roll_deg * math.sin(pan)
