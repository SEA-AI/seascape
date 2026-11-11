"""A slew from rest to rest: the rate ramps up at a constant acceleration, cruises at
its peak and ramps down, a trapezoid; a short slew never reaches its rate, a triangle.
"""

import math
from typing import NamedTuple


def _peak_deg_s(distance_deg: float, rate_deg_s: float, accel_deg_s2: float) -> float:
    """The fastest a slew of `distance_deg` gets, at or under `rate_deg_s`."""
    # Up to the peak and back down covers peak^2 / accel.
    return min(rate_deg_s, math.sqrt(abs(distance_deg) * accel_deg_s2))


def duration_s(
    distance_deg: float, rate_deg_s: float, accel_deg_s2: float | None
) -> float:
    """How long a slew of `distance_deg` at `rate_deg_s` takes."""
    if distance_deg == 0.0:
        return 0.0
    if accel_deg_s2 is None:
        return abs(distance_deg) / rate_deg_s
    peak = _peak_deg_s(distance_deg, rate_deg_s, accel_deg_s2)
    # The two ramps cover peak * ramp_s, as far as cruising at the peak for ramp_s.
    return abs(distance_deg) / peak + peak / accel_deg_s2


def covered_deg(
    distance_deg: float, rate_deg_s: float, accel_deg_s2: float | None, t_s: float
) -> float:
    """How far the slew has gone `t_s` after it starts, signed as `distance_deg`."""
    total_s = duration_s(distance_deg, rate_deg_s, accel_deg_s2)
    t_s = min(max(t_s, 0.0), total_s)
    if accel_deg_s2 is None:
        done = rate_deg_s * t_s
    else:
        peak = _peak_deg_s(distance_deg, rate_deg_s, accel_deg_s2)
        ramp_s = peak / accel_deg_s2
        up_s, down_s = min(t_s, ramp_s), max(t_s - (total_s - ramp_s), 0.0)
        # The ramp up, then the peak, less what the ramp down falls short of it.
        done = (
            0.5 * accel_deg_s2 * up_s**2
            + peak * (t_s - up_s)
            - 0.5 * accel_deg_s2 * down_s**2
        )
    return math.copysign(min(done, abs(distance_deg)), distance_deg)


class Leg(NamedTuple):
    """A move's slew, from where the last move's hold ended: `distance_deg` signed,
    clockwise positive, at `rate_deg_s`, taking `duration_s` before its `hold_s`.
    """

    start_s: float
    start_deg: float
    distance_deg: float
    rate_deg_s: float
    duration_s: float
    hold_s: float


def chain(
    moves: list[tuple[float, float, float]], accel_deg_s2: float | None
) -> list[Leg]:
    """Slews one after another from rest at 0 deg, t = 0: each move is a target angle,
    its rate and the hold once there, in degrees and seconds.
    """
    legs, at_s, at_deg = [], 0.0, 0.0
    for to_deg, rate_deg_s, hold_s in moves:
        distance = to_deg - at_deg
        took_s = duration_s(distance, rate_deg_s, accel_deg_s2)
        legs.append(Leg(at_s, at_deg, distance, rate_deg_s, took_s, hold_s))
        at_s, at_deg = at_s + took_s + hold_s, to_deg
    return legs
