"""A turn from rest to rest: the rate ramps up at a constant acceleration, cruises at
its cap and ramps down, a trapezoid; a short turn never reaches the cap, a triangle.
"""

import math


def peak_deg_s(
    distance_deg: float, rate_deg_s: float, accel_deg_s2: float | None
) -> float:
    """The fastest a turn of `distance_deg` gets, at or under `rate_deg_s`."""
    if accel_deg_s2 is None:
        return rate_deg_s
    # Up to the peak and back down covers peak^2 / accel.
    return min(rate_deg_s, math.sqrt(abs(distance_deg) * accel_deg_s2))


def duration_s(
    distance_deg: float, rate_deg_s: float, accel_deg_s2: float | None
) -> float:
    """How long the turn takes."""
    if distance_deg == 0.0:
        return 0.0
    peak = peak_deg_s(distance_deg, rate_deg_s, accel_deg_s2)
    ramp_s = 0.0 if accel_deg_s2 is None else peak / accel_deg_s2
    # The two ramps cover peak * ramp_s, as far as cruising at the peak for ramp_s.
    return abs(distance_deg) / peak + ramp_s


def covered_deg(
    distance_deg: float, rate_deg_s: float, accel_deg_s2: float | None, t_s: float
) -> float:
    """How far the turn has gone `t_s` after it starts, signed as `distance_deg`."""
    total_s = duration_s(distance_deg, rate_deg_s, accel_deg_s2)
    t_s = min(max(t_s, 0.0), total_s)
    peak = peak_deg_s(distance_deg, rate_deg_s, accel_deg_s2)
    if accel_deg_s2 is None:
        done = peak * t_s
    else:
        ramp_s = peak / accel_deg_s2
        up_s, down_s = min(t_s, ramp_s), max(t_s - (total_s - ramp_s), 0.0)
        cruise_s = t_s - up_s - down_s
        done = (
            0.5 * accel_deg_s2 * up_s**2
            + peak * cruise_s
            + peak * down_s
            - 0.5 * accel_deg_s2 * down_s**2
        )
    return math.copysign(min(done, abs(distance_deg)), distance_deg)
