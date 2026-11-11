"""A slew from rest to rest: the rate ramps up at a constant acceleration, cruises at
its peak and ramps down, a trapezoid; a short slew never reaches its rate, a triangle.
"""

import math


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
