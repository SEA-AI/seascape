"""The sun's direct beam, from the sky's diffuse light.

Erbs, Klein & Duffie, "Estimation of the diffuse radiation fraction for hourly, daily
and monthly-average global radiation", Solar Energy 28(4) 293, 1982: the diffuse
fraction k = D / (D + B sin e) of the light on a horizontal surface, D from the sky and
B the beam across itself at elevation e.
"""

import math


def beam_normal(diffuse: float, diffuse_fraction: float, elevation_rad: float) -> float:
    """B across the beam, in the diffuse light's units.

    Args:
        diffuse: D, the sky's light on a horizontal surface.
        diffuse_fraction: k, the sky's share of the light on a horizontal surface.
        elevation_rad: The sun's elevation e; a horizontal surface gets B sin e."""
    k = diffuse_fraction
    return diffuse * (1.0 - k) / (k * math.sin(elevation_rad))
