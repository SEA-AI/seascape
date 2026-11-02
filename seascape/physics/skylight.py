"""The sky's luminance by direction: the CIE standard general sky.

ISO 15469:2004 / CIE S 011/E:2003; Darula & Kittler, "CIE general sky standard
defining luminance distributions", Proc. eSim 2002. A sky's luminance at zenith angle Z
and angle chi from the sun, over its zenith's, is f(chi) phi(Z) / (f(Z_s) phi(0)), Z_s
being the sun's zenith angle. The gradation is phi(Z) = 1 + a exp(b / cos Z) and the
indicatrix f(chi) = 1 + c (exp(d chi) - exp(d pi / 2)) + e cos^2 chi.
"""

import numpy as np

# ISO 15469 table 1: a, b, c, d, e of types 1-15.
TYPES: dict[int, tuple[float, float, float, float, float]] = {
    1: (4.0, -0.70, 0.0, -1.0, 0.00),
    2: (4.0, -0.70, 2.0, -1.5, 0.15),
    3: (1.1, -0.80, 0.0, -1.0, 0.00),
    4: (1.1, -0.80, 2.0, -1.5, 0.15),
    5: (0.0, -1.00, 0.0, -1.0, 0.00),
    6: (0.0, -1.00, 2.0, -1.5, 0.15),
    7: (0.0, -1.00, 5.0, -2.5, 0.30),
    8: (0.0, -1.00, 10.0, -3.0, 0.45),
    9: (-1.0, -0.55, 2.0, -1.5, 0.15),
    10: (-1.0, -0.55, 5.0, -2.5, 0.30),
    11: (-1.0, -0.55, 10.0, -3.0, 0.45),
    12: (-1.0, -0.32, 10.0, -3.0, 0.45),
    13: (-1.0, -0.32, 16.0, -3.0, 0.30),
    14: (-1.0, -0.15, 16.0, -3.0, 0.30),
    15: (-1.0, -0.15, 24.0, -2.8, 0.15),
}


def gradation(cie_type: int, zenith_rad: np.ndarray | float) -> np.ndarray:
    """phi(Z): how the sky brightens or darkens toward the zenith."""
    a, b, *_ = TYPES[cie_type]
    return 1.0 + a * np.exp(b / np.cos(zenith_rad))


def indicatrix(cie_type: int, from_sun_rad: np.ndarray | float) -> np.ndarray:
    """f(chi): how the sky brightens toward the sun."""
    _, _, c, d, e = TYPES[cie_type]
    return (
        1.0
        + c * (np.exp(d * from_sun_rad) - np.exp(d * np.pi / 2))
        + e * np.cos(from_sun_rad) ** 2
    )


def relative_luminance(
    cie_type: int,
    zenith_rad: np.ndarray,
    from_sun_rad: np.ndarray,
    sun_zenith_rad: float,
) -> np.ndarray:
    """The luminance of sky `cie_type` over its zenith's, above the horizon."""
    return (
        indicatrix(cie_type, from_sun_rad)
        * gradation(cie_type, zenith_rad)
        / (indicatrix(cie_type, sun_zenith_rad) * gradation(cie_type, 0.0))
    )
