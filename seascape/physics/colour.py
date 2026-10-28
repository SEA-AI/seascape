"""Colour temperature and white balance."""

import numpy as np

# CIE 15:2004: D65's correlated colour temperature, the white of a linear sRGB render.
D65_K = 6504.0

# IEC 61966-2-1: CIE XYZ to linear sRGB.
XYZ_TO_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


def planck_xy(t_k: float) -> tuple[float, float]:
    """The Planckian locus in CIE 1931 xy, from 1667 to 25000 K: Kim et al.'s cubic
    spline (US 7,024,034)."""
    if not 1667.0 <= t_k <= 25000.0:
        raise ValueError(f"{t_k} K is outside the cubic's 1667-25000 K")
    u = 1e3 / t_k
    if t_k <= 4000.0:
        x = -0.2661239 * u**3 - 0.2343589 * u**2 + 0.8776956 * u + 0.179910
    else:
        x = -3.0258469 * u**3 + 2.1070379 * u**2 + 0.2226347 * u + 0.240390
    if t_k <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t_k <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def _srgb(t_k: float) -> np.ndarray:
    x, y = planck_xy(t_k)
    return XYZ_TO_SRGB @ np.array([x / y, 1.0, (1.0 - x - y) / y])


def white_balance_gains(error_k: float) -> tuple[float, float, float]:
    """The R, G, B gains of a camera that takes D65 light for `D65_K + error_k`: it
    divides out the light it guessed, relative to D65, with green fixed at 1."""
    gains = _srgb(D65_K) / _srgb(D65_K + error_k)
    r, _, b = gains / gains[1]
    return float(r), 1.0, float(b)
