"""Colour temperature, white balance and a camera's tone curve."""

from collections.abc import Callable
from typing import Literal

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


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1's sRGB transfer function, inverted: encoded value to linear."""
    return np.where(
        encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
    )


# Grossberg & Nayar 2004's EMoR mean response f0, the mean of the 201 curves of DoRF
# (Columbia CAVE, emor.txt), at the irradiances `srgb_decode(TONE_IN)`.
EMOR_F0 = np.array([
    0.0000, 0.0305, 0.0638, 0.1078, 0.1620, 0.2266, 0.3020, 0.3844, 0.4696,
    0.5533, 0.6325, 0.7055, 0.7731, 0.8363, 0.8947, 0.9495, 1.0000,
])  # fmt: skip

# sRGB-encoded inputs a tone curve is given at, enough for a spline through them to
# stay within a level of 8 bits.
TONE_IN = np.linspace(0.0, 1.0, len(EMOR_F0))

type Tone = Literal["bt709", "emor"]

_CURVES: dict[Tone, Callable[[np.ndarray], np.ndarray]] = {
    # ITU-R BT.709-6 item 1.2, the OETF, of linear light.
    "bt709": lambda linear: np.where(
        linear < 0.018, 4.5 * linear, 1.099 * linear**0.45 - 0.099
    ),
    "emor": lambda linear: np.interp(linear, srgb_decode(TONE_IN), EMOR_F0),
}


def tone_curve(tone: Tone, encoded: np.ndarray = TONE_IN) -> np.ndarray:
    """A camera's encoded output at the sRGB-encoded inputs `encoded`. EMoR's is
    linear between its knots."""
    return _CURVES[tone](srgb_decode(encoded))
