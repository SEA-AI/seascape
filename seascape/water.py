"""The colour of the water's body from its chlorophyll. Imports no Blender.

Reflectance: Morel & Maritorena, "Bio-optical properties of oceanic waters: a
reappraisal", JGR 106(C4) 7163, 2001 (doi:10.1029/2000JC000319), for Case 1 waters,
where phytoplankton and what comes with it set the colour.

Colour: Wyman, Sloan & Shirley, "Simple analytic approximations to the CIE XYZ color
matching functions", JCGT 2(2) 1, 2013, under a Planckian light at D65's temperature
carried to D65's white by the Bradford transform: no tabulated illuminant.
"""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seascape.colour import D65_K, XYZ_TO_SRGB

type FloatArray = npt.NDArray[np.float64]

_TABLE_CSV = Path(__file__).parent / "data" / "morel_maritorena_2001.csv"

# Every 10 nm across the visible the table covers.
WAVELENGTH_NM = np.arange(400.0, 701.0, 10.0)

# CODATA 2018: the second radiation constant, in nm K.
C2_NM_K = 1.438776877e7

# IEC 61966-2-1: the sRGB white, D65, in CIE 1931 xy.
D65_XY = (0.3127, 0.3290)

# Lam 1985, the Bradford transform: XYZ to the sharpened cone space in which a von
# Kries scaling carries one white to another.
BRADFORD = np.array(
    [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]
)

# Morel & Maritorena 2001: R = f bb / a, eq. 7, with f at 0.33; the first guess of a
# is u1 Kd, u1 = 0.75; eq. 8' refines it, mu_d at 0.90, and R holds within three loops.
F = 0.33
U1 = 0.75
MU_D = 0.90
LOOPS = 3


@functools.lru_cache(maxsize=1)
def _table() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Table 2 as (wavelength, Kw, e, chi), frozen: the cache shares them."""
    columns = np.loadtxt(_TABLE_CSV, delimiter=",", comments="#", unpack=True)
    for column in columns:
        column.flags.writeable = False
    wavelength_nm, kw, e, chi = columns
    return wavelength_nm, kw, e, chi


def reflectance(chl_mg_m3: float, wavelength_nm: FloatArray) -> FloatArray:
    """R(0-), the irradiance reflectance just under the surface, of Case 1 water."""
    table_nm, kw, e, chi = _table()
    kd = np.interp(wavelength_nm, table_nm, kw + chi * chl_mg_m3**e)  # eq. 3
    # Morel 1974: pure seawater's scattering, half of it backward.
    bw = 0.00288 * (wavelength_nm / 500.0) ** -4.32
    # Eqs. 12-14: particles scatter as 0.416 Chl^0.766 at 550 nm, backward by a share
    # that turns spectrally neutral at 2 mg/m^3.
    v = 0.5 * (np.log10(chl_mg_m3) - 0.3) if chl_mg_m3 < 2.0 else 0.0
    bbp = (
        (0.002 + 0.01 * (0.5 - 0.25 * np.log10(chl_mg_m3)) * (wavelength_nm / 550) ** v)
        * 0.416
        * chl_mg_m3**0.766
    )
    bb = 0.5 * bw + bbp  # eq. 11
    a = U1 * kd
    for _ in range(LOOPS):
        r = F * bb / a
        a = kd * MU_D * (1 - r) / (1 + 2.25 * r)  # eq. 8'
    return F * bb / a


def _matching(wavelength_nm: FloatArray) -> FloatArray:
    """CIE 1931 2-degree x, y, z bar: Wyman, Sloan & Shirley 2013's multi-lobe fit."""

    def lobe(mu: float, below: float, above: float) -> FloatArray:
        width = np.where(wavelength_nm < mu, below, above)
        return np.exp(-0.5 * ((wavelength_nm - mu) / width) ** 2)

    return np.array(
        [
            1.056 * lobe(599.8, 37.9, 31.0)
            + 0.362 * lobe(442.0, 16.0, 26.7)
            - 0.065 * lobe(501.1, 20.4, 26.2),
            0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1),
            1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8),
        ]
    )


def body_color(chl_mg_m3: float, ior: float) -> tuple[float, float, float]:
    """The body's diffuse colour in linear sRGB: R(0-) under daylight, times t / n^2
    out through a surface of index `ior` (Lee, Carder & Arnone 2002), t from Fresnel
    at normal incidence. Clipped at 0, where the colour is outside sRGB."""
    matching = _matching(WAVELENGTH_NM)
    # Planck's law, relative: the light D65's correlated temperature names.
    light = WAVELENGTH_NM**-5 / np.expm1(C2_NM_K / (WAVELENGTH_NM * D65_K))
    white = matching @ light / (matching[1] @ light)
    xyz = matching @ (light * reflectance(chl_mg_m3, WAVELENGTH_NM))
    xyz /= matching[1] @ light
    x, y = D65_XY
    d65 = np.array([x / y, 1.0, (1 - x - y) / y])
    scale = np.diag((BRADFORD @ d65) / (BRADFORD @ white))
    xyz = np.linalg.solve(BRADFORD, scale @ BRADFORD @ xyz)
    t = 1 - ((ior - 1) / (ior + 1)) ** 2
    r, g, b = np.maximum(XYZ_TO_SRGB @ xyz * t / ior**2, 0.0)
    return float(r), float(g), float(b)
