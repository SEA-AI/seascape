"""The colour of the water's body from its chlorophyll. Imports no Blender.

Reflectance: Morel & Maritorena, "Bio-optical properties of oceanic waters: a
reappraisal", JGR 106(C4) 7163, 2001 (doi:10.1029/2000JC000319), for Case 1 waters,
where phytoplankton and what comes with it set the colour.

Colour: Wyman, Sloan & Shirley, "Simple analytic approximations to the CIE XYZ color
matching functions", JCGT 2(2) 1, 2013, under CIE D65 (CIE 15:2004).
"""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt

from seascape.colour import XYZ_TO_SRGB

type FloatArray = npt.NDArray[np.float64]

_TABLE_CSV = Path(__file__).parent / "data" / "morel_maritorena_2001.csv"

# CIE 15:2004 Table T.1: D65's relative spectral power, every 10 nm from 400 to 700.
WAVELENGTH_NM = np.arange(400.0, 701.0, 10.0)
D65 = np.array(
    [
        *(82.7549, 91.4860, 93.4318, 86.6823, 104.865, 117.008, 117.812, 114.861),
        *(115.923, 108.811, 109.354, 107.802, 104.790, 107.689, 104.405, 104.046),
        *(100.000, 96.3342, 95.7880, 88.6856, 90.0062, 89.5991, 87.6987, 83.2886),
        *(83.6992, 80.0268, 80.2146, 82.2778, 78.2842, 69.7213, 71.6091),
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
    """The body's diffuse colour in linear sRGB: R(0-) under D65, times t / n^2 out
    through a surface of index `ior` (Lee, Carder & Arnone 2002), t from Fresnel at
    normal incidence. Clipped at 0, where the colour is outside sRGB."""
    matching = _matching(WAVELENGTH_NM)
    xyz = (matching * D65 * reflectance(chl_mg_m3, WAVELENGTH_NM)).sum(axis=1)
    xyz /= (matching[1] * D65).sum()
    t = 1 - ((ior - 1) / (ior + 1)) ** 2
    r, g, b = np.maximum(XYZ_TO_SRGB @ xyz * t / ior**2, 0.0)
    return float(r), float(g), float(b)
