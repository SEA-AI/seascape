"""The colour of the water's body from its chlorophyll."""

from pathlib import Path

import numpy as np

_CURVE_CSV = Path(__file__).parent / "data" / "water_colour.csv"

_CHL_MG_M3, *_RGB = np.loadtxt(_CURVE_CSV, delimiter=",", comments="#", unpack=True)


def body_color(chl_mg_m3: float, ior: float) -> tuple[float, float, float]:
    """The body's diffuse colour in linear sRGB: R(0-) under daylight, times t / n^2
    out through a surface of index `ior` (Lee, Carder & Arnone 2002), t from Fresnel
    at normal incidence. Clipped at 0, where the colour is outside sRGB."""
    log_chl = np.log(_CHL_MG_M3)
    rgb = [np.interp(np.log(chl_mg_m3), log_chl, channel) for channel in _RGB]
    t = 1 - ((ior - 1) / (ior + 1)) ** 2
    r, g, b = np.maximum(np.array(rgb) * t / ior**2, 0.0)
    return float(r), float(g), float(b)
