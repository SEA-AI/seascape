"""Fog overhead: a layer of drops at the surface that the sky and the sun shine
through, in LWIR from LOWTRAN's path table and in the visible by Koschmieder."""

import numpy as np
import numpy.typing as npt

from seascape.physics import lwir
from seascape.physics.atmosphere import ATMOSPHERE, Atmosphere

FloatArray = npt.NDArray[np.float64]

# LOWTRAN 7's depth of a fog at the surface where its input gives none (lowtran7.f,
# the vertical structure algorithm's case 1: "DEPTH OF FOG/CLOUD IS 0.2 KM").
DEPTH_M = 200.0


def _slant_m(elev_rad: npt.ArrayLike) -> FloatArray:
    """The path up through the layer; at or below the horizon it never leaves it."""
    up = np.sin(np.asarray(elev_rad, dtype=np.float64))
    return np.divide(DEPTH_M, up, out=np.full_like(up, np.inf), where=up > 0.0)


def transmittance(
    elev_rad: npt.ArrayLike, visibility_km: float, atmosphere: Atmosphere = ATMOSPHERE
) -> FloatArray:
    """What the fog passes in LWIR looking up at `elev_rad`: its drops' depth along the
    slant path, the path table's less the same path's without aerosol. The gases are
    the sky's above it."""
    path = _slant_m(elev_rad)
    drops = lwir.path_optical_depth(
        path, visibility_km, atmosphere
    ) - lwir.path_optical_depth(path, None, atmosphere)
    return np.exp(-drops)


def sky_radiance(
    radiance: npt.ArrayLike,
    elev_rad: npt.ArrayLike,
    visibility_km: float,
    t_air_k: float,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> FloatArray:
    """`radiance`, an LWIR sky, seen up through the fog: it passes `transmittance` of
    the sky and emits the rest as a blackbody at air temperature (Kirchhoff), as
    `lwir.cloudy_sky_radiance` takes a cloud."""
    passed = transmittance(elev_rad, visibility_km, atmosphere)
    return passed * np.asarray(radiance) + (1.0 - passed) * lwir.band_radiance(t_air_k)


def cover(visibility_km: float, atmosphere: Atmosphere = ATMOSPHERE) -> float:
    """The fog's LWIR emissivity as a horizontal surface sees the sky: the cloud
    fraction it amounts to. Each elevation weighs its projected solid angle,
    sin e cos e de, which is uniform in sin^2 e."""
    # Judgement: midpoints of a smooth integrand.
    n = 1000
    up = np.sqrt((np.arange(n) + 0.5) / n)
    return float(1.0 - transmittance(np.arcsin(up), visibility_km, atmosphere).mean())


def sun_transmittance(elev_rad: float, extinction_per_m: float) -> float:
    """What the fog leaves of the sun's direct beam: Koschmieder's extinction along
    the slant path. Drops far larger than the wavelength extinguish it alike across
    the sun's band, Mie's efficiency near 2 (van de Hulst 1957)."""
    return float(np.exp(-extinction_per_m * _slant_m(elev_rad)))
