"""The aerosol overhead that the sky and the sun shine through: maritime haze, and
in fog a layer of drops at the surface. In LWIR from LOWTRAN's tables,
in the visible by Koschmieder."""

import math

import numpy as np
import numpy.typing as npt

from seascape.physics import lwir
from seascape.physics.atmosphere import ATMOSPHERE, Atmosphere

FloatArray = npt.NDArray[np.float64]

# WMO's fog: a visibility under 1 km (WMO No. 8, 2018).
FOG_KM = 1.0

# LOWTRAN 7's shortest maritime haze, HZ2K's 2 km profile (lowtran7.f, AERPRF).
HAZE_KM = 2.0

# Typical of water clouds (Kokhanovsky, Earth-Sci. Rev. 64 189, 2004, 3.1.4).
ASYMMETRY = 0.85

# LOWTRAN 7's depth of a fog at the surface where its input gives none (lowtran7.f,
# the vertical structure algorithm's case 1: "DEPTH OF FOG/CLOUD IS 0.2 KM").
DEPTH_M = 200.0


def extinction_per_m(visibility_km: float | None) -> float:
    """Koschmieder's law: over `visibility_km` a dark target keeps 2% of its contrast
    against the horizon sky."""
    if visibility_km is None:
        return 0.0
    return math.log(1 / 0.02) / (visibility_km * 1000)


def fog_weight(visibility_km: float | None) -> float:
    """How far the air is fog: 0 for haze at `HAZE_KM` and over, 1 at `FOG_KM` and
    under, linear in 1 / visibility between, as LOWTRAN scales aerosol."""
    if visibility_km is None:
        return 0.0
    mist = (1 / visibility_km - 1 / HAZE_KM) / (1 / FOG_KM - 1 / HAZE_KM)
    return min(max(mist, 0.0), 1.0)


def _slant_m(elev_rad: npt.ArrayLike) -> FloatArray:
    """The path up through the fog; at or below the horizon it never leaves it."""
    up = np.sin(np.asarray(elev_rad, dtype=np.float64))
    return np.divide(DEPTH_M, up, out=np.full_like(up, np.inf), where=up > 0.0)


def _fog_depth(
    elev_rad: npt.ArrayLike, visibility_km: float, atmosphere: Atmosphere
) -> FloatArray:
    """The fog drops' LWIR depth along the slant path: the path table's less the same
    path's without aerosol. The sky already holds the gases."""
    path = _slant_m(elev_rad)
    return lwir.path_optical_depth(
        path, visibility_km, atmosphere
    ) - lwir.path_optical_depth(path, None, atmosphere)


def depth(
    elev_rad: npt.ArrayLike,
    visibility_km: float | None,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> FloatArray:
    """The aerosol's LWIR optical depth looking up at `elev_rad`: haze and fog blended
    by `fog_weight`."""
    weight = fog_weight(visibility_km)
    if weight == 1.0:
        assert visibility_km is not None
        return _fog_depth(elev_rad, visibility_km, atmosphere)
    haze = None if visibility_km is None else max(visibility_km, HAZE_KM)
    hazy = lwir.haze_depth(elev_rad, haze, atmosphere)
    if weight == 0.0:
        return hazy
    fog = _fog_depth(elev_rad, FOG_KM, atmosphere)
    return (1.0 - weight) * hazy + weight * fog


def transmittance(
    elev_rad: npt.ArrayLike,
    visibility_km: float | None,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> FloatArray:
    """What the aerosol passes in LWIR of the sky behind, looking up at `elev_rad`."""
    return np.exp(-depth(elev_rad, visibility_km, atmosphere))


def sky_radiance(
    radiance: npt.ArrayLike,
    elev_rad: npt.ArrayLike,
    visibility_km: float | None,
    t_air_k: float,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> FloatArray:
    """`radiance`, an LWIR sky, seen up through the aerosol: it passes `transmittance`
    of the sky and emits the rest as a blackbody at air temperature (Kirchhoff)."""
    passed = transmittance(elev_rad, visibility_km, atmosphere)
    return passed * np.asarray(radiance) + (1.0 - passed) * lwir.band_radiance(t_air_k)


def cover(visibility_km: float | None, atmosphere: Atmosphere = ATMOSPHERE) -> float:
    """The aerosol's LWIR emissivity as a horizontal surface sees the sky: the cloud
    fraction it amounts to. Each elevation weighs its projected solid angle,
    sin e cos e de, which is uniform in sin^2 e."""
    # Judgement: midpoints of a smooth integrand.
    n = 1000
    up = np.sqrt((np.arange(n) + 0.5) / n)
    return float(1.0 - transmittance(np.arcsin(up), visibility_km, atmosphere).mean())


def fog_cover(
    visibility_km: float | None, atmosphere: Atmosphere = ATMOSPHERE
) -> float:
    """`cover` of the fog alone, blended in by `fog_weight`. Haze's longwave is already
    in the clear sky, Swinbank's."""
    weight = fog_weight(visibility_km)
    if weight == 0.0:
        return 0.0
    assert visibility_km is not None
    return weight * cover(min(visibility_km, FOG_KM), atmosphere)


def fog_beam_depth(visibility_km: float | None) -> float:
    """The fog's optical depth overhead in the visible, blended in by `fog_weight`.
    Haze's is already in the clear-sky beam, Meinel's and the Sky Texture's."""
    weight = fog_weight(visibility_km)
    if weight == 0.0:
        return 0.0
    assert visibility_km is not None
    return weight * extinction_per_m(min(visibility_km, FOG_KM)) * DEPTH_M


def sun_transmittance(elev_rad: float, visibility_km: float | None) -> float:
    """What the fog leaves of the sun's direct beam along the slant path. Drops far
    larger than the wavelength extinguish it alike across the sun's band, Mie's
    efficiency near 2 (van de Hulst 1957)."""
    depth = fog_beam_depth(visibility_km)
    if depth == 0.0:
        return 1.0
    up = math.sin(elev_rad)
    if up <= 0.0:
        return 0.0
    return math.exp(-depth / up)


def diffuse_transmittance(visibility_km: float | None) -> float:
    """What the fog passes of sunlight, direct and scattered, its drops absorbing none:
    the two-stream T = 2 / (2 + (1 - g) tau) of a non-absorbing layer, tau
    `fog_beam_depth` (Stephens, AT622 notes 15, Colorado State, eq. 15.20b).

    Limitation: 15.20b takes the light as diffuse. A collimated sun moves the split
    between reflected and passed with its angle, most for a low sun, which this does
    not follow."""
    tau = fog_beam_depth(visibility_km)
    return 2.0 / (2.0 + (1.0 - ASYMMETRY) * tau)
