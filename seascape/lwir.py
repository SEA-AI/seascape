"""Band-integrated LWIR emissivity of a water surface, and the sky it reflects.

Blender is an RGB renderer with no concept of the 8-14 um band, and its Fresnel node
takes a scalar IOR where water needs a complex one (n + i*k). So these curves are
evaluated here and the shader consumes them as 1D lookups. Angles are radians.

A sea surface emits and reflects, and the two are complements: `1 - eps` of what it does
not emit comes back as reflected sky. Leave the reflection out and the sea goes black at
grazing incidence, which is most of a maritime image.

Sources
-------
Optical constants: Nalli et al., "Temperature-dependent optical constants of water in
the thermal infrared derived from data archaeology", Optics Continuum 1(4) 738, 2022
(doi:10.1364/OPTCON.450833); data doi:10.6084/m9.figshare.19341533, CC BY 4.0. That is
Downing & Williams 1975 extended across 271-311 K using Pinkley et al. 1977.

Facet averaging: Masuda, Takashima & Takayama, "Emissivity of pure and sea waters
for the model sea surface in the infrared window regions", Remote Sensing of Environment
24(2) 313, 1988 (doi:10.1016/0034-4257(88)90032-6). The inter-facet reflection term it
omits is in Wu & Smith, "Emissivity of rough sea surface for 8-13 um: modeling and
verification", Applied Optics 36(12) 2609, 1997 (doi:10.1364/AO.36.002609).

Sky emissivity and path transmittance: LOWTRAN7 band models (AFGL-TR-88-0177, public
domain), not a radiometric reference; the headers of `data/lowtran_*.csv` say how each
was run and from what air. The sky is normalised by its horizon, taken as a blackbody
at air temperature, which holds where a horizontal path is opaque: every profile but
subarctic winter. Along a path, water vapour takes most of the band.

Cloud: optically thick water cloud is close to a blackbody in this band (Stephens,
"Radiation profiles in extended water clouds. II", J. Atmos. Sci. 35(11) 2123, 1978;
Smith & Toumi, "Measuring cloud cover and brightness temperature with a ground-based
thermal infrared camera", J. Appl. Meteor. Climatol. 47(2) 683, 2008).

Planck's law and Fresnel for an absorbing medium are textbook, but carry two assumptions
that fail silently:

- Kirchhoff's law, eps = 1 - R. Holds because water is opaque across this band well
  inside any depth the sensor resolves, so there is no transmitted term.
- Band emissivity is the Planck-weighted mean of the spectral emissivity. Exact only for
  a flat sensor response.
"""

import functools
import math
from pathlib import Path
from statistics import NormalDist
from typing import Literal

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

_TABLE_CSV = Path(__file__).parent / "data" / "water_nk.csv"
_PATH_CSV = Path(__file__).parent / "data" / "lowtran_path.csv"
_SKY_CSV = Path(__file__).parent / "data" / "lowtran_sky.csv"

type Atmosphere = Literal[
    "tropical",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
    "us_standard",
    "north_sea",
    "north_sea_winter",
    "north_sea_summer",
]
# Judgement: European waters, as measured at Helgoland.
ATMOSPHERE: Atmosphere = "north_sea"
_SURFACE_CSV = Path(__file__).parent / "data" / "lowtran_surface.csv"


def _columns(csv: Path) -> list[str]:
    """The names after the first in the last comment line, the header."""
    with csv.open() as f:
        header = [line for line in f if line.startswith("#")][-1]
    return header.removeprefix("#").strip().split(",")[1:]


def _surface(column: str) -> dict[str, float]:
    names = ["profile", *_columns(_SURFACE_CSV)]
    rows = np.genfromtxt(_SURFACE_CSV, delimiter=",", dtype=None, names=names)
    return dict(zip(rows["profile"].tolist(), rows[column].tolist(), strict=True))


# Each profile's air and sea at the surface; `data/lowtran_surface.csv` says whose.
SURFACE_AIR_K = _surface("air_k")
SURFACE_SEA_K = _surface("sea_k")

BAND_M = (8.0e-6, 14.0e-6)

# SI defining constants, exact since the 2019 redefinition.
PLANCK_H = 6.62607015e-34  # J s
LIGHT_C = 2.99792458e8  # m s^-1
BOLTZMANN_K = 1.380649e-23  # J K^-1

T_SEA_K = SURFACE_SEA_K[ATMOSPHERE]


def _checked_kelvin(t_k: float) -> float:
    """Reject anything that is not a temperature.

    Written as `not > 0` so nan raises as well as zero and negatives; nan would
    otherwise survive np.clip and turn a whole lookup into silent NaN.
    """
    if not t_k > 0.0:
        raise ValueError(
            f"temperature must be a positive number of kelvin, got {t_k!r}"
        )
    return float(t_k)


@functools.lru_cache(maxsize=1)
def _table() -> tuple[FloatArray, FloatArray, FloatArray]:
    """The shipped table as (wavenumber, temperature, n and k on that grid).

    n and k come back shaped (temperature, wavenumber). Arrays are frozen because the
    cache hands the same objects to every caller.
    """
    raw = np.loadtxt(_TABLE_CSV, delimiter=",", comments="#")
    grid, column = np.unique(raw[:, 0], return_inverse=True)
    temperatures, row = np.unique(raw[:, 1], return_inverse=True)

    nk = np.full((2, len(temperatures), len(grid)), np.nan)
    nk[0, row, column] = raw[:, 2]
    nk[1, row, column] = raw[:, 3]
    if np.isnan(nk).any():
        raise ValueError(f"{_TABLE_CSV.name} is missing rows: the grid has holes")

    for a in (grid, temperatures, nk):
        a.setflags(write=False)
    return grid, temperatures, nk


def optical_constants(
    t_k: float = T_SEA_K,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Wavelength (m), n, k across the band at `t_k`, ascending in wavelength.

    Linearly interpolated between the table's steps and clamped outside its span.
    """
    grid, temperatures, nk = _table()
    t = float(np.clip(_checked_kelvin(t_k), temperatures[0], temperatures[-1]))
    n, k = (
        np.array([np.interp(t, temperatures, col) for col in plane.T]) for plane in nk
    )

    lam = 1e-2 / grid  # cm^-1 -> m
    order = np.argsort(lam)
    lam, n, k = lam[order], n[order], k[order]
    inside = (lam >= BAND_M[0]) & (lam <= BAND_M[1])
    return lam[inside], n[inside], k[inside]


def planck(lam_m: npt.ArrayLike, t_k: float) -> FloatArray:
    """Spectral radiance of a blackbody, W m^-2 sr^-1 m^-1.

    A negative temperature otherwise returns a negative radiance rather than failing.
    """
    _checked_kelvin(t_k)
    lam = np.asarray(lam_m, dtype=np.float64)
    numerator = 2 * PLANCK_H * LIGHT_C**2 / lam**5
    return numerator / (np.exp(PLANCK_H * LIGHT_C / (lam * BOLTZMANN_K * t_k)) - 1.0)


def fresnel_emissivity(
    theta_rad: npt.ArrayLike, n: npt.ArrayLike, k: npt.ArrayLike
) -> FloatArray:
    """Unpolarised emissivity 1-R at incidence `theta_rad` from vacuum.

    numpy's complex sqrt takes the correct branch, so grazing needs no special case.
    Inputs broadcast, so angle and wavelength can be evaluated as a grid.
    """
    n_c = np.asarray(n, dtype=np.float64) + 1j * np.asarray(k, dtype=np.float64)
    theta = np.asarray(theta_rad, dtype=np.float64)
    cos_i = np.cos(theta)
    q = np.sqrt(n_c**2 - np.sin(theta) ** 2)
    r_s = (cos_i - q) / (cos_i + q)
    r_p = (n_c**2 * cos_i - q) / (n_c**2 * cos_i + q)
    return 1.0 - 0.5 * (np.abs(r_s) ** 2 + np.abs(r_p) ** 2)


CURVE_ANGLES = 91
# Judgement; a square, as `_facets` lays it on a grid.
FACET_SAMPLES = 4096


def emissivity_curve(
    *, t_sea_k: float = T_SEA_K, slope_sigma: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Planck-weighted, band-integrated emissivity against viewing zenith (rad).

    `slope_sigma` is the RMS slope per axis the renderer does *not* resolve; at 0 this
    is flat-surface Fresnel. Above 0 it averages Fresnel over facets with Gaussian
    slopes, weighted by the area each presents to the viewer,
    which is the Masuda 1988 construction. Only the unresolved slope belongs here:
    slope the wave normals carry is applied per pixel by the shader, and integrating it
    again would count it twice.

    Shadowing between facets and reflections from one facet to another are not
    included; both raise emissivity further at grazing, so this is a lower bound there.
    Wu & Smith put the multiple-reflection term at 0.02-0.03 around 73 deg.
    """
    lam, n, k = optical_constants(t_sea_k)
    weight = planck(lam, t_sea_k)
    band = np.trapezoid(weight, lam)
    theta = np.linspace(0.0, np.pi / 2, CURVE_ANGLES)

    flat = np.trapezoid(fresnel_emissivity(theta[:, None], n, k) * weight, lam, -1)
    if slope_sigma <= 0.0:
        return theta, flat / band

    table = flat / band
    normal = _facets(slope_sigma)
    view = np.stack([np.sin(theta), np.zeros(CURVE_ANGLES), np.cos(theta)], axis=-1)
    cos_i = view @ normal.T  # (angle, facet)
    area = _seen_area(cos_i, normal)
    eps = np.interp(np.arccos(np.clip(cos_i, -1.0, 1.0)), theta, table)
    return theta, (eps * area).sum(axis=1) / area.sum(axis=1)


def _facets(slope_sigma: float) -> FloatArray:
    """Unit normals of facets whose slopes are Gaussian, `slope_sigma` per axis.

    On a grid of the normal's quantiles: a random draw's error would be a bias every
    pixel shares. The grid's midpoints stop short of the tails, so it is rescaled to
    the sigma.
    """
    side = math.isqrt(FACET_SAMPLES)
    quantiles = np.vectorize(NormalDist().inv_cdf)((np.arange(side) + 0.5) / side)
    quantiles /= quantiles.std()
    east, north = np.meshgrid(slope_sigma * quantiles, slope_sigma * quantiles)
    normal = np.stack([-east.ravel(), -north.ravel(), np.ones(side**2)], axis=-1)
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def _seen_area(cos_i: FloatArray, normal: FloatArray) -> FloatArray:
    """Per unit of mean surface: a facet's own area is the cell it covers over n_z."""
    return np.clip(cos_i, 0.0, None) / normal[:, 2]


def reflected_sky(
    mirror_rad: npt.ArrayLike,
    sky_elev_rad: npt.ArrayLike,
    sky: npt.ArrayLike,
    *,
    t_sea_k: float = T_SEA_K,
    slope_sigma: float = 0.0,
) -> FloatArray:
    """The sky a rough sea reflects, against the elevation of the viewer's mirror
    direction off the mean surface; `sky` is radiance at the ascending `sky_elev_rad`.

    Masuda 1988's facet average, applied to the reflected sky: each facet reflects
    the sky in its own mirror direction, weighted by the area it presents to the
    viewer and by its own reflectance. The facets seen at grazing lean toward the
    viewer, so they reflect sky above the mirror direction. Weighted so, eps B +
    (1 - eps) times this is the facets' own sum, eps from `emissivity_curve`. A
    reflection below the horizon reads the sky's lowest elevation, as the world does.
    """
    mirror = np.asarray(mirror_rad, dtype=np.float64)
    elev = np.asarray(sky_elev_rad, dtype=np.float64)
    radiance = np.asarray(sky, dtype=np.float64)
    if slope_sigma <= 0.0:
        return np.interp(mirror, elev, radiance)
    normal = _facets(slope_sigma)
    view = np.stack([np.cos(mirror), np.zeros_like(mirror), np.sin(mirror)], axis=-1)
    cos_i = view @ normal.T  # (angle, facet)
    up = 2 * cos_i * normal[:, 2] - view[..., 2:]
    theta, flat = emissivity_curve(t_sea_k=t_sea_k)
    reflectance = 1.0 - np.interp(np.arccos(np.clip(cos_i, -1.0, 1.0)), theta, flat)
    weight = _seen_area(cos_i, normal) * reflectance
    seen = np.interp(np.arcsin(np.clip(up, -1.0, 1.0)), elev, radiance)
    return (seen * weight).sum(axis=-1) / weight.sum(axis=-1)


_BAND_LAM = np.linspace(*BAND_M, 512)


def band_radiance(t_k: float) -> float:
    """Blackbody radiance integrated over the band, W m^-2 sr^-1.

    On its own grid: the seawater table's ends land inside the band and integrate low.
    """
    return float(np.trapezoid(planck(_BAND_LAM, t_k), _BAND_LAM))


# np.interp clamps past the ends: a hull hotter than the grid reads as its top.
_TB_GRID = np.linspace(200.0, 400.0, 1024)
# Through band_radiance, so the two stay inverses.
_TB_RADIANCE = np.array([band_radiance(t) for t in _TB_GRID])


def brightness_temperature(radiance: npt.ArrayLike) -> FloatArray:
    """Invert `band_radiance`: the blackbody temperature that emits this in-band.

    A real surface is not a blackbody, so this reads below its true temperature
    wherever emissivity does.
    """
    return np.interp(np.asarray(radiance, dtype=np.float64), _TB_RADIANCE, _TB_GRID)


@functools.lru_cache(maxsize=1)
def _sky_table() -> tuple[FloatArray, dict[str, FloatArray]]:
    """(elevation deg, each profile's sky against its horizon)."""
    raw = np.loadtxt(_SKY_CSV, delimiter=",", comments="#")
    raw.setflags(write=False)
    curves = {name: raw[:, i + 1] for i, name in enumerate(_columns(_SKY_CSV))}
    return raw[:, 0], curves


def sky_radiance(
    elev_rad: npt.ArrayLike,
    t_air_k: float | None = None,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> FloatArray:
    """Downwelling in-band sky radiance at an elevation above the horizon, under air at
    `t_air_k`, or the atmosphere's own.

    Below the horizon the curve holds at ambient, which is what a ray that misses the
    sea should see.
    """
    if t_air_k is None:
        t_air_k = SURFACE_AIR_K[atmosphere]
    elev, curves = _sky_table()
    fraction = np.interp(
        np.asarray(elev_rad, dtype=np.float64), np.radians(elev), curves[atmosphere]
    )
    return fraction * band_radiance(t_air_k)


# g / c_p: air lifted from the surface cools at it to the cloud base, where it condenses
# (Wallace & Hobbs, Atmospheric Science, 2nd ed., 2006).
DRY_LAPSE_K_PER_M = 9.8e-3


def cloudy_sky_radiance(
    elev_rad: npt.ArrayLike,
    cloud: npt.ArrayLike,
    cloud_base_m: float,
    t_air_k: float | None = None,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> FloatArray:
    """`sky_radiance` with a `cloud` fraction, 0 to 1, a blackbody at the cloud base
    seen through the clear column.

    The column, emissivity eps the clear sky over a blackbody at air temperature,
    transmits 1 - eps of what lies behind it (Kirchhoff). It is taken to lie wholly
    under the cloud, which holds for a low one: water vapour sits low.
    """
    if t_air_k is None:
        t_air_k = SURFACE_AIR_K[atmosphere]
    clear = sky_radiance(elev_rad, t_air_k, atmosphere)
    transmitted = 1.0 - clear / band_radiance(t_air_k)
    base = band_radiance(t_air_k - DRY_LAPSE_K_PER_M * cloud_base_m)
    return clear + np.asarray(cloud) * transmitted * base


@functools.lru_cache(maxsize=1)
def _path_table() -> dict[str, tuple[FloatArray, FloatArray, FloatArray]]:
    """Per profile: (range m, 1 / visibility km with 0 for no aerosol, optical depth),
    the optical depth shaped (range, visibility) and the visibilities ascending."""
    columns = _columns(_PATH_CSV)[1:]
    km = [c.removeprefix("vis_").removesuffix("km") for c in columns]
    inverse = np.array([0.0 if c == "none" else 1 / float(c) for c in km])
    order = np.argsort(inverse)
    names = np.loadtxt(_PATH_CSV, delimiter=",", comments="#", usecols=0, dtype=str)
    raw = np.loadtxt(
        _PATH_CSV, delimiter=",", comments="#", usecols=range(1, 2 + len(columns))
    )
    tables = {}
    for name in np.unique(names):
        rows = raw[names == name]
        depth = -np.log(rows[:, 1:][:, order])
        rows.setflags(write=False)
        depth.setflags(write=False)
        tables[str(name)] = (rows[:, 0], inverse[order], depth)
    return tables


def path_optical_depth(
    range_m: npt.ArrayLike,
    visibility_km: float | None,
    atmosphere: Atmosphere = ATMOSPHERE,
) -> FloatArray:
    """-ln of the band transmittance along a horizontal path near the sea.

    Linear in 1 / visibility, which is how aerosol extinction scales, and in log range.
    A visibility below the table's shortest is held at it. Short of the table it
    follows the power law of its first two rows: band depth grows as a power of the
    path, not linearly, while the strongest lines saturate.
    """
    ranges, inverse, depth = _path_table()[atmosphere]
    x = 0.0 if visibility_km is None else 1 / visibility_km
    x = float(np.clip(x, inverse[0], inverse[-1]))
    at = np.array([np.interp(x, inverse, row) for row in depth])
    d = np.asarray(range_m, dtype=np.float64)
    power = np.log(at[1] / at[0]) / np.log(ranges[1] / ranges[0])
    near = at[0] * (np.maximum(d, 0.0) / ranges[0]) ** power
    far = np.interp(np.log(np.maximum(d, ranges[0])), np.log(ranges), at)
    return np.where(d < ranges[0], near, far)
