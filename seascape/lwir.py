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

Sky emissivity: one LOWTRAN7 run, midlatitude summer profile without aerosol, observer
at 12 m, integrated over the band. LOWTRAN7 is public-domain (AFGL-TR-88-0177); the run
is reproducible with `lowtran` on PyPI. A band model with a fixed profile, not a
radiometric reference.

Path transmittance: LOWTRAN7 again, the same profile, a horizontal path at 12 m with
the maritime aerosol at each meteorological range; `data/lowtran_path.csv` says how.
Water vapour takes most of it.

Planck's law and Fresnel for an absorbing medium are textbook, but carry two assumptions
that fail silently:

- Kirchhoff's law, eps = 1 - R. Holds because water is opaque across this band well
  inside any depth the sensor resolves, so there is no transmitted term.
- Band emissivity is the Planck-weighted mean of the spectral emissivity. Exact only for
  a flat sensor response.
"""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]

_TABLE_CSV = Path(__file__).parent / "data" / "water_nk.csv"
_PATH_CSV = Path(__file__).parent / "data" / "lowtran_path.csv"

BAND_M = (8.0e-6, 14.0e-6)

# SI defining constants, exact since the 2019 redefinition.
PLANCK_H = 6.62607015e-34  # J s
LIGHT_C = 2.99792458e8  # m s^-1
BOLTZMANN_K = 1.380649e-23  # J K^-1

T_SEA_K = 288.0
T_AIR_K = 288.0

# Downwelling sky emissivity against elevation above the horizon, in degrees.
# Normalised by the horizon value, which is ambient: a horizontal path is optically
# thick, so the sky at the horizon is a blackbody at air temperature. One curve then
# serves any air temperature, scaled by Planck.
_SKY_EPS = (
    (0.0, 1.0000),
    (1.0, 0.9898),
    (2.0, 0.9863),
    (3.0, 0.9789),
    (5.0, 0.9507),
    (7.0, 0.9129),
    (10.0, 0.8535),
    (15.0, 0.7663),
    (20.0, 0.6973),
    (30.0, 0.6001),
    (45.0, 0.5143),
    (60.0, 0.4671),
    (90.0, 0.4352),
)


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
FACET_SAMPLES = 4096


def emissivity_curve(
    *, t_sea_k: float = T_SEA_K, slope_sigma: float = 0.0
) -> tuple[FloatArray, FloatArray]:
    """Planck-weighted, band-integrated emissivity against viewing zenith (rad).

    `slope_sigma` is the RMS slope per axis the renderer does *not* resolve; at 0 this
    is flat-surface Fresnel. Above 0 it averages Fresnel over facets drawn from a
    Gaussian slope distribution, weighted by the area each presents to the viewer,
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
    rng = np.random.default_rng(0)
    slope = rng.normal(0.0, slope_sigma, size=(FACET_SAMPLES, 2))
    normal = np.stack([-slope[:, 0], -slope[:, 1], np.ones(FACET_SAMPLES)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)

    view = np.stack([np.sin(theta), np.zeros(CURVE_ANGLES), np.cos(theta)], axis=-1)
    cos_i = view @ normal.T  # (angle, facet)
    area = np.clip(cos_i, 0.0, None)
    eps = np.interp(np.arccos(np.clip(cos_i, -1.0, 1.0)), theta, table)
    return theta, (eps * area).sum(axis=1) / area.sum(axis=1)


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


def sky_radiance(elev_rad: npt.ArrayLike, t_air_k: float = T_AIR_K) -> FloatArray:
    """Downwelling in-band sky radiance at an elevation above the horizon.

    Below the horizon the curve holds at ambient, which is what a ray that misses the
    sea should see.
    """
    elev, eps = np.array(_SKY_EPS, dtype=np.float64).T
    fraction = np.interp(np.asarray(elev_rad, dtype=np.float64), np.radians(elev), eps)
    return fraction * band_radiance(t_air_k)


@functools.lru_cache(maxsize=1)
def _path_table() -> tuple[FloatArray, FloatArray, FloatArray]:
    """(range m, 1 / visibility km with 0 for no aerosol, optical depth), the optical
    depth shaped (range, visibility) and the visibilities ascending."""
    with _PATH_CSV.open() as f:
        header = [line for line in f if line.startswith("#")][-1]
    columns = header.removeprefix("# range_m,").strip().split(",")
    inverse = np.array([0.0 if c == "none" else 1 / float(c[4:-2]) for c in columns])
    raw = np.loadtxt(_PATH_CSV, delimiter=",", comments="#")
    order = np.argsort(inverse)
    depth = -np.log(raw[:, 1:][:, order])
    for a in (raw, inverse, depth):
        a.setflags(write=False)
    return raw[:, 0], inverse[order], depth


def path_optical_depth(
    range_m: npt.ArrayLike, visibility_km: float | None
) -> FloatArray:
    """-ln of the band transmittance along a horizontal path near the sea.

    Linear in 1 / visibility, which is how aerosol extinction scales, and in log range.
    A visibility below the table's shortest is held at it. Short of the table it
    follows the power law of its first two rows: band depth grows as a power of the
    path, not linearly, while the strongest lines saturate.
    """

    ranges, inverse, depth = _path_table()
    x = 0.0 if visibility_km is None else 1 / visibility_km
    x = float(np.clip(x, inverse[0], inverse[-1]))
    at = np.array([np.interp(x, inverse, row) for row in depth])
    d = np.asarray(range_m, dtype=np.float64)
    power = np.log(at[1] / at[0]) / np.log(ranges[1] / ranges[0])
    near = at[0] * (np.maximum(d, 0.0) / ranges[0]) ** power
    far = np.interp(np.log(np.maximum(d, ranges[0])), np.log(ranges), at)
    return np.where(d < ranges[0], near, far)
