"""Photographed skies, and where each one's sun is, measured from its pixels.

Sources
-------
Poly Haven's pure skies. Its FAQ (https://docs.polyhaven.com/en/faq): "All our HDRIs
are unclipped", so a sun keeps its radiance. Hold-Geoffroy, Sunkavalli, Hadap,
Gambaretto & Lalonde, "Deep outdoor illumination estimation", CVPR 2017
(arXiv:1611.06403): an HDR panorama's sun is its brightest region. The Astronomical
Almanac: the sun's mean semi-diameter is 16 arcminutes, so a uniform disc has an RMS
radius of 0.19 degrees.

Cloud: Li, Lu & Yang, "A hybrid thresholding algorithm for cloud detection on
ground-based color images", J. Atmos. Oceanic Technol. 28(10) 1286, 2011. Clear sky is
bluer than cloud in (B - R) / (B + R) of camera counts, and cloud is grey, B = R. The
clear sky's own blueness falls toward the horizon and the sun, so a cloud is told from
it by its departure from a clear sky fitted to the photo, as Aebi et al. ("Cloud
fraction determined by thermal infrared and visible all-sky cameras", Atmos. Meas.
Tech. 11 5549, 2018, doi:10.5194/amt-11-5549-2018) fit theirs to the pixels that are
not cloud, against zenith angle, and refit; the clear-sky library of Ghonima et al.
(Atmos. Meas. Tech. 5 2881, 2012, doi:10.5194/amt-5-2881-2012) indexes it by zenith
and angle from the sun. An overcast photo has no clear sky to fit; `skies.toml` marks
which.
"""

import math
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from importlib import resources

import cv2
import numpy as np
from pydantic import Field, model_validator

from seascape.assets import File
from seascape.physics import colour, earth, lwir

LIBRARY = resources.files("seascape.assets") / "skies.toml"
LUMINANCE = np.asarray(colour.LUMINANCE, np.float32)
# The midpoint of CARLA's ClearNoon (sun 45 deg up) and ClearSunset (15 deg).
NOON_DEG = 30.0
# Judgement: the peak must outshine the median of the ring round it by this factor. An
# unclipped disc clears it; a sun behind cloud or below the horizon does not. The ring
# starts past DISC_DEG and ends inside AUREOLE_DEG, so it holds neither disc nor glare.
DISC_CONTRAST = 100.0
RING_DEG = (2.0, 5.0)
# Judgement: DISC_DEG reaches past the sun's semi-diameter (module docstring) for the
# glare a lens adds. DISC_FLOOR sits above the ring, which a disc keeps under
# 1 / DISC_CONTRAST of its peak, and keeps the disc's dimmer edge.
DISC_DEG = 1.0
DISC_FLOOR = 0.1
# Judgement: brackets the sun's own RMS radius (module docstring), from a star's one
# pixel to a glow degrees wide.
DISC_RMS_DEG = (0.1, 0.6)
# Judgement: cells finer than a glow, blurred over a few of them, so a patch of sky
# outweighs one bright pixel.
GLOW_CELLS = (128, 256)
GLOW_BLUR = 5
# sRGB's approximate display gamma (IEC 61966-2-1): counts = radiance ** (1 / GAMMA).
GAMMA = 2.2
# Judgement: in (B - R) / (B + R) units. Aebi et al. (2018, Sect. 2.1.1) refit
# with 1.2 K of brightness temperature, another quantity, so no value carries over.
CLEAR_BAND = 0.03
# Judgement: under this blueness a clear sky is too grey for colour to rank cloud, so a
# share divides by this and not by a blueness near 0.
GREY_CLEAR = 0.05
# Aebi et al. (2018, Sect. 2.1.1): the clear-sky fit is repeated up to 10 times. Also
# bounds the darkness refit.
FIT_PASSES = 10
# The bluest pixel of each FIT_STRIDE square is fitted: the clear sky is smooth.
# Judgement: 4 px, a sixteenth of the pixels.
FIT_STRIDE = 4
# Judgement: powers of the horizon's nearness and the sun's. Ghonima et al. (2012,
# Sect. 3.2.1) index their clear sky by the same two angles, as its red-blue ratio
# rises toward both; the powers are not theirs. The steepest halve 5 deg above the
# horizon and 24 deg from the sun.
HORIZON_POWERS = (0, 1, 2, 4, 8)
SUN_POWERS = (0, 1, 2, 4, 8, 16)
# Judgement: the aureole round a disc, as white as cloud, and how far round each of its
# pixels it is filled from. Ghonima et al. (Sect. 4.1) lower the thick-cloud threshold
# out to 35 deg from the sun, where a clear sky saturates; this takes only the white
# core.
AUREOLE_DEG = 8.0
INPAINT_RADIUS_DEG = 0.5
# Judgement: halfway up the share's 0 to 1, so a pixel is clear sky to judge darkness
# against when colour reads it more clear than cloud.
CLEAR_SHARE = 0.5
# Judgement: the luminance, as a fraction of the clear sky's beside it, where a pixel
# starts to read as shaded cloud and where it reads as all cloud. The first sits near
# enough 1 for a shaded base to show and far enough under it for the clear sky's own
# gradient not to.
DARK = (0.9, 0.6)
# Judgement: the Gaussian sigma of "beside", wide enough to reach clear sky past a
# cloud's edge and narrow enough that the clear sky's gradient across it stays small.
BESIDE_DEG = 3.0
# Judgement: cells a quarter of a sigma wide. A cell's own spread is 1 / sqrt(12) of its
# width, which adds under 1% to sigma.
CELLS_PER_SIGMA = 4


class Photo(File):
    """One sky, and its sun as the image holds it: a bearing clockwise from +Y as
    Cycles maps the image unrotated, and an elevation and the disc's `Sun.floor` only
    where a disc shows.
    """

    sun_bearing_deg: float = Field(ge=0.0, lt=360.0)
    sun_elevation_deg: float | None = Field(default=None, gt=-90.0, le=90.0)
    sun_floor: float | None = Field(default=None, gt=0.0)
    # Measured by `diffuse_irradiance`, where no disc shows: a hidden sun's beam is
    # scaled from it.
    diffuse_irradiance: float | None = Field(default=None, gt=0.0)
    # Labelled by eye: no clear sky shows, so none can be fitted.
    overcast: bool = False
    # The genus over most of the sky, by eye (WMO Cloud Atlas); high cloud is cirrus.
    cloud_genus: lwir.CloudGenus
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)

    @model_validator(mode="after")
    def _a_disc_has_a_floor(self) -> "Photo":
        if (self.sun_elevation_deg is None) != (self.sun_floor is None):
            raise ValueError("sun_elevation_deg and sun_floor come together")
        if (self.sun_elevation_deg is None) == (self.diffuse_irradiance is None):
            raise ValueError("diffuse_irradiance is for a photo with no disc")
        return self

    @property
    def sun(self) -> str:
        """The sun's elevation, in words."""
        elevation = self.sun_elevation_deg
        return "no disc" if elevation is None else f"{elevation:.1f} deg"

    @property
    def weather(self) -> str:
        """The sky in CARLA's words, such as "clear noon"; no disc, no time of day."""
        if self.overcast:
            sky = "overcast"
        elif self.cloud_genus in ("clear", "cirrus"):
            sky = "clear"
        else:
            sky = "cloudy"
        if (elevation := self.sun_elevation_deg) is None:
            return sky
        return f"{sky} {'noon' if elevation >= NOON_DEG else 'sunset'}"


@cache
def library() -> dict[str, Photo]:
    """Every sky photo in the library, by name."""
    with LIBRARY.open("rb") as handle:
        return {name: Photo(**body) for name, body in tomllib.load(handle).items()}


@dataclass(frozen=True)
class Sun:
    """`floor`: the luminance a pixel of the disc exceeds; None with no disc."""

    bearing_deg: float
    elevation_deg: float | None
    floor: float | None = None


def _bearing_deg(u: np.ndarray) -> np.ndarray:
    """Cycles reads direction (x, y) at u = 0.5 - atan2(y, x) / 2 pi."""
    return (360.0 * u - 90.0) % 360.0


def _elevation_deg(rows: np.ndarray, h: int) -> np.ndarray:
    return 90.0 - (rows + 0.5) / h * 180.0


def _glow_deg(lum: np.ndarray) -> float:
    """The bearing of the brightest patch, a few degrees wide, above the horizon."""
    rows, cols = GLOW_CELLS
    h, w = lum.shape[0] // rows * rows, lum.shape[1] // cols * cols
    cells = lum[:h, :w].reshape(rows, h // rows, cols, w // cols).mean(axis=(1, 3))
    cells = cells[: rows // 2]
    half = GLOW_BLUR // 2
    # Wrapped across the seam in bearing, clipped at the zenith and the horizon.
    blurred = np.stack([np.roll(cells, s, axis=1) for s in range(-half, half + 1)])
    padded = np.pad(blurred.sum(axis=0), ((half, half), (0, 0)), mode="edge")
    blurred = np.stack([padded[s : s + rows // 2] for s in range(GLOW_BLUR)])
    _, col = np.unravel_index(np.argmax(blurred.sum(axis=0)), blurred.shape[1:])
    return float(_bearing_deg(np.array((col + 0.5) / cols)))


def sun(radiance: np.ndarray) -> Sun:
    """`radiance`, (h, w, 3), an equirectangular image top row first."""
    lum = radiance @ LUMINANCE
    h, w = lum.shape
    glow = _glow_deg(lum)
    row, col = np.unravel_index(np.argmax(lum), lum.shape)
    # A band of rows round the peak, wide enough for the ring.
    band = slice(max(row - h // 30, 0), min(row + h // 30 + 1, h))
    rows = np.arange(h)[band]
    bearings = _bearing_deg((np.arange(w) + 0.5) / w)
    elevations = _elevation_deg(rows, h)
    seen = earth.direction(bearings[None, :], elevations[:, None])
    peak = earth.direction(bearings[col], _elevation_deg(np.array(row), h))
    angle = np.degrees(np.arccos(np.clip(seen @ peak, -1.0, 1.0)))
    near = lum[band]
    ring = np.median(near[(angle > RING_DEG[0]) & (angle < RING_DEG[1])])
    if near.max() < DISC_CONTRAST * ring:
        return Sun(glow, None)
    floor = DISC_FLOOR * float(near.max())
    disc = (angle < DISC_DEG) & (near > floor)
    # Weighted by radiance and by each pixel's solid angle, cos(elevation).
    cosines = np.broadcast_to(np.cos(np.radians(elevations))[:, None], near.shape)
    weight = near[disc] * cosines[disc]
    centre = (seen[disc] * weight[:, None]).sum(axis=0)
    centre /= np.linalg.norm(centre)
    rms_deg = math.sqrt(float((angle[disc] ** 2 * weight).sum() / weight.sum()))
    if not DISC_RMS_DEG[0] <= rms_deg <= DISC_RMS_DEG[1]:
        return Sun(glow, None)
    return Sun(
        math.degrees(math.atan2(centre[0], centre[1])) % 360.0,
        math.degrees(math.asin(centre[2])),
        floor,
    )


def diffuse_irradiance(radiance: np.ndarray) -> float:
    """Irradiance on a horizontal surface from the sky above the horizon, in the image's
    own units.

    Args:
        radiance: Equirectangular sky, (h, w, 3), top row first.
    """
    h, w = radiance.shape[:2]
    elevation = np.radians(_elevation_deg(np.arange(h // 2), h))
    # An equirectangular pixel spans dω = cos(elevation) (π / h) (2π / w).
    weight = np.sin(elevation) * np.cos(elevation) * (2.0 * math.pi**2 / (h * w))
    return float(weight @ (radiance[: h // 2] @ LUMINANCE).sum(axis=1))


def _nearness(
    elevation_deg: np.ndarray, cos_sun: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The horizon's nearness and the sun's, 0 to 1."""
    horizon = (1.0 - np.sin(np.radians(elevation_deg))).astype(np.float32)
    return horizon, ((1.0 + cos_sun) / 2.0).astype(np.float32)


def _terms(elevation_deg: np.ndarray, cos_sun: np.ndarray) -> Iterator[np.ndarray]:
    """Products of powers of the horizon's nearness and the sun's."""
    horizon, sun = _nearness(elevation_deg, cos_sun)
    return (horizon**i * sun**j for i in HORIZON_POWERS for j in SUN_POWERS)


def _fitted(
    coefficients: np.ndarray, elevation_deg: np.ndarray, cos_sun: np.ndarray
) -> np.ndarray:
    """`_terms` weighted by `coefficients` and summed."""
    horizon, sun = _nearness(elevation_deg, cos_sun)
    by = coefficients.reshape(len(HORIZON_POWERS), len(SUN_POWERS))
    per_row = sum(horizon**i * c for i, c in zip(HORIZON_POWERS, by, strict=True))
    per_row = per_row.astype(np.float32)
    return sum(per_row[:, k, None] * sun**j for k, j in enumerate(SUN_POWERS))


def _least_squares(basis: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Least squares by the normal equations, as small as `basis` is wide."""
    b = basis.astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(b.T @ b, b.T @ value, rcond=None)
    return coefficients


def _farthest(away: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the largest `away` in each FIT_STRIDE square."""
    s = FIT_STRIDE
    h, w = away.shape
    padded = np.pad(away, ((0, -h % s), (0, -w % s)), constant_values=-np.inf)
    rows, cols = padded.shape[0] // s, padded.shape[1] // s
    squares = padded.reshape(rows, s, cols, s).swapaxes(1, 2).reshape(rows, cols, -1)
    at = squares.argmax(axis=-1)
    return np.arange(rows)[:, None] * s + at // s, np.arange(cols) * s + at % s


def _envelope(
    away: np.ndarray, side: np.ndarray, elevation_deg: np.ndarray, cos_sun: np.ndarray
) -> np.ndarray:
    """The clear sky's blueness, fitted to the largest `away` in each FIT_STRIDE square.

    Cloud only greys the sky, so the clear sky is the envelope on its side: a gap
    anywhere in a square holds it there. A square's noise pushes the fit out by the top
    of its spread, which the share reads as a trace of cloud on clear sky.

    Args:
        away: Each pixel's blueness times `side`; -inf where it cannot be clear sky.
        side: 1 where the clear sky is bluer than grey, -1 where redder.
        elevation_deg: Each row's elevation, (h, 1).
        cos_sun: Cosine of each pixel's angle from the sun.
    """
    rows, cols = _farthest(away)
    fitted = np.isfinite(away[rows, cols])
    rows, cols = rows[fitted], cols[fitted]
    outward = side[rows, cols]
    seen = outward * away[rows, cols]
    basis = np.stack(list(_terms(elevation_deg[rows, 0], cos_sun[rows, cols])), axis=-1)
    clear_ones = np.ones(seen.shape, bool)
    # The first refit keeps what lies past the fit, so a few gaps in a mostly grey sky
    # pull it off grey before the band can take the grey for clear.
    band = 0.0
    for _ in range(FIT_PASSES):
        coefficients = _least_squares(basis[clear_ones], seen[clear_ones])
        refit = outward * (seen - basis @ coefficients) >= -band
        band = CLEAR_BAND
        if not refit.any() or np.array_equal(refit, clear_ones):
            break
        clear_ones = refit
    return _fitted(coefficients, elevation_deg, cos_sun)


def cloud(radiance: np.ndarray, photo: Photo) -> np.ndarray:
    """How much of each pixel is cloud: 0 as blue as the photo's clear sky there, 1 as
    grey as B = R, and none below the horizon.

    A pixel as blue as the clear sky but darker than the clear sky beside it is a
    shaded cloud base, lit by the blue sky: DARK ramps it in, added to the colour's.

    Limitation: a cloud as white as the circumsolar sky beside it reads clear, a disc's
    aureole takes the cloud round it, a glow reads red only where the first fit lies
    under grey, a cloud greyer than a red glow and as bright as it reads clear, a
    glow redder than its fit can follow reads cloud, a shaded base bluer than the glow
    round it reads clear and the glow cloud, a gap in an overcast photo reads cloud,
    haze darker than the clear sky beside it reads cloud, a shaded base much wider than
    BESIDE_DEG reads cloud only at its rim, and a deck no darker than the clear sky
    fitted under it reads only as far as it is grey.

    Args:
        radiance: The photo's linear RGB, (h, w, 3), top row first.
        photo: Its sun places the aureole; an overcast one is cloud throughout.
    """
    h, w = radiance.shape[:2]
    out = np.zeros((h, w), np.float32)
    sky = slice(0, h // 2)
    if photo.overcast:
        out[sky] = 1.0
        return out
    counts = np.maximum(radiance[sky][..., ::2], 0.0) ** (1 / GAMMA)
    red, blue = counts[..., 0], counts[..., 1]
    blueness = (blue - red) / np.maximum(blue + red, 1e-6)
    elevation = _elevation_deg(np.arange(h // 2), h)[:, None]
    bearing = _bearing_deg((np.arange(w) + 0.5) / w)[None, :]
    disc = photo.sun_elevation_deg is not None
    # A judgement: with no disc, the glow's bearing at the horizon.
    sun_elevation_deg = photo.sun_elevation_deg if disc else 0.0
    toward = earth.direction(
        np.array(photo.sun_bearing_deg), np.array(sun_elevation_deg)
    )
    cos_sun = earth.direction(bearing, elevation) @ toward
    aureole = cos_sun > math.cos(math.radians(AUREOLE_DEG)) if disc else None
    clear = _envelope(blueness, np.ones_like(blueness), elevation, cos_sun)
    # A dusk's glow is redder than grey and cloud greys it toward blue, so there the
    # clear sky is each square's reddest pixel.
    if (clear < 0.0).any():
        side = np.where(clear < 0.0, -1.0, 1.0).astype(np.float32)
        away = side * blueness
        clear = _envelope(np.where(away > 0.0, away, -np.inf), side, elevation, cos_sun)
    share = np.clip((clear - blueness) / np.maximum(clear, GREY_CLEAR), 0.0, 1.0)
    # Colour and darkness are two ways off the clear sky: either, or part of both.
    share = np.minimum(share + _darkness(radiance[sky], share), 1.0).astype(np.float32)
    out[sky] = share if aureole is None else _filled(share, aureole)
    return out


def _darkness(radiance: np.ndarray, share: np.ndarray) -> np.ndarray:
    """How much darker each pixel is than the clear sky beside it, 0 to 1 over DARK.

    Args:
        radiance: The sky above the horizon, (h, w, 3), top row first.
        share: Cloud share by colour; what lies under CLEAR_SHARE is the clear sky
            judged against.
    """
    log_lum = np.log(np.maximum(radiance @ LUMINANCE, 1e-9)).astype(
        np.float32, copy=False
    )
    clear = (share < CLEAR_SHARE).astype(np.float32)
    if not clear.any():
        return np.zeros(share.shape, np.float32)
    sigma_px = BESIDE_DEG * radiance.shape[1] / 360.0
    # Shaded cloud read clear by colour leaves the clear set until none is DARK[0]
    # darker than the rest beside it.
    for _ in range(FIT_PASSES):
        beside = _beside(clear, log_lum, sigma_px)
        kept = clear * (log_lum - beside >= math.log(DARK[0]))
        if not kept.any() or np.array_equal(kept, clear):
            break
        clear = kept
    else:
        beside = _beside(clear, log_lum, sigma_px)
    start, full = -math.log(DARK[0]), -math.log(DARK[1])
    return np.clip((beside - log_lum - start) / (full - start), 0.0, 1.0)


def _beside(weight: np.ndarray, value: np.ndarray, sigma_px: float) -> np.ndarray:
    """`value` averaged round each pixel over `weight`, a Gaussian of `sigma_px`."""
    h, w = weight.shape
    f = max(1, int(sigma_px / CELLS_PER_SIGMA))
    # OpenCV's area resize is fast only by a whole factor; the margin padded to one
    # weighs nothing.
    margin = ((0, -h % f), (0, -w % f))
    weight, value = np.pad(weight, margin), np.pad(value, margin)
    whole = weight.shape[::-1]
    size = (whole[0] // f, whole[1] // f)
    cells = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
    sums = cv2.resize(weight * value, size, interpolation=cv2.INTER_AREA)
    mean = _pull_push(cells, sums / np.maximum(cells, 1e-12), sigma_px / f)
    return cv2.resize(mean, whole, interpolation=cv2.INTER_LINEAR)[:h, :w]


def _pull_push(weight: np.ndarray, value: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized convolution (Knutsson & Westin, "Normalized and differential
    convolution", CVPR 1993) at doubling scales, each finer one trusted as far as it
    holds a weight of one cell, the rest pushed from the coarser (Gortler et al.,
    "The Lumigraph", SIGGRAPH 1996).
    """
    h, w = weight.shape
    # Doublings until a sigma spans the strip's circumference.
    levels = max(0, math.ceil(math.log2(w / (2.0 * math.pi * sigma)))) + 1
    mean = value
    for k in reversed(range(levels)):
        size = (max(1, math.ceil(w / 2**k)), max(1, math.ceil(h / 2**k)))
        cells = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
        sums = cv2.resize(weight * value, size, interpolation=cv2.INTER_AREA)
        total = _wrapped_blur(cells, sigma)
        level = _wrapped_blur(sums, sigma) / np.maximum(total, 1e-12)
        level = cv2.resize(level, (w, h), interpolation=cv2.INTER_LINEAR)
        if k == levels - 1:
            # Nothing coarser to push from.
            mean = level
            continue
        # 2 pi sigma^2 turns a unit-sum Gaussian's weight into cells.
        trust = np.minimum(1.0, total * 2.0 * math.pi * sigma**2)
        trust = cv2.resize(trust, (w, h), interpolation=cv2.INTER_LINEAR)
        mean = trust * level + (1.0 - trust) * mean
    return mean


def _wrapped_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of an equirectangular strip, wrapped in bearing."""
    # OpenCV's float kernel reaches 4 sigma.
    pad = min(math.ceil(4.0 * sigma), a.shape[1])
    wrapped = cv2.copyMakeBorder(a, 0, 0, pad, pad, cv2.BORDER_WRAP)
    return cv2.GaussianBlur(wrapped, (0, 0), sigma)[:, pad:-pad]


def _filled(share: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """A cloud share with a hole carried in from round it (Bertalmio, Bertozzi &
    Sapiro, "Navier-Stokes, fluid dynamics, and image and video inpainting", CVPR
    2001). OpenCV's Telea overshoots on float images.

    Args:
        share: Cloud share, (h, w) float32, a type OpenCV's inpaint takes.
        hole: True where the share is replaced.
    """
    return cv2.inpaint(
        share,
        hole.astype(np.uint8),
        max(1, round(INPAINT_RADIUS_DEG * share.shape[1] / 360.0)),
        cv2.INPAINT_NS,
    )
