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
Tech. 11 5549, 2018, doi:10.5194/amt-11-5549-2018) fit their clear sky to the pixels
near it and refit; the clear-sky library of Ghonima et al. (Atmos. Meas. Tech. 5 2881,
2012, doi:10.5194/amt-5-2881-2012) indexes it by zenith and angle from the sun. An
overcast photo has no clear sky to fit; the library says which photos are.
"""

import math
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from pydantic import ConfigDict, Field, model_validator

from seascape.assets import SHA256
from seascape.model import Model

LIBRARY = Path(__file__).parent / "skies.toml"
# ITU-R BT.709.
LUMINANCE = np.array([0.2126, 0.7152, 0.0722], np.float32)
# Judgements: the ring round the peak, and how much brighter than it the peak must be,
# between an unclipped sun's contrast and that of a sun behind cloud or below the
# horizon.
DISC_CONTRAST = 100.0
RING_DEG = (2.0, 5.0)
# Judgements: how far from the peak the disc reaches, and its floor as a fraction of
# the peak.
DISC_DEG = 1.0
DISC_FLOOR = 0.1
# A judgement round the disc's RMS radius: a star is one pixel, a glow degrees wide.
DISC_RMS_DEG = (0.1, 0.6)
# The grid a glow is found on, and its blur in cells: a judgement, a few degrees.
GLOW_CELLS = (128, 256)
GLOW_BLUR = 5
# sRGB's approximate display gamma (IEC 61966-2-1): counts = radiance ** (1 / GAMMA).
GAMMA = 2.2
# Judgements: how far under the fitted clear sky a pixel's blueness may fall and still
# be refitted as clear; the clear blueness under which a sky is too grey to show a cloud
# by colour; the passes the fit takes at most to settle.
CLEAR_BAND = 0.03
GREY_CLEAR = 0.05
FIT_PASSES = 10
# Every FIT_STRIDE-th pixel each way is fitted: the clear sky is smooth.
FIT_STRIDE = 4
# A judgement: powers of the horizon's nearness and the sun's, steep enough to follow a
# haze at the horizon and a glow a few degrees round the sun.
HORIZON_POWERS = (0, 1, 2, 4, 8)
SUN_POWERS = (0, 1, 2, 4, 8, 16)
# Judgements: the aureole round a disc, as white as cloud and too steep to fit, and the
# neighbourhood each of its pixels is filled from.
AUREOLE_DEG = 8.0
INPAINT_RADIUS_PX = 3


class Photo(Model):
    """One sky, and its sun as the image holds it: a bearing clockwise from +Y as
    Cycles maps the image unrotated, and an elevation and the disc's `Sun.floor` only
    where a disc shows."""

    model_config = ConfigDict(frozen=True)

    url: str
    sha256: str = Field(pattern=SHA256)
    sun_bearing_deg: float = Field(ge=0.0, lt=360.0)
    sun_elevation_deg: float | None = Field(default=None, gt=-90.0, le=90.0)
    sun_floor: float | None = Field(default=None, gt=0.0)
    # Labelled by eye: no clear sky shows, so none can be fitted.
    overcast: bool = False
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)

    @model_validator(mode="after")
    def _a_disc_has_a_floor(self) -> "Photo":
        if (self.sun_elevation_deg is None) != (self.sun_floor is None):
            raise ValueError("sun_elevation_deg and sun_floor come together")
        return self

    @property
    def sun(self) -> str:
        elevation = self.sun_elevation_deg
        return "no disc" if elevation is None else f"{elevation:.1f} deg"


@cache
def library() -> dict[str, Photo]:
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


def _directions(bearing_deg: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    b, e = np.radians(bearing_deg), np.radians(elevation_deg)
    return np.stack(
        np.broadcast_arrays(np.cos(e) * np.sin(b), np.cos(e) * np.cos(b), np.sin(e)), -1
    )


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
    seen = _directions(bearings[None, :], elevations[:, None])
    peak = _directions(bearings[col], _elevation_deg(np.array(row), h))
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


def _clear_basis(elevation_deg: np.ndarray, cos_sun: np.ndarray) -> np.ndarray:
    """Products of powers of the horizon's nearness and the sun's, (..., terms)."""
    horizon = 1.0 - np.sin(np.radians(elevation_deg))
    sun = (1.0 + cos_sun) / 2.0
    return np.stack(
        [horizon**i * sun**j for i in HORIZON_POWERS for j in SUN_POWERS], axis=-1
    ).astype(np.float32)


def cloud(radiance: np.ndarray, photo: Photo) -> np.ndarray:
    """How much of each pixel of `radiance`, (h, w, 3) of `photo` top row first, is
    cloud: 0 as blue as the photo's clear sky there, 1 as grey as B = R, and none below
    the horizon.

    Limitation: colour alone, so a cloud as white as the circumsolar sky beside it reads
    clear, a disc's aureole takes the cloud round it, a dusk's glow redder than grey
    reads cloud, and a gap in an overcast photo reads cloud.
    """
    h, w = radiance.shape[:2]
    out = np.zeros((h, w), np.float32)
    sky = slice(0, h // 2)
    if photo.overcast:
        out[sky] = 1.0
        return out
    counts = np.maximum(radiance[sky], 0.0) ** (1.0 / GAMMA)
    red, blue = counts[..., 0], counts[..., 2]
    blueness = (blue - red) / np.maximum(blue + red, 1e-6)
    elevation = _elevation_deg(np.arange(h // 2), h)[:, None]
    bearing = _bearing_deg((np.arange(w) + 0.5) / w)[None, :]
    # A judgement: with no disc, the glow's bearing at the horizon.
    toward = _directions(
        np.array(photo.sun_bearing_deg), np.array(photo.sun_elevation_deg or 0.0)
    )
    cos_sun = _directions(bearing, elevation) @ toward
    basis = _clear_basis(elevation, cos_sun)
    sampled = np.zeros(blueness.shape, bool)
    sampled[::FIT_STRIDE, ::FIT_STRIDE] = True
    aureole = np.zeros(blueness.shape, bool)
    if photo.sun_elevation_deg is not None:
        aureole = cos_sun > math.cos(math.radians(AUREOLE_DEG))
    clear_ones = sampled
    for _ in range(FIT_PASSES):
        coefficients, *_ = np.linalg.lstsq(
            basis[clear_ones], blueness[clear_ones], rcond=None
        )
        clear = basis @ coefficients
        refit = sampled & (blueness >= clear - CLEAR_BAND)
        if np.array_equal(refit, clear_ones):
            break
        clear_ones = refit
    cloud = np.clip((clear - blueness) / np.maximum(clear, GREY_CLEAR), 0.0, 1.0)
    if aureole.any():
        # Bertalmio, Bertozzi & Sapiro, "Navier-Stokes, fluid dynamics, and image and
        # video inpainting", CVPR 2001: the cloud round the aureole, carried in.
        # OpenCV's Telea overshoots on float images.
        cloud = cv2.inpaint(
            cloud.astype(np.float32),
            aureole.astype(np.uint8),
            INPAINT_RADIUS_PX,
            cv2.INPAINT_NS,
        )
    out[sky] = cloud
    return out
