"""Photographed skies, and where each one's sun is, measured from its pixels.

Sources
-------
Poly Haven's pure skies. Its FAQ (https://docs.polyhaven.com/en/faq): "All our HDRIs
are unclipped", so a sun keeps its radiance. Hold-Geoffroy, Sunkavalli, Hadap,
Gambaretto & Lalonde, "Deep outdoor illumination estimation", CVPR 2017
(arXiv:1611.06403): an HDR panorama's sun is its brightest region. The Astronomical
Almanac: the sun's mean semi-diameter is 16 arcminutes, so a uniform disc has an RMS
radius of 0.19 degrees.
"""

import math
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from pydantic import ConfigDict, Field

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


class Photo(Model):
    """One sky, and its sun as the image holds it: a bearing clockwise from +Y as
    Cycles maps the image unrotated, and an elevation only where a disc shows."""

    model_config = ConfigDict(frozen=True)

    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sun_bearing_deg: float = Field(ge=0.0, lt=360.0)
    sun_elevation_deg: float | None = Field(default=None, gt=-90.0, le=90.0)
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)


@cache
def library() -> dict[str, Photo]:
    with LIBRARY.open("rb") as handle:
        return {name: Photo(**body) for name, body in tomllib.load(handle).items()}


@dataclass(frozen=True)
class Sun:
    bearing_deg: float
    elevation_deg: float | None


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
    disc = (angle < DISC_DEG) & (near > DISC_FLOOR * near.max())
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
    )
