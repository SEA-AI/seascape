"""A thermal frame as a camera writes one: 16-bit counts, or 8-bit grey after AGC."""

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from seascape.config import Agc as Setting
from seascape.config import Equalization, InformationBased, Linear

# The unit radiometric thermal cameras write, so real and rendered frames read alike.
CENTIKELVIN = 100
COUNTS = np.iinfo(np.uint16).max + 1
# Judgement: FLIR publishes no spatial extent for the Boson's detail filter.
SMOOTHING_PX = 5


def counts(t_k: np.ndarray) -> np.ndarray:
    """Temperatures as 16-bit centikelvin."""
    hottest_k = np.iinfo(np.uint16).max / CENTIKELVIN
    if t_k.max() > hottest_k:
        raise ValueError(f"{t_k.max():.1f} K overflows 16-bit centikelvin")
    return np.rint(t_k * CENTIKELVIN).astype(np.uint16)


def kelvin(path: Path) -> np.ndarray | None:
    """The temperatures in a 16-bit centikelvin frame, or None for any other image."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint16:
        return None
    return image / CENTIKELVIN


def transfer(frames: Sequence[np.ndarray], setting: Setting) -> np.ndarray:
    """The grey of every 16-bit count, for these frames together."""
    if isinstance(setting, Linear):
        c = np.concatenate([counts(t_k).ravel() for t_k in frames])
        return _line(int(c.min()), int(c.max()))
    if isinstance(setting, InformationBased):
        split = [_split(t_k, setting) for t_k in frames]
        c = np.concatenate([np.rint(low).ravel() for low, _ in split]).astype(np.int64)
        # Judgement: FLIR weights detail pixels up (§6.8.1) without saying by how much;
        # here by their detail against the frame's mean, so a frame of even texture
        # equalises as plateau does.
        detail = np.concatenate([np.abs(high).ravel() for _, high in split])
        mean = detail.mean()
        weights = 1 + detail / mean if mean > 0 else np.ones_like(detail)
        return _plateau(c, weights, setting)
    c = np.concatenate([counts(t_k).ravel() for t_k in frames]).astype(np.int64)
    return _plateau(c, np.ones(c.size), setting)


def _line(low: int, high: int) -> np.ndarray:
    # One temperature throughout has no contrast to stretch; mid-grey, not NaN.
    if high == low:
        return np.full(COUNTS, 127.5)
    return np.clip(255 * (np.arange(COUNTS) - low) / (high - low), 0.0, 255.0)


def _plateau(c: np.ndarray, weights: np.ndarray, setting: Equalization) -> np.ndarray:
    """Each count's slope is the blend of its clipped histogram share and the linear
    ramp, scaled up until the curve fills 0-255 but never steeper than the max gain.
    A curve that cannot fill the range, as a bland frame's cannot, is centred on it."""
    low = int(c.min())
    hist = np.bincount(c - low, weights=weights)
    hist = np.minimum(hist, setting.plateau * weights.sum())
    slope = (
        setting.linear_fraction / hist.size
        + (1 - setting.linear_fraction) * hist / hist.sum()
    )
    # The gain that fills 255 once the steepest counts sit at the cap: at each
    # breakpoint the m steepest are clipped, and the fill is linear in between.
    steepest = np.sort(slope)[::-1]
    gains = setting.max_gain / steepest
    filled = setting.max_gain * np.arange(1, steepest.size + 1) + gains * (
        steepest.sum() - np.cumsum(steepest)
    )
    gain = np.interp(255, np.r_[0, filled], np.r_[0, gains])
    slope = np.minimum(gain * slope, setting.max_gain)
    # Each count at the middle of its step, the curve centred on 127.5.
    level = 127.5 + np.cumsum(slope) - slope / 2 - slope.sum() / 2
    return np.interp(np.arange(COUNTS), low + np.arange(level.size), level)


def _split(t_k: np.ndarray, setting: InformationBased) -> tuple[np.ndarray, np.ndarray]:
    """A frame's background and detail, in counts, as an edge-preserving filter
    separates them: a step past `smoothing` stays in the background."""
    c = counts(t_k).astype(np.float32)
    low = cv2.bilateralFilter(c, SMOOTHING_PX, setting.smoothing, SMOOTHING_PX)
    return low, c - low


def grey(t_k: np.ndarray, curve: np.ndarray, setting: Setting) -> np.ndarray:
    """8-bit grey through a `transfer` curve. Information-based, the background takes
    the curve and the detail is added on top at the curve's slope there, times DDE."""
    if not isinstance(setting, InformationBased):
        return np.rint(curve[counts(t_k)]).astype(np.uint8)
    low, high = _split(t_k, setting)
    at = np.clip(np.rint(low), 0, COUNTS - 1).astype(np.int64)
    value = curve[at] + setting.dde * np.gradient(curve)[at] * high
    return np.rint(np.clip(value, 0.0, 255.0)).astype(np.uint8)


class Agc:
    """8-bit grey from one camera's frames in order, its transfer curve damped over
    time as a thermal camera damps its AGC, so a sequence does not flicker.

    A still is a sequence of one. Without a step between frames, each frame takes its
    own curve.
    """

    def __init__(self, setting: Setting, step_s: float = math.inf) -> None:
        self._setting = setting
        self._keep = math.exp(-step_s / setting.tau_s)
        self._curve: np.ndarray | None = None

    def __call__(self, t_k: np.ndarray) -> np.ndarray:
        curve = transfer([t_k], self._setting)
        if self._curve is not None:
            curve = self._keep * self._curve + (1 - self._keep) * curve
        self._curve = curve
        return grey(t_k, curve, self._setting)
