"""A thermal frame as a camera writes one: 16-bit counts, or 8-bit grey after AGC."""

import math
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from seascape.config import Agc as Setting

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


type Layers = tuple[np.ndarray, np.ndarray | None]


def transfer(frames: Sequence[np.ndarray], setting: Setting) -> np.ndarray:
    """The grey of every 16-bit count, for these frames together."""
    return _transfer([_layers(t_k, setting) for t_k in frames], setting)


def grey(t_k: np.ndarray, curve: np.ndarray, setting: Setting) -> np.ndarray:
    """8-bit grey through a `transfer` curve."""
    return _grey(_layers(t_k, setting), curve, setting)


def _layers(t_k: np.ndarray, setting: Setting) -> Layers:
    """A frame in counts, split into background and detail where the AGC splits it."""
    c = counts(t_k)
    if not setting.information_based:
        return c, None
    # A step well past `smoothing` stays in the background.
    c = c.astype(np.float32)
    low = cv2.bilateralFilter(c, SMOOTHING_PX, setting.smoothing, SMOOTHING_PX)
    return low, c - low


def _transfer(frames: list[Layers], setting: Setting) -> np.ndarray:
    c = np.concatenate([np.rint(low).ravel() for low, _ in frames]).astype(np.int64)
    weights = np.ones(c.size)
    details = [high for _, high in frames if high is not None]
    if details:
        # Judgement: the Boson+ datasheet weights detail pixels up (§6.8.1) without
        # saying by how much; here by their detail against the frames' mean, so
        # frames of even texture equalise as plateau does.
        detail = np.concatenate([np.abs(high).ravel() for high in details])
        if (mean := detail.mean()) > 0:
            weights += detail / mean
    return _plateau(c, weights, setting)


def _plateau(c: np.ndarray, weights: np.ndarray, setting: Setting) -> np.ndarray:
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
    # breakpoint the m steepest are clipped, and the fill is linear in between. A
    # count the frame lacks has no slope to clip, and would make the fill 0 * inf.
    steepest = np.sort(slope[slope > 0])[::-1]
    gains = setting.max_gain / steepest
    filled = setting.max_gain * np.arange(1, steepest.size + 1) + gains * (
        steepest.sum() - np.cumsum(steepest)
    )
    gain = np.interp(255, np.r_[0, filled], np.r_[0, gains])
    slope = np.minimum(gain * slope, setting.max_gain)
    # Each count at the middle of its step, the curve centred on 127.5.
    level = 127.5 + np.cumsum(slope) - slope / 2 - slope.sum() / 2
    level = 255.0 * np.clip(level / 255.0, 0.0, 1.0) ** (1.0 / setting.ace)
    return np.interp(np.arange(COUNTS), low + np.arange(level.size), level)


def _grey(frame: Layers, curve: np.ndarray, setting: Setting) -> np.ndarray:
    """The background takes the curve, squeezed inside the detail headroom; detail
    goes on top at the curve's mean slope over the frame, times DDE."""
    low, high = frame
    at = np.clip(np.rint(low), 0, COUNTS - 1).astype(np.int64)
    if high is None:
        return np.rint(curve[at]).astype(np.uint8)
    squeeze = (255 - 2 * setting.detail_headroom) / 255
    background = setting.detail_headroom + squeeze * curve[at]
    # One gain for the whole frame, as FLIR's detail-to-background ratio names it: the
    # slope over the counts the frame holds, each once. At the local slope a smooth sea,
    # steep where it fills the histogram, would gain its own grain twice; over the
    # frame's span, a small hot exhaust would take the scene's detail away.
    gain = np.gradient(curve)[np.unique(at)].mean()
    detail = setting.dde * squeeze * gain * high
    return np.rint(np.clip(background + detail, 0, 255)).astype(np.uint8)


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
        frame = _layers(t_k, self._setting)
        curve = _transfer([frame], self._setting)
        if self._curve is not None:
            curve = self._keep * self._curve + (1 - self._keep) * curve
        self._curve = curve
        return _grey(frame, curve, self._setting)
