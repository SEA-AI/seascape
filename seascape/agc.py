"""A thermal frame as a camera writes one: 16-bit counts, or 8-bit grey after AGC."""

import math
from pathlib import Path

import cv2
import numpy as np

# The unit radiometric thermal cameras write, so real and rendered frames read alike.
CENTIKELVIN = 100
# A display choice, not physics: slow enough not to flash when a hot target enters,
# fast enough to follow it.
TAU_S = 1.0


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


def grey(t_k: np.ndarray, low_k: float, high_k: float) -> np.ndarray:
    """8-bit grey, black at `low_k` and white at `high_k`."""
    # One temperature throughout has no contrast to stretch; mid-grey, not NaN.
    if high_k - low_k < 1e-6:
        low_k, high_k = low_k - 0.5, low_k + 0.5
    scaled = np.clip((t_k - low_k) / (high_k - low_k), 0.0, 1.0)
    return np.rint(255 * scaled).astype(np.uint8)


class Agc:
    """8-bit grey from one camera's frames in order, its span damped over time as a
    thermal camera damps its AGC, so a sequence does not flicker.

    A still is a sequence of one. Without a step between frames, each frame takes its
    own span.
    """

    def __init__(self, step_s: float = math.inf) -> None:
        self._keep = math.exp(-step_s / TAU_S)
        self._span: tuple[float, float] | None = None

    def __call__(self, t_k: np.ndarray) -> np.ndarray:
        # Full span: a target is a small fraction of the frame, and trimming the tails
        # flattens it to white.
        low, high = float(t_k.min()), float(t_k.max())
        if self._span is not None:
            keep = self._keep
            low = keep * self._span[0] + (1 - keep) * low
            high = keep * self._span[1] + (1 - keep) * high
        self._span = low, high
        return grey(t_k, low, high)
