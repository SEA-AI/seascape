"""A thermal frame as a camera writes one: 16-bit counts, or 8-bit grey after AGC.

The AGC is the plateau equalisation of Teledyne FLIR, Boson+ Product Datasheet
102-2013-45, §6.8, at its factory defaults. Tail rejection defaults to 0 % and is
left out, as are ACE, DDE+ and Information-Based Equalization, which the datasheet
names but does not define.
"""

import math
from pathlib import Path

import cv2
import numpy as np

# The unit radiometric thermal cameras write, so real and rendered frames read alike.
CENTIKELVIN = 100
# Boson+ datasheet §6.8.2: no histogram bin holds more than this fraction of the
# pixels. Judgement: a bin is one count, as the datasheet gives no bin width.
PLATEAU = 0.10
# Boson+ datasheet §6.8.5: the share of the transfer function that is the frame's
# min-max line, the rest its clipped histogram.
LINEAR = 0.20
# Boson+ datasheet §6.8.4: the steepest the transfer function may be, in grey levels
# per count. FLIR Camera Adjustments 102-2013-100-01 §2.2 gives the unit: 200 counts
# need a gain of about 1.25 to fill 255. Judgement: a count is a centikelvin, the
# counts this module writes; a Boson's raw counts per kelvin are not published.
MAX_GAIN = 1.65
# Boson+ datasheet §6.8.10 keeps 85 % of the transfer function each frame, and §5.1
# runs the pipeline at 60 Hz by default: about 0.1 s.
TAU_S = -1 / 60 / math.log(0.85)


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


def transfer(t_k: np.ndarray) -> np.ndarray:
    """The grey of every 16-bit count, equalised to the frame's histogram.

    Each count's slope is the blend of its clipped histogram share and the min-max
    line, scaled up until the curve fills 0-255 but never steeper than `MAX_GAIN`.
    A curve that cannot fill the range, as a bland frame's cannot, is centred on it.
    """
    c = counts(t_k).ravel()
    low = int(c.min())
    hist = np.minimum(np.bincount(c - low), PLATEAU * c.size)
    slope = LINEAR / hist.size + (1 - LINEAR) * hist / hist.sum()
    # The gain that fills 255 once the steepest counts sit at MAX_GAIN: at each
    # breakpoint the m steepest are clipped, and the fill is linear in between.
    steepest = np.sort(slope)[::-1]
    gains = MAX_GAIN / steepest
    filled = MAX_GAIN * np.arange(1, steepest.size + 1) + gains * (
        steepest.sum() - np.cumsum(steepest)
    )
    gain = np.interp(255, np.r_[0, filled], np.r_[0, gains])
    slope = np.minimum(gain * slope, MAX_GAIN)
    # Each count at the middle of its step, the curve centred on 127.5.
    level = 127.5 + np.cumsum(slope) - slope / 2 - slope.sum() / 2
    return np.interp(
        np.arange(np.iinfo(np.uint16).max + 1), low + np.arange(level.size), level
    )


def grey(t_k: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """8-bit grey through a `transfer` curve."""
    return np.rint(curve[counts(t_k)]).astype(np.uint8)


class Agc:
    """8-bit grey from one camera's frames in order, its transfer function damped over
    time as a thermal camera damps its AGC, so a sequence does not flicker.

    A still is a sequence of one. Without a step between frames, each frame takes its
    own transfer function.
    """

    def __init__(self, step_s: float = math.inf) -> None:
        self._keep = math.exp(-step_s / TAU_S)
        self._curve: np.ndarray | None = None

    def __call__(self, t_k: np.ndarray) -> np.ndarray:
        curve = transfer(t_k)
        if self._curve is not None:
            curve = self._keep * self._curve + (1 - self._keep) * curve
        self._curve = curve
        return grey(t_k, curve)
