"""An uncooled LWIR camera's lens and focal plane, on the radiance Cycles renders.

Blender has no 8-14 um band, so diffraction, which scales with wavelength, and noise
quoted in kelvin act here, between the float render and `agc`.

ponytail: no bolometer lag; the Boson's time constant is unpublished.
"""

import math

import numpy as np
import numpy.typing as npt

from seascape.config import Camera
from seascape.physics import lwir

type FloatArray = npt.NDArray[np.float64]

_BAND_LAM = np.linspace(*lwir.BAND_M, 64)
# The background a NETD is quoted against: 30 C (Teledyne FLIR, Boson datasheet
# 102-2013-40, sec. 11.1, https://flir.netx.net/file/asset/52701/original/attachment/).
NETD_BACKGROUND_K = 303.15


def diffraction_mtf(
    freq_per_m: npt.ArrayLike, f_number: float, t_k: float
) -> FloatArray:
    """A circular aperture's MTF at `freq_per_m` in the image plane, over the band.

    Each wavelength's is the incoherent OTF of a clear circular pupil, cut off at
    1 / (lambda N) (Goodman, Introduction to Fourier Optics, 2005, sec. 6.3); the band
    weights them by the radiance a blackbody at `t_k` puts at each, as a flat detector
    sees it (Boreman, Modulation Transfer Function in Optical and Electro-Optical
    Systems, 2001).
    """
    freq = np.asarray(freq_per_m, dtype=np.float64)
    x = np.minimum(freq[..., None] * _BAND_LAM * f_number, 1.0)
    mono = 2 / math.pi * (np.arccos(x) - x * np.sqrt(1.0 - x * x))
    weight = lwir.planck(_BAND_LAM, t_k)
    return (mono * weight).sum(axis=-1) / weight.sum()


def diffract(
    radiance: FloatArray, f_number: float, pitch_m: float, t_k: float
) -> FloatArray:
    """`radiance` through the lens's diffraction, the frame mirrored at its edges."""
    # Four Airy-disc diameters, 2.44 lambda N each to the first dark ring (Goodman
    # 2005), at the band's longest.
    pad = math.ceil(4 * 2.44 * lwir.BAND_M[1] * f_number / pitch_m)
    padded = np.pad(radiance, pad, mode="reflect")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.rfftfreq(padded.shape[1])[None, :]
    mtf = diffraction_mtf(np.hypot(fx, fy) / pitch_m, f_number, t_k)
    out = np.fft.irfft2(np.fft.rfft2(padded) * mtf, s=padded.shape)
    return out[pad:-pad, pad:-pad]


def falloff(shape: tuple[int, int], hfov_rad: float) -> FloatArray:
    """Natural vignetting, cos^4 of each pixel centre's field angle (Kerr 2007)."""
    height, width = shape
    focal_px = width / 2 / math.tan(hfov_rad / 2)
    y = np.arange(height) + 0.5 - height / 2
    x = np.arange(width) + 0.5 - width / 2
    tan2 = (x[None, :] ** 2 + y[:, None] ** 2) / focal_px**2
    return (1.0 + tan2) ** -2


def seen(
    radiance: FloatArray, camera: Camera, noise: np.random.Generator
) -> FloatArray:
    """The brightness temperature `camera` reads off a rendered `radiance` frame."""
    bolometer = camera.microbolometer
    if camera.lens.vignetting:
        radiance = radiance * falloff(radiance.shape, math.radians(camera.hfov_deg))
    if bolometer.f_number is not None and bolometer.pitch_um is not None:
        # Brightness temperature rises with radiance, so it keeps the median.
        median_k = float(lwir.brightness_temperature(np.median(radiance)))
        radiance = diffract(
            radiance, bolometer.f_number, bolometer.pitch_um * 1e-6, median_k
        )
    if bolometer.netd_k > 0:
        # The noise is a radiance, so it reads larger in kelvin where the scene is
        # colder: a NETD varies with dW/dT (Boson datasheet sec. 11.1).
        sigma = lwir.band_radiance(NETD_BACKGROUND_K + bolometer.netd_k)
        sigma -= lwir.band_radiance(NETD_BACKGROUND_K)
        radiance = radiance + sigma * noise.standard_normal(radiance.shape)
    return lwir.brightness_temperature(radiance)
