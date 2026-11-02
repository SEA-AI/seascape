"""A lens's diffraction."""

import math

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


def pupil_mtf(
    freq_per_m: npt.ArrayLike, wavelength_m: npt.ArrayLike, f_number: float
) -> FloatArray:
    """The incoherent OTF of a clear circular pupil at `freq_per_m` in the image plane,
    cut off at 1 / (lambda N) (Goodman, Introduction to Fourier Optics, 2005, sec.
    6.3). Inputs broadcast."""
    x = np.minimum(np.asarray(freq_per_m) * np.asarray(wavelength_m) * f_number, 1.0)
    return 2 / math.pi * (np.arccos(x) - x * np.sqrt(1.0 - x * x))


def airy_kernel(
    wavelengths_m: tuple[float, ...], f_number: float, pitch_m: float
) -> FloatArray:
    """The share of a point's Airy pattern landing on each pixel of a `pitch_m` grid
    centred on the point's pixel. One channel per wavelength; each sums to 1."""
    # Four Airy discs each side at the longest wavelength; a disc is 2.44 lambda N
    # across (Goodman 2005).
    half = math.ceil(4 * 2.44 * max(wavelengths_m) * f_number / pitch_m)
    # Twice the kernel's width, so the spread past it wraps onto nothing kept.
    n = 4 * half + 2
    # The pattern is sampled finer than the pixel, so the OTF is never cut short of
    # the shortest wavelength's cutoff.
    sub = math.ceil(2 * pitch_m / (min(wavelengths_m) * f_number)) + 1
    fy, fx = np.meshgrid(*[np.fft.fftfreq(n * sub, pitch_m / sub)] * 2, indexing="ij")
    otf = pupil_mtf(np.hypot(fx, fy)[..., None], wavelengths_m, f_number)
    # Times a pixel's transfer function, sampled at pixel centres, is the pattern
    # integrated over each pixel.
    box = np.sinc(fx * pitch_m) * np.sinc(fy * pitch_m)
    spread = np.fft.ifft2(otf * box[..., None], axes=(0, 1)).real
    pixels = np.fft.fftshift(spread[::sub, ::sub] * sub**2, axes=(0, 1))
    kept = pixels[n // 2 - half : n // 2 + half + 1, n // 2 - half : n // 2 + half + 1]
    return kept / kept.sum(axis=(0, 1))
