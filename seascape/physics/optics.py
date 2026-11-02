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
    """The share of a point's Airy pattern that lands on each pixel of a grid of
    `pitch_m` round the one it is centred on, one channel per wavelength, each summing
    to 1."""
    # Four Airy-disc diameters, 2.44 lambda N each to the first dark ring (Goodman
    # 2005), at the longest wavelength.
    half = math.ceil(4 * 2.44 * max(wavelengths_m) * f_number / pitch_m)
    # Twice the kernel's width, so the spread past it wraps onto nothing kept.
    n = 4 * half + 2
    # Odd, so a pixel's subsamples centre on it, and enough that the subgrid's Nyquist
    # passes the shortest wavelength's cutoff: no spread is cut.
    sub = 2 * math.ceil(pitch_m / (min(wavelengths_m) * f_number)) + 1
    freq = np.fft.fftfreq(n * sub, pitch_m / sub)
    otf = pupil_mtf(
        np.hypot(freq[:, None], freq[None, :])[..., None], wavelengths_m, f_number
    )
    spread = np.fft.ifft2(otf, axes=(0, 1)).real
    # Subsample 0 is the point; roll each pixel's subsamples into one block to sum.
    spread = np.roll(spread, sub // 2, axis=(0, 1))
    pixels = spread.reshape(n, sub, n, sub, -1).sum(axis=(1, 3))
    pixels = np.fft.fftshift(pixels, axes=(0, 1))
    kept = pixels[n // 2 - half : n // 2 + half + 1, n // 2 - half : n // 2 + half + 1]
    return kept / kept.sum(axis=(0, 1))
