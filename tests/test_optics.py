"""A lens's diffraction."""

import math

import pytest

from seascape.physics import optics


def test_a_fine_pixel_takes_the_airy_peak_over_its_area() -> None:
    """A unit point's Airy peak is pi / (4 lambda^2 N^2) (Born & Wolf, Principles of
    Optics, sec. 8.5.2); a pixel far finer than the disc takes it times its area."""
    wavelengths_m, f_number, pitch_m = (650e-9, 450e-9), 8.0, 0.5e-6
    kernel = optics.airy_kernel(wavelengths_m, f_number, pitch_m)
    centre = kernel.shape[0] // 2
    peaks = [math.pi * (pitch_m / (2 * lam * f_number)) ** 2 for lam in wavelengths_m]
    assert kernel[centre, centre] == pytest.approx(peaks, rel=0.02)
    assert kernel.sum(axis=(0, 1)) == pytest.approx([1.0, 1.0])
    assert kernel.min() >= 0.0
