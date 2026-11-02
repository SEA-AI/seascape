"""The CIE standard general sky against the clear sky it generalises. No Blender."""

import numpy as np
import pytest

from seascape.physics import skylight


def test_type_12_is_the_cie_clear_sky() -> None:
    """CIE 22-1973 after Kittler (1967): f(chi) = 0.91 + 10 exp(-3 chi) + 0.45 cos^2
    chi and phi(Z) = 1 - exp(-0.32 / cos Z)."""
    sun = np.radians(40.0)
    chi = np.radians(np.arange(0.0, 180.0, 15.0))[:, None]
    z = np.radians(np.arange(0.0, 90.0, 10.0))[None, :]
    f = 0.91 + 10.0 * np.exp(-3.0 * chi) + 0.45 * np.cos(chi) ** 2
    f_sun = 0.91 + 10.0 * np.exp(-3.0 * sun) + 0.45 * np.cos(sun) ** 2
    clear = f * (1.0 - np.exp(-0.32 / np.cos(z))) / (f_sun * (1.0 - np.exp(-0.32)))
    # exp(-3 pi / 2) leaves 0.91 to two figures.
    assert skylight.relative_luminance(12, z, chi, sun) == pytest.approx(
        clear, rel=2e-3
    )
