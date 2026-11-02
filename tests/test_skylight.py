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


def test_type_7_lights_the_sky_behind_at_a_third_of_the_sky_ahead() -> None:
    """At a 20 deg sun: 0-45 deg up within 60 deg of the anti-sun bearing, over 0-5 deg
    up within 4 deg of the sun's, each weighted by solid angle."""
    sun = np.radians(20.0)
    means = []
    for centre, half, top in ((180.0, 60.0, 45.0), (0.0, 4.0, 5.0)):
        bearing = np.radians(np.arange(centre - half, centre + half, 0.5))[None, :]
        elevation = np.radians(np.arange(0.0, top, 0.5) + 0.25)[:, None]
        cos_chi = np.sin(elevation) * np.sin(sun) + np.cos(elevation) * np.cos(
            sun
        ) * np.cos(bearing)
        lum = skylight.relative_luminance(
            7, np.pi / 2 - elevation, np.arccos(cos_chi), np.pi / 2 - sun
        )
        solid = np.broadcast_to(np.cos(elevation), lum.shape)
        means.append((lum * solid).sum() / solid.sum())
    behind, ahead = means
    assert behind / ahead == pytest.approx(0.30, abs=0.005)
