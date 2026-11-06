import math

import numpy as np
import pytest

from seascape.physics import earth, solar


def test_the_beam_leaves_the_sky_its_diffuse_fraction_of_a_horizontal_surface() -> None:
    diffuse, k, elevation = 2.0, 0.7, math.radians(20.0)
    beam = solar.beam_normal(diffuse, k, elevation)
    assert diffuse / (diffuse + beam * math.sin(elevation)) == pytest.approx(k)


def test_a_direction_points_along_its_compass_bearing() -> None:
    np.testing.assert_allclose(earth.direction(0.0, 0.0), (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(earth.direction(90.0, 0.0), (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(earth.direction(33.0, 90.0), (0.0, 0.0, 1.0), atol=1e-12)
