import math

import pytest

from seascape.physics import solar


def test_the_beam_leaves_the_sky_its_diffuse_fraction_of_a_horizontal_surface() -> None:
    diffuse, k, elevation = 2.0, 0.7, math.radians(20.0)
    beam = solar.beam_normal(diffuse, k, elevation)
    assert diffuse / (diffuse + beam * math.sin(elevation)) == pytest.approx(k)
