import numpy as np
import pytest

from seascape import water
from seascape.config import Sea
from seascape.sea import SEAWATER_IOR


def test_the_default_water_is_the_colour_the_sea_had_before_chlorophyll() -> None:
    chl_mg_m3 = Sea().chlorophyll_mg_m3

    color = water.body_color(chl_mg_m3, SEAWATER_IOR)

    # The constant the shader carried, from the same model at 0.2 mg/m^3.
    assert color == pytest.approx((0.0, 0.0065, 0.018), abs=1e-4)


def test_chlorophyll_turns_the_water_from_blue_to_green() -> None:
    chl_mg_m3 = np.geomspace(0.03, 10.0, 8)

    green_over_blue = np.array(
        [np.divide(*water.body_color(c, SEAWATER_IOR)[1:]) for c in chl_mg_m3]
    )

    assert np.all(np.diff(green_over_blue) > 0)
    assert green_over_blue[0] < 1 < green_over_blue[-1]
