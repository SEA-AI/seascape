"""The radiating skin against Fairall et al. 1996's own numbers. No Blender."""

import math

import pytest
from pydantic import ValidationError

from seascape.config import Sea, WarmLayer
from seascape.physics import skin
from tests.scenarios import variant

# Fairall et al. 1996, Table 3: wind, net cooling, u*^2, depth, f_w, peak warming.
TABLE_3 = [
    (1.0, 100.0, 0.0015, 0.7, 0.44, 3.8),
    (2.0, 110.0, 0.0052, 2.1, 0.54, 1.6),
    (3.0, 125.0, 0.011, 4.2, 0.59, 0.9),
    (4.0, 165.0, 0.016, 6.0, 0.63, 0.6),
    (5.0, 190.0, 0.025, 10.0, 0.68, 0.4),
    (6.0, 230.0, 0.034, 14.0, 0.72, 0.25),
    (7.0, 260.0, 0.049, 19.0, 0.76, 0.15),
]
# The paper gives no water temperature. Judgement: 29 C, inside the western Pacific warm
# pool, water above 28 C (Webster & Lukas, BAMS 73(9) 1377, 1992).
COARE_K = 302.15


@pytest.mark.parametrize(
    ("wind", "cooling", "stress", "depth", "trapped", "rise"), TABLE_3
)
def test_the_warm_layer_reproduces_fairall_s_clear_day(
    wind: float,
    cooling: float,
    stress: float,
    depth: float,
    trapped: float,
    rise: float,
) -> None:
    """The paper prints no seawater or air constants; these are COARE 3.6's."""
    got_rise, got_depth = skin.warm_layer(
        skin.CLEAR_DAY_SOLAR_W_M2, cooling, stress, skin.CLEAR_DAY_HOURS, COARE_K
    )
    assert got_rise == pytest.approx(rise, rel=0.15)
    assert got_depth == pytest.approx(depth, rel=0.2)
    assert skin.solar_trapped(got_depth) == pytest.approx(trapped, abs=0.03)


def test_the_wind_alone_spans_table_3() -> None:
    """Through the paper's drag coefficient, not the table's u*^2."""
    warm = WarmLayer()
    assert warm.rise_k(1.0, COARE_K) == pytest.approx(3.8, rel=0.15)
    assert warm.rise_k(7.0, COARE_K) == pytest.approx(0.15, rel=0.2)


def test_no_sun_no_warm_layer() -> None:
    assert skin.warm_layer(0.0, 50.0, 0.01, 7.0, 290.0) == (0.0, math.inf)


def test_a_calm_warm_layer_is_finite() -> None:
    assert 0.0 < WarmLayer().rise_k(0.0, 290.0) < 10.0


def test_the_cool_skin_is_fairall_s_noon_and_night_means() -> None:
    sea = Sea(t_sea_k=290.0)
    assert sea.t_skin_k(sun_up=True) == pytest.approx(290.0 - 0.18)
    assert sea.t_skin_k(sun_up=False) == pytest.approx(290.0 - 0.30)


def test_a_warm_layer_warms_the_skin_more_in_light_wind() -> None:
    light, strong = (
        Sea(t_sea_k=290.0, wind_speed_mps=v, warm_layer=WarmLayer()).t_skin_k(True)
        for v in (1.0, 7.0)
    )
    assert light > strong > 290.0 - skin.COOL_SKIN_DAY_K


def test_the_skin_follows_a_copied_bulk() -> None:
    """Derived where it is read, so `model_copy` cannot leave it behind."""
    sea = Sea(t_sea_k=290.0).model_copy(update={"t_sea_k": 300.0})
    assert sea.t_skin_k(True) == pytest.approx(300.0 - skin.COOL_SKIN_DAY_K)


def test_a_warm_layer_needs_the_sun_up() -> None:
    with pytest.raises(ValidationError, match="sun above the horizon"):
        variant(sea={"warm_layer": {}}, sky={"sun_elevation_deg": -5.0})
