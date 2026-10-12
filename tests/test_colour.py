"""Colour temperature and white balance. No Blender."""

import pytest

from seascape import colour


def test_the_locus_passes_through_illuminant_a() -> None:
    """CIE 15:2004: illuminant A, a Planckian radiator at 2856 K."""
    assert colour.planck_xy(2856.0) == pytest.approx((0.4476, 0.4074), abs=1e-3)


def test_a_guess_of_d65_balances_nothing() -> None:
    assert colour.white_balance_gains(0.0) == pytest.approx((1.0, 1.0, 1.0))


def test_a_warmer_guess_boosts_blue_and_a_bluer_one_red() -> None:
    warm_r, _, warm_b = colour.white_balance_gains(-1500.0)
    cool_r, _, cool_b = colour.white_balance_gains(1500.0)
    assert warm_b > 1.0 > warm_r
    assert cool_r > 1.0 > cool_b


def test_the_locus_refuses_what_its_cubic_does_not_cover() -> None:
    with pytest.raises(ValueError, match="1667-25000"):
        colour.planck_xy(1000.0)
