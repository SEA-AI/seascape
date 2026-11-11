"""Colour temperature, white balance and tone curves. No Blender."""

import numpy as np
import pytest

from seascape.config import Processing
from seascape.physics import colour


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


def test_every_allowed_balance_has_finite_positive_gains() -> None:
    bounds = {
        type(m).__name__: m for m in Processing.model_fields["white_balance_k"].metadata
    }
    for error_k in np.linspace(bounds["Ge"].ge, bounds["Le"].le, 221):
        gains = colour.white_balance_gains(float(error_k))
        assert all(0.0 < g < 10.0 for g in gains), error_k


def test_bt709_crushes_the_shadows_that_srgb_lifts() -> None:
    """BT.709's toe is 4.5 to sRGB's 12.92 (IEC 61966-2-1)."""
    curve = colour.tone_curve("bt709")
    assert (curve[0], curve[-1]) == pytest.approx((0.0, 1.0), abs=1e-3)
    assert (curve[1:-1] < colour.TONE_IN[1:-1]).all()


def test_srgb_encode_inverts_srgb_decode() -> None:
    linear = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(colour.srgb_decode(colour.srgb_encode(linear)), linear)


@pytest.mark.parametrize(
    ("srgb_hex", "linear"),
    # IEC 61966-2-1: 128 of 255 decodes to ((128/255 + 0.055) / 1.055) ** 2.4.
    [("#000000", 0.0), ("#FFFFFF", 1.0), ("#808080", 0.21586)],
)
def test_an_srgb_hex_decodes_to_linear(srgb_hex: str, linear: float) -> None:
    assert colour.srgb_hex_decode(srgb_hex) == pytest.approx((linear,) * 3, abs=1e-5)
