"""The thermal camera's two outputs: 16-bit counts, and 8-bit grey after AGC."""

import math

import cv2
import numpy as np
import pytest

from seascape import agc


def column(*t_k: float) -> np.ndarray:
    return np.array(t_k)[:, None]


def test_two_temperatures_separate_to_black_and_white() -> None:
    assert agc.Agc()(column(272.0, 295.0)).ravel().tolist() == pytest.approx(
        [0, 255], abs=1
    )


def test_the_middle_temperature_is_mid_grey() -> None:
    """Catches a gamma encode, which puts 0.5 at 0.74."""
    assert agc.Agc()(column(270.0, 285.0, 300.0))[1, 0] == 128


def test_a_target_is_not_flattened_to_white() -> None:
    """A percentile stretch would trim the few rows a distant hull occupies."""
    grey = agc.Agc()(column(*[285.0] * 40, 300.0)).ravel().astype(int)
    assert (grey[0], grey[-1]) == pytest.approx((0, 255), abs=1)


def test_a_frame_of_one_temperature_is_mid_grey() -> None:
    assert agc.Agc()(column(290.0, 290.0)).ravel().tolist() == [128, 128]


def test_noise_on_a_uniform_frame_gains_at_most_the_cap() -> None:
    """A min-max stretch would spread 50 mK of noise across all 256 greys."""
    t_k = np.random.default_rng(0).normal(290.0, 0.05, (64, 64))
    grey = agc.Agc()(t_k)
    assert abs(grey.mean() - 128) < 5
    assert grey.std() < 1.05 * agc.MAX_GAIN * t_k.std() * agc.CENTIKELVIN


def test_the_slope_never_exceeds_the_cap() -> None:
    """A 1 K ramp is 101 counts: the cap leaves it short of the full range."""
    t_k = np.linspace(290.0, 291.0, 101)[:, None]
    assert np.diff(agc.Agc()(t_k).astype(int), axis=0).max() <= 2
    assert np.ptp(agc.Agc()(t_k)) <= agc.MAX_GAIN * 101


def test_without_a_step_each_frame_takes_its_own_curve() -> None:
    tone = agc.Agc()
    tone(column(280.0, 290.0))
    assert tone(column(290.0, 300.0)).ravel().tolist() == pytest.approx([0, 255], abs=1)


def test_the_curve_follows_a_step_change_over_the_time_constant() -> None:
    """A hot target entering must not flash the background: after one time constant
    the curve has moved 1 - 1/e of the way, and after many it has arrived."""
    before, after = column(280.0, 290.0), column(280.0, 287.0, 300.0)
    step_s = agc.TAU_S / 10
    tone = agc.Agc(step_s)
    tone(before)
    for _ in range(9):
        tone(after)
    moved = 1 - math.exp(-1)
    curve = (1 - moved) * agc.transfer(before) + moved * agc.transfer(after)
    assert tone(after).tolist() == agc.grey(after, curve).tolist()
    for _ in range(200):
        tone(after)
    assert tone(after).tolist() == agc.Agc()(after).tolist()


def test_counts_survive_a_png(tmp_path) -> None:
    t_k = column(271.234, 300.987)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), agc.counts(t_k))
    assert agc.kelvin(path) == pytest.approx(t_k, abs=0.005)


def test_an_8_bit_image_has_no_temperatures(tmp_path) -> None:
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.zeros((2, 2), np.uint8))
    assert agc.kelvin(path) is None


def test_a_temperature_past_16_bits_raises() -> None:
    with pytest.raises(ValueError, match="overflows"):
        agc.counts(column(700.0))
