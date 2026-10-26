"""The thermal camera's two outputs: 16-bit counts, and 8-bit grey after AGC."""

import math

import cv2
import numpy as np
import pytest

from seascape import agc


def column(*t_k: float) -> np.ndarray:
    return np.array(t_k)[:, None]


def test_a_frame_is_stretched_to_the_full_range() -> None:
    assert agc.Agc()(column(272.0, 295.0)).ravel().tolist() == [0, 255]


def test_the_middle_temperature_is_mid_grey() -> None:
    """Catches a gamma encode, which puts 0.5 at 0.74."""
    assert agc.Agc()(column(270.0, 285.0, 300.0))[1, 0] == 128


def test_a_target_is_not_flattened_to_white() -> None:
    """A percentile stretch would trim the few rows a distant hull occupies."""
    grey = agc.Agc()(column(*[285.0] * 40, 300.0)).ravel()
    assert (grey[0], grey[-1]) == (0, 255)


def test_a_frame_of_one_temperature_is_mid_grey() -> None:
    assert agc.Agc()(column(290.0, 290.0)).ravel().tolist() == [128, 128]


def test_without_a_step_each_frame_takes_its_own_span() -> None:
    tone = agc.Agc()
    tone(column(280.0, 290.0))
    assert tone(column(290.0, 300.0)).ravel().tolist() == [0, 255]


def test_the_span_follows_a_step_change_over_the_time_constant() -> None:
    """A hot target entering must not flash the background: after one time constant
    the span has moved 1 - 1/e of the way, and after many it has arrived."""
    step_s = 0.1
    tone = agc.Agc(step_s)
    tone(column(280.0, 290.0))
    frames = round(agc.TAU_S / step_s)
    for _ in range(frames - 1):
        tone(column(280.0, 300.0))
    # The top of the span sits at 290 + 10 (1 - 1/e) K, so 290 K is 10 / 16.3 grey.
    assert tone(column(280.0, 290.0, 300.0))[1, 0] == 156
    for _ in range(20 * frames):
        tone(column(280.0, 300.0))
    assert tone(column(280.0, 290.0, 300.0)).ravel().tolist() == [0, 128, 255]


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


def test_an_infinite_plateau_equalises_the_histogram() -> None:
    """Cumulative counts 1, 3, 4 stretched from the first to the last: 0, 2/3, 1."""
    tone = agc.PlateauAgc(math.inf)
    assert tone(column(280.0, 290.0, 290.0, 300.0)).ravel().tolist() == [
        0,
        170,
        170,
        255,
    ]


def test_a_plateau_of_one_projects_the_histogram() -> None:
    """Every occupied bin one step, however many pixels it holds."""
    tone = agc.PlateauAgc(1)
    grey = tone(column(280.0, *[290.0] * 50, 300.0)).ravel()
    assert (grey[0], grey[1], grey[-1]) == (0, 128, 255)


def test_a_plateau_of_one_ignores_the_spacing_of_temperatures() -> None:
    """Unlike a linear stretch, which would put 281 K near black."""
    assert agc.PlateauAgc(1)(column(280.0, 281.0, 300.0))[1, 0] == 128


def test_a_plateau_frame_of_one_temperature_is_mid_grey() -> None:
    assert agc.PlateauAgc(10)(column(290.0, 290.0)).ravel().tolist() == [128, 128]


def test_a_plateau_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        agc.PlateauAgc(0)


def test_the_plateau_transfer_converges_after_a_step_change() -> None:
    step_s = 0.1
    tone = agc.PlateauAgc(1, step_s)
    tone(column(280.0, 300.0))
    hot = column(280.0, 285.0, 290.0, 300.0)
    # 285 K was black in the first frame, so it starts below its new grey.
    assert tone(hot)[1, 0] < 85
    for _ in range(20 * round(agc.TAU_S / step_s)):
        tone(hot)
    assert tone(hot).ravel().tolist() == [0, 85, 170, 255]
