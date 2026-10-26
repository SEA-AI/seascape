import math

import numpy as np
import pytest

from docs.sea_texture import (
    MIN_CROP_PX,
    glcm_entropy,
    gradient_variance,
    histogram_entropy,
    sobel_magnitude,
    target_metrics,
)


def test_a_flat_crop_has_no_texture() -> None:
    flat = np.full((32, 32), 117, np.uint8)
    assert histogram_entropy(flat) == 0
    assert glcm_entropy(flat) == 0
    assert gradient_variance(sobel_magnitude(flat)) == 0


def test_a_one_pixel_checkerboard_has_one_bit_and_ln2_of_co_occurrence() -> None:
    # Across and down a neighbour always differs, diagonally it always matches:
    # two equally likely pairs at every offset.
    board = (np.indices((32, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    assert histogram_entropy(board) == pytest.approx(1)
    assert glcm_entropy(board) == pytest.approx(math.log(2))


def test_gradient_variance_is_over_the_magnitudes_of_its_edges() -> None:
    # Steps of 40 and 100 grey levels: a 3x3 Sobel answers 4x either side of each.
    image = np.repeat(np.array([0, 40, 140], np.uint8), 10)[None, :].repeat(20, 0)
    assert gradient_variance(sobel_magnitude(image)) == pytest.approx(
        ((4 * 100 - 4 * 40) / 2) ** 2
    )


def test_a_crop_under_the_floor_keeps_its_gradients_but_no_entropy() -> None:
    image = np.random.default_rng(0).integers(0, 256, (64, 64), dtype=np.uint8)
    side = math.isqrt(MIN_CROP_PX)
    small = target_metrics(image, (0, 0, side - 1, side))
    big = target_metrics(image, (0, 0, side, side))
    assert math.isnan(small["hist H"]) and math.isnan(small["GLCM H"])
    assert small["grad var"] > 0
    assert big["hist H"] > 0 and big["GLCM H"] > 0
