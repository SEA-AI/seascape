import math

import cv2
import numpy as np
import pytest

from docs.sea_texture import (
    BOX_PX,
    MIN_CROP_PX,
    glcm_entropy,
    gradient_variance,
    histogram_entropy,
    sobel_magnitude,
    structure,
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


def test_the_sky_s_noise_leaves_the_sea_s_structure() -> None:
    rng = np.random.default_rng(0)
    # Streaks four times as long across as down, under as much noise.
    streaks = cv2.GaussianBlur(rng.standard_normal((256, 256)), (0, 0), 2.0, sigmaY=0.5)
    sea = streaks / streaks.std()
    sky, noise = rng.standard_normal((2, 256, 256))

    def texture(frame: np.ndarray) -> np.ndarray:
        return frame - cv2.blur(frame, (BOX_PX, BOX_PX))

    clean = structure(texture(sea), np.zeros_like(sky))
    seen = structure(texture(sea + noise), texture(sky))
    bare = structure(texture(noise), texture(sky))["std"]

    for metric in ("std", "lx", "gx/gy"):
        assert seen[metric] == pytest.approx(clean[metric], rel=0.1), metric
    assert math.isnan(bare) or bare < 0.1
