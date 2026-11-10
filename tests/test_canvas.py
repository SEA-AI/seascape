"""Laying pictures out for a person to look at."""

import numpy as np
import pytest
from PIL import Image

from seascape.outputs import canvas

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def test_a_grid_fills_rows_first_and_leaves_the_rest_matte() -> None:
    tiles = [Image.new("RGB", (4, 2), colour) for colour in (RED, GREEN, BLUE)]

    sheet = np.asarray(canvas.grid(tiles, 2, gap_px=1))

    assert sheet.shape == (5, 9, 3)
    assert (sheet[0, 0] == RED).all()
    assert (sheet[0, 5] == GREEN).all()
    assert (sheet[3, 0] == BLUE).all()
    assert (sheet[3, 5] == canvas.MATTE).all()


def test_beside_and_above_keep_every_picture_whole_and_gap_apart() -> None:
    wide, tall = Image.new("RGB", (3, 2), RED), Image.new("RGB", (2, 4), BLUE)

    row = np.asarray(canvas.beside([wide, tall], gap_px=1))
    column = np.asarray(canvas.above([wide, tall], gap_px=1))

    assert row.shape == (4, 6, 3)
    assert (row[:2, :3] == RED).all() and (row[:, 4:] == BLUE).all()
    assert column.shape == (7, 3, 3)
    assert (column[:2] == RED).all() and (column[3:, :2] == BLUE).all()


def test_a_table_centres_each_cell_in_its_column() -> None:
    """An empty cell keeps its place and a short one sits at the top of its row, so
    captions line up across columns.
    """
    wide, narrow, empty = (
        Image.new("RGB", size, RED) for size in ((6, 2), (2, 3), (0, 0))
    )

    sheet = np.asarray(canvas.table([[wide, empty], [narrow, wide]], gap_px=1))

    assert sheet.shape == (6, 13, 3)
    assert (sheet[3:, 2:4] == RED).all()
    assert (sheet[3:, :2] == canvas.MATTE).all()
    assert (sheet[:2, 7:] == canvas.MATTE).all()
    assert (sheet[3:5, 7:] == RED).all()
    assert (sheet[5, 7:] == canvas.MATTE).all()


def test_nothing_lays_out_as_an_empty_picture() -> None:
    """A montage column with no camera in a band is an empty cell."""
    assert canvas.beside([]).size == (0, 0)
    assert canvas.above([]).size == (0, 0)
    assert canvas.table([]).size == (0, 0)
    assert canvas.grid([], 3).size == (0, 0)


@pytest.mark.parametrize("weight", ["Regular", "Medium"])
def test_the_typeface_is_the_brand_books(weight: canvas.Weight) -> None:
    assert canvas.font(20, weight).getname() == ("Barlow Semi Condensed", weight)


def test_the_typeface_has_the_glyphs_a_caption_uses() -> None:
    def drawn(char: str) -> tuple[tuple[int, int], bytes]:
        mask = canvas.font(20).getmask(char)
        return mask.size, np.array(mask).tobytes()

    # A glyph the font lacks draws its notdef box, as U+FFFF does.
    assert all(drawn(char) != drawn("\uffff") for char in "°+-0w")
