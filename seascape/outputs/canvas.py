"""Pictures laid out for a person to look at."""

import colorsys
from collections.abc import Sequence
from functools import cache
from importlib import resources
from itertools import batched

import numpy as np
from PIL import Image, ImageFont, ImageOps
from PIL.ImageFont import FreeTypeFont

# Brand Book 2026 files: the logo PNGs and the typeface, with its licence.
BRAND = resources.files("seascape.outputs") / "brand"
# JPEG holds no alpha.
_MODE = "RGB"
MATTE = (24, 24, 24)
INK = (232, 232, 232)

type Colour = str | tuple[int, int, int]


@cache
def font(size: int = 16) -> FreeTypeFont:
    """The Brand Book's Barlow Semi Condensed Regular at `size` pixels."""
    with resources.as_file(BRAND / "BarlowSemiCondensed-Regular.ttf") as path:
        return ImageFont.truetype(path, size)


def beside(
    images: Sequence[Image.Image], *, gap_px: int = 0, matte: Colour = MATTE
) -> Image.Image:
    """`images` left to right, `gap_px` apart, tops aligned."""
    width = sum(image.width for image in images) + gap_px * max(len(images) - 1, 0)
    out = Image.new(_MODE, (width, max((i.height for i in images), default=0)), matte)
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + gap_px
    return out


def above(
    images: Sequence[Image.Image], *, gap_px: int = 0, matte: Colour = MATTE
) -> Image.Image:
    """`images` top to bottom, `gap_px` apart, left edges aligned."""
    height = sum(image.height for image in images) + gap_px * max(len(images) - 1, 0)
    out = Image.new(_MODE, (max((i.width for i in images), default=0), height), matte)
    y = 0
    for image in images:
        out.paste(image, (0, y))
        y += image.height + gap_px
    return out


def grid(
    images: Sequence[Image.Image],
    columns: int,
    *,
    gap_px: int = 0,
    matte: Colour = MATTE,
) -> Image.Image:
    """`images` row by row, `columns` to a row."""
    blanks = [Image.new(_MODE, (0, 0))] * (-len(images) % columns)
    rows = list(batched([*images, *blanks], columns, strict=True))
    return table(rows, gap_px=gap_px, matte=matte)


def table(
    rows: Sequence[Sequence[Image.Image]], *, gap_px: int = 0, matte: Colour = MATTE
) -> Image.Image:
    """`rows` of cells, each centred in its column and at the top of its row."""
    widths = [max(cell.width for cell in column) for column in zip(*rows, strict=True)]
    lines = [
        beside(
            [_centred(c, w, matte) for c, w in zip(row, widths, strict=True)],
            gap_px=gap_px,
            matte=matte,
        )
        for row in rows
    ]
    return above(lines, gap_px=gap_px, matte=matte)


def _centred(image: Image.Image, width: int, matte: Colour) -> Image.Image:
    left = (width - image.width) // 2
    border = (left, 0, width - image.width - left, 0)
    return ImageOps.expand(image.convert(_MODE), border=border, fill=matte)


def on_backdrop(image: Image.Image) -> Image.Image:
    """A dark hull shows on mid grey, not on the matte."""
    backdrop = Image.new("RGBA", image.size, (110, 110, 110, 255))
    return Image.alpha_composite(backdrop, image.convert("RGBA")).convert("RGB")


def palette(n: int) -> np.ndarray:
    """`n` colours apart from their neighbours: golden-ratio hues, three values."""
    hues = (np.arange(n) * 0.618034) % 1.0
    return np.array(
        [
            [
                round(255 * c)
                for c in colorsys.hsv_to_rgb(h, 0.75, (1.0, 0.8, 0.6)[k % 3])
            ]
            for k, h in enumerate(hues)
        ],
        dtype=np.uint8,
    ).reshape(-1, 3)
