"""SEA.AI's Brand Book 2026: colours, typeface and logos."""

from functools import cache
from importlib import resources
from io import BytesIO
from typing import Literal

from PIL import Image, ImageFont
from PIL.ImageFont import FreeTypeFont

# The logo PNGs, typeface and its licence, beside this file.
_ASSETS = resources.files(__package__)
FOCUS_RED = "#CB0D00"
NIGHT_BLUE = "#0B1731"
OCEAN_TEAL = "#06404C"
SKY_GREY = "#7B9194"
FOG_WHITE = "#DFDED9"
# The family's name, for a plotting library that finds it among installed fonts.
FAMILY = "Barlow Semi Condensed"

type Weight = Literal["Regular", "Medium"]


@cache
def font(size: int = 16, weight: Weight = "Regular") -> FreeTypeFont:
    """The typeface at `size` pixels: Regular for text, Medium for labels and
    headings.
    """
    data = (_ASSETS / f"BarlowSemiCondensed-{weight}.ttf").read_bytes()
    return ImageFont.truetype(BytesIO(data), size)


@cache
def logo(colour: str) -> Image.Image:
    """The SEA.AI mark in `colour` (black or white) as RGBA."""
    with (_ASSETS / f"logo_{colour}.png").open("rb") as file:
        return Image.open(file).convert("RGBA")
