"""SEA.AI's colours and typeface, for the figures in docs/."""

import functools

from PIL import ImageFont

FOCUS_RED = "#CB0D00"
NIGHT_BLUE = "#0B1731"
OCEAN_TEAL = "#06404C"
SKY_GREY = "#7B9194"
FOG_WHITE = "#DFDED9"
FONT = "Barlow Semi Condensed"
# One per target; red alone hides on a white hull.
MASK_COLOURS = [FOCUS_RED, "#FFB000", "#00C2D1"]


@functools.cache
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(f"BarlowSemiCondensed-{weight}.ttf", size)
    except OSError as e:
        raise OSError(f"{FONT} is not installed; it is on Google Fonts") from e
