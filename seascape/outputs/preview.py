"""A render's frames as pictures to share, each with the SEA.AI mark on it."""

from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np
from PIL import Image

from seascape.config import Agc, Mount, Scenario
from seascape.outputs import agc, canvas

# Official SEA.AI Brand Book 2026 PNGs; never redraw the mark.
BRAND = resources.files("seascape.outputs") / "brand"
MIN_MARK_PX = 15  # Brand Book 2026 minimum height
# Judgement: the mark's height over the image's short edge.
MARK_SHARE = 0.03
# Judgement: readable over a photo without hiding what is under it.
MARK_OPACITY = 0.7
# Brand Book clear space: the cursor icon's share of the mark's width.
CURSOR_SHARE = 0.179
DARK = 128  # luma below this takes the white mark


@cache
def _logo(colour: str) -> Image.Image:
    with (BRAND / f"logo_{colour}.png").open("rb") as file:
        return Image.open(file).convert("RGBA")


def picture(path: Path, setting: Agc) -> Image.Image:
    """The frame at `path` as an RGB picture: a thermal one through `setting`."""
    t_k = agc.kelvin(path)
    frame = Image.open(path) if t_k is None else Image.fromarray(agc.Agc(setting)(t_k))
    return frame.convert("RGB")


def _mark_size(image: Image.Image) -> tuple[int, int, int]:
    """The mark's width and height on `image`, and the clear space around it."""
    height = max(MIN_MARK_PX, round(MARK_SHARE * min(image.size)))
    white = _logo("white")
    width = round(height * white.width / white.height)
    return width, height, round(width * CURSOR_SHARE)


def watermark(image: Image.Image, *, below: bool = False) -> Image.Image:
    """`image` with the mark at its bottom right, white or black by what is behind.

    `below` adds a strip under `image` for the mark, so it covers none of the picture.
    """
    width, height, clear = _mark_size(image)
    if below:
        strip = Image.new("RGB", (image.width, image.height + height + 2 * clear))
        strip.paste(canvas.MATTE, (0, 0, *strip.size))
        strip.paste(image, (0, 0))
        image = strip
    marked = image.convert("RGBA")
    # A frame too small for the clear space still takes the mark, flush to its corner.
    left = max(0, marked.width - clear - width)
    top = max(0, marked.height - clear - height)
    behind = marked.crop((left, top, left + width, top + height)).convert("L")
    logo = _logo("white" if np.asarray(behind).mean() < DARK else "black")
    logo = logo.resize((width, height), Image.Resampling.LANCZOS)
    logo.putalpha(logo.getchannel("A").point(lambda a: round(a * MARK_OPACITY)))
    marked.alpha_composite(logo, (left, top))
    return marked.convert("RGB")


def shown(scenario: Scenario, folder: Path) -> list[tuple[Mount, Path]]:
    """The frame to show of each camera in `folder`: the still, or a clip's middle."""
    ext = scenario.outputs.format
    found: list[tuple[Mount, Path]] = []
    for band in scenario.outputs.bands:
        for mount in scenario.band_mounts(band):
            if scenario.outputs.frames == 1:
                found.append((mount, folder / f"{mount.name}.{ext}"))
            else:
                clip = sorted((folder / mount.name).glob(f"*.{ext}"))
                if not clip:
                    raise FileNotFoundError(f"{folder / mount.name}: no frames")
                found.append((mount, clip[len(clip) // 2]))
    return found
