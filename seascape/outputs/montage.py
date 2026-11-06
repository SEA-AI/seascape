"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote. Captions sit under each
frame, never on it: in an exr they would corrupt radiance.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from seascape.config import Agc, Scenario
from seascape.outputs import agc

TILE_H = 260
CAPTION_H = 26
GUTTER = 4

MATTE = (24, 24, 24)
INK = (232, 232, 232)


def font(size: int = 16) -> FreeTypeFont | ImageFont.ImageFont:
    """Pillow's default font at `size` pixels."""
    return ImageFont.load_default(size=size)


def _tile(
    path: Path,
    font: FreeTypeFont | ImageFont.ImageFont,
    caption: str,
    setting: Agc,
    width: int,
) -> Image.Image:
    t_k = agc.kelvin(path)
    frame = (
        Image.open(path) if t_k is None else Image.fromarray(agc.Agc(setting)(t_k))
    ).convert("RGB")
    height = round(frame.height * width / frame.width)
    frame = frame.resize((width, height), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (width, height + CAPTION_H), MATTE)
    tile.paste(frame, (0, 0))
    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
    draw.text(
        ((width - (right - left)) / 2, height + (CAPTION_H - (bottom - top)) / 2 - top),
        caption,
        font=font,
        fill=INK,
    )
    return tile


def _place(
    row: list[tuple[float, Image.Image]], px_per_deg: float
) -> list[tuple[int, Image.Image]]:
    """Each tile's x under its bearing, in pasting order; a later tile covers a
    neighbour's overlap, and one more than half covered goes beside it.
    """
    placed: list[tuple[int, Image.Image]] = []
    right = -math.inf
    for bearing_deg, tile in sorted(row, key=lambda pair: pair[0]):
        x = round(bearing_deg * px_per_deg - tile.width / 2)
        if right - x > tile.width / 2:
            x = round(right) + GUTTER
        placed.append((x, tile))
        right = max(right, x + tile.width)
    return placed


def compose(scenario: Scenario, into: Path) -> Path:
    """Write `montage.png` beside the frames in `into`, one row per band, each frame
    under its bearing and as wide as the angle it sees.
    """
    typeface = font()
    mounts = [m for band in scenario.outputs.bands for m in scenario.band_mounts(band)]
    if not mounts:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")
    px_per_deg = TILE_H / max(
        m.camera.hfov_deg * m.camera.height_px / m.camera.width_px for m in mounts
    )
    # The circle is cut in its widest gap, so no two neighbours land at opposite ends.
    around = sorted(m.nominal_bearing_deg % 360.0 for m in mounts)
    after = [*around[1:], around[0] + 360.0]
    gaps = [b - a for a, b in zip(around, after, strict=True)]
    widest = gaps.index(max(gaps))
    cut_deg = around[widest] + gaps[widest] / 2
    rows: list[list[tuple[float, Image.Image]]] = []
    for band in scenario.outputs.bands:
        tiles = []
        for mount in scenario.band_mounts(band):
            # An exr is radiance; making it a picture is the render's job.
            frame = into / f"{mount.name}.{scenario.outputs.format}"
            if scenario.outputs.format == "exr" or not frame.exists():
                raise FileNotFoundError(f"{frame}: render it, as png or jpg")
            width = round(mount.camera.hfov_deg * px_per_deg)
            tile = _tile(frame, typeface, mount.name, mount.camera.agc, width)
            # Nominal is close enough to lay a sheet out by.
            tiles.append(((mount.nominal_bearing_deg - cut_deg) % 360.0, tile))
        if tiles:
            rows.append(tiles)

    placed = [_place(row, px_per_deg) for row in rows]
    left = min(x for row in placed for x, _ in row)
    sheet_w = max(x + tile.width for row in placed for x, tile in row) - left
    heights = [max(tile.height for _, tile in row) for row in placed]
    sheet = Image.new("RGB", (sheet_w, sum(heights) + GUTTER * (len(rows) - 1)), MATTE)
    y = 0
    for row, row_h in zip(placed, heights, strict=True):
        for x, tile in row:
            sheet.paste(tile, (x - left, y))
        y += row_h + GUTTER

    out = into / "montage.png"
    sheet.save(out)
    return out
