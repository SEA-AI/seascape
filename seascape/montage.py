"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote. Captions sit under each
frame, never on it: in an exr they would corrupt radiance.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from seascape import agc
from seascape.config import Scenario

TILE_H = 260
CAPTION_H = 26
GUTTER = 4

MATTE = (24, 24, 24)
INK = (232, 232, 232)


def _font() -> FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=16)


def _tile(
    path: Path, font: FreeTypeFont | ImageFont.ImageFont, caption: str
) -> Image.Image:
    t_k = agc.kelvin(path)
    frame = (
        Image.open(path) if t_k is None else Image.fromarray(agc.Agc()(t_k))
    ).convert("RGB")
    width = round(frame.width * TILE_H / frame.height)
    frame = frame.resize((width, TILE_H), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (width, TILE_H + CAPTION_H), MATTE)
    tile.paste(frame, (0, 0))
    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = draw.textbbox((0, 0), caption, font=font)
    draw.text(
        ((width - (right - left)) / 2, TILE_H + (CAPTION_H - (bottom - top)) / 2 - top),
        caption,
        font=font,
        fill=INK,
    )
    return tile


def compose(scenario: Scenario, into: Path) -> Path:
    """Write `montage.png` beside the frames in `into`, one row per band."""
    font = _font()
    rows: list[list[Image.Image]] = []
    for band in scenario.outputs.bands:
        tiles = []
        for mount in scenario.rig.mounts:
            if mount.camera.kind != band:
                continue
            # An exr is radiance; making it a picture is the render's job.
            frame = into / f"{mount.name}.{scenario.outputs.format}"
            if scenario.outputs.format == "exr" or not frame.exists():
                raise FileNotFoundError(f"{frame}: render it, as png or jpg")
            tiles.append(_tile(frame, font, mount.name))
        if tiles:
            rows.append(tiles)
    if not rows:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")

    widths = [sum(t.width for t in row) + GUTTER * (len(row) - 1) for row in rows]
    heights = [max(t.height for t in row) for row in rows]
    sheet = Image.new(
        "RGB", (max(widths), sum(heights) + GUTTER * (len(rows) - 1)), MATTE
    )
    y = 0
    for row, row_w, row_h in zip(rows, widths, heights, strict=True):
        x = (sheet.width - row_w) // 2
        for tile in row:
            sheet.paste(tile, (x, y))
            x += tile.width + GUTTER
        y += row_h + GUTTER

    out = into / "montage.png"
    sheet.save(out)
    return out
