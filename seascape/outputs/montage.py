"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote. Captions sit under each
frame, never on it: in an exr they would corrupt radiance.
"""

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from seascape.config import Mount, Scenario
from seascape.outputs import agc

TILE_H = 260
CAPTION_H = 26
GUTTER = 4
FAN_R = 10

MATTE = (24, 24, 24)
INK = (232, 232, 232)
SHADE = (88, 88, 88)
LIT = (255, 200, 60)


def font(size: int = 16) -> FreeTypeFont | ImageFont.ImageFont:
    """Pillow's default font at `size` pixels."""
    return ImageFont.load_default(size=size)


def _wrapped(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _apart_deg(a: Mount, b: Mount) -> float:
    return abs(_wrapped(a.nominal_bearing_deg - b.nominal_bearing_deg))


def _fan(
    draw: ImageDraw.ImageDraw, x: int, y: int, mounts: Sequence[Mount], lit: Mount
) -> None:
    """Every camera's wedge seen from above, bow up, with `lit`'s on top."""
    box = (x - FAN_R, y - FAN_R, x + FAN_R, y + FAN_R)
    draw.ellipse(box, outline=SHADE)
    others = [m for m in mounts if m.name != lit.name]
    for mount, fill in [*((m, SHADE) for m in others), (lit, LIT)]:
        half = mount.camera.hfov_deg / 2
        # Pillow turns clockwise from 3 o'clock, so the bow sits at -90.
        start = mount.nominal_bearing_deg - half - 90.0
        draw.pieslice(box, start, start + 2 * half, fill=fill)


def _tile(
    path: Path,
    typeface: FreeTypeFont | ImageFont.ImageFont,
    mount: Mount,
    mounts: Sequence[Mount],
) -> Image.Image:
    t_k = agc.kelvin(path)
    frame = (
        Image.open(path)
        if t_k is None
        else Image.fromarray(agc.Agc(mount.camera.agc)(t_k))
    ).convert("RGB")
    width = round(frame.width * TILE_H / frame.height)
    frame = frame.resize((width, TILE_H), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (width, TILE_H + CAPTION_H), MATTE)
    tile.paste(frame, (0, 0))
    draw = ImageDraw.Draw(tile)
    middle = TILE_H + CAPTION_H // 2
    _fan(draw, GUTTER + FAN_R, middle, mounts, mount)
    bearing = _wrapped(mount.nominal_bearing_deg)
    caption = f"{mount.name}  {bearing:+.0f}°  {mount.camera.hfov_deg:.0f}° wide"
    left, top, right, bottom = draw.textbbox((0, 0), caption, font=typeface)
    x = max((width - (right - left)) / 2, 2 * (GUTTER + FAN_R))
    draw.text((x, middle - (bottom - top) / 2 - top), caption, font=typeface, fill=INK)
    return tile


def _columns(mounts: Sequence[Mount]) -> list[list[Mount]]:
    """`mounts` in columns, left to right by bearing, each column's in bearing order.

    The cameras listed first set the columns.
    """
    columns: list[list[Mount]] = []
    for mount in mounts:
        seen_by = [
            column
            for column in columns
            if _apart_deg(mount, column[0]) <= column[0].camera.hfov_deg / 2
        ]
        if seen_by:
            min(seen_by, key=lambda column: _apart_deg(mount, column[0])).append(mount)
        else:
            columns.append([mount])

    # Cut the circle in its widest gap, so no two neighbours land at opposite ends.
    around = sorted(column[0].nominal_bearing_deg % 360.0 for column in columns)
    after = [*around[1:], around[0] + 360.0]
    gaps = [b - a for a, b in zip(around, after, strict=True)]
    widest = gaps.index(max(gaps))
    cut_deg = around[widest] + gaps[widest] / 2

    def rightward(mount: Mount) -> float:
        return (mount.nominal_bearing_deg - cut_deg) % 360.0

    columns.sort(key=lambda column: rightward(column[0]))
    return [sorted(column, key=rightward) for column in columns]


def compose(scenario: Scenario, into: Path) -> Path:
    """Write `montage.png` in `into`: a row per band, a column per bearing."""
    bands = [band for band in scenario.outputs.bands if scenario.band_mounts(band)]
    if not bands:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")
    # The widest cameras set the columns.
    mounts = sorted(
        (m for band in bands for m in scenario.band_mounts(band)),
        key=lambda m: m.camera.hfov_deg,
        reverse=True,
    )
    typeface = font()
    tiles: dict[str, Image.Image] = {}
    for mount in mounts:
        # An exr is radiance; making it a picture is the render's job.
        frame = into / f"{mount.name}.{scenario.outputs.format}"
        if scenario.outputs.format == "exr" or not frame.exists():
            raise FileNotFoundError(f"{frame}: render it, as png or jpg")
        tiles[mount.name] = _tile(frame, typeface, mount, mounts)

    columns = _columns(mounts)
    rows = [
        [[tiles[m.name] for m in column if m.camera.band == band] for column in columns]
        for band in bands
    ]

    def width(cell: list[Image.Image]) -> int:
        return sum(tile.width for tile in cell) + GUTTER * max(len(cell) - 1, 0)

    column_w = [max(width(row[c]) for row in rows) for c in range(len(columns))]
    row_h = TILE_H + CAPTION_H + GUTTER
    sheet_w = sum(column_w) + GUTTER * (len(columns) - 1)
    sheet = Image.new("RGB", (sheet_w, row_h * len(rows) - GUTTER), MATTE)
    for r, row in enumerate(rows):
        left = 0
        for cell, cell_w in zip(row, column_w, strict=True):
            x = left + (cell_w - width(cell)) // 2
            for tile in cell:
                sheet.paste(tile, (x, r * row_h))
                x += tile.width + GUTTER
            left += cell_w + GUTTER

    out = into / "montage.png"
    sheet.save(out)
    return out
