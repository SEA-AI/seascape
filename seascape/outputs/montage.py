"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote. Captions sit under each
frame, never on it: in an exr they would corrupt radiance.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont

from seascape.config import Band, Mount, Scenario
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


def _cut_deg(bearings_deg: Sequence[float]) -> float:
    """The middle of the widest gap between `bearings_deg`: cut there, the circle
    leaves no two neighbours at opposite ends.
    """
    around = sorted(bearing % 360.0 for bearing in bearings_deg)
    after = [*around[1:], around[0] + 360.0]
    gap, start = max((b - a, a) for a, b in zip(around, after, strict=True))
    return start + gap / 2


def columns(mounts: Iterable[Mount]) -> list[list[Mount]]:
    """`mounts` in columns, left to right by bearing, each column's in bearing order.

    The widest cameras set the columns; each other joins the one whose first camera
    sees its bearing, or starts its own.
    """
    laid: list[list[Mount]] = []
    for mount in sorted(mounts, key=lambda m: m.camera.hfov_deg, reverse=True):
        seen_by = [
            column
            for column in laid
            if _apart_deg(mount, column[0]) <= column[0].camera.hfov_deg / 2
        ]
        if seen_by:
            min(seen_by, key=lambda column: _apart_deg(mount, column[0])).append(mount)
        else:
            laid.append([mount])

    cut_deg = _cut_deg([column[0].nominal_bearing_deg for column in laid])

    def rightward(mount: Mount) -> float:
        return (mount.nominal_bearing_deg - cut_deg) % 360.0

    laid.sort(key=lambda column: rightward(column[0]))
    return [sorted(column, key=rightward) for column in laid]


def _beside(images: Sequence[Image.Image]) -> Image.Image:
    """`images` left to right, GUTTER apart, tops aligned."""
    width = sum(image.width for image in images) + GUTTER * max(len(images) - 1, 0)
    height = max((image.height for image in images), default=0)
    out = Image.new("RGB", (width, height), MATTE)
    x = 0
    for image in images:
        out.paste(image, (x, 0))
        x += image.width + GUTTER
    return out


def _stacked(
    column: Sequence[Mount], bands: Sequence[Band], tiles: Mapping[str, Image.Image]
) -> Image.Image:
    """A row per band of `column`'s tiles, side by side and centred."""
    cells = [
        _beside([tiles[m.name] for m in column if m.camera.band == band])
        for band in bands
    ]
    row_h = TILE_H + CAPTION_H + GUTTER
    out = Image.new(
        "RGB", (max(c.width for c in cells), row_h * len(cells) - GUTTER), MATTE
    )
    for r, cell in enumerate(cells):
        out.paste(cell, ((out.width - cell.width) // 2, r * row_h))
    return out


def compose(scenario: Scenario, into: Path) -> Path:
    """Write `montage.png` in `into`: a row per band, a column per bearing."""
    bands = [band for band in scenario.outputs.bands if scenario.band_mounts(band)]
    if not bands:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")
    mounts = [m for band in bands for m in scenario.band_mounts(band)]
    typeface = font()
    tiles: dict[str, Image.Image] = {}
    for mount in mounts:
        # An exr is radiance; making it a picture is the render's job.
        frame = into / f"{mount.name}.{scenario.outputs.format}"
        if scenario.outputs.format == "exr" or not frame.exists():
            raise FileNotFoundError(f"{frame}: render it, as png or jpg")
        tiles[mount.name] = _tile(frame, typeface, mount, mounts)

    out = into / "montage.png"
    _beside([_stacked(column, bands, tiles) for column in columns(mounts)]).save(out)
    return out
