"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote. Captions sit under each
frame, never on it: in an exr they would corrupt radiance.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from PIL.ImageFont import FreeTypeFont

from seascape.angles import wrapped_deg
from seascape.config import Mount, Scenario
from seascape.outputs import brand, canvas, preview

TILE_H = 260
CAPTION_H = 26
GUTTER = 4
FAN_R = 10

SHADE = (88, 88, 88)
LIT = (255, 200, 60)


def _apart_deg(a: Mount, b: Mount) -> float:
    return abs(wrapped_deg(a.nominal_azimuth_deg - b.nominal_azimuth_deg))


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
        start = mount.nominal_azimuth_deg - half - 90.0
        draw.pieslice(box, start, start + 2 * half, fill=fill)


def _tile(
    path: Path,
    typeface: FreeTypeFont,
    mount: Mount,
    mounts: Sequence[Mount],
) -> Image.Image:
    frame = preview.picture(path, mount.camera.agc)
    frame = ImageOps.scale(frame, TILE_H / frame.height, Image.Resampling.LANCZOS)

    tile = ImageOps.expand(frame, border=(0, 0, 0, CAPTION_H), fill=canvas.MATTE)
    draw = ImageDraw.Draw(tile)
    middle = TILE_H + CAPTION_H // 2
    _fan(draw, GUTTER + FAN_R, middle, mounts, mount)
    bearing = wrapped_deg(mount.nominal_azimuth_deg)
    caption = f"{mount.name}  {bearing:+.0f}°  {mount.camera.hfov_deg:.0f}° wide"
    text_px = draw.textlength(caption, font=typeface)
    x = max((tile.width - text_px) / 2, 2 * (GUTTER + FAN_R))
    draw.text((x, middle), caption, font=typeface, fill=canvas.INK, anchor="lm")
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
    # Ties go by name, which is unique, so the input's order never shows.
    for mount in sorted(mounts, key=lambda m: (-m.camera.hfov_deg, m.name)):
        seen_by = [
            column
            for column in laid
            if _apart_deg(mount, column[0]) <= column[0].camera.hfov_deg / 2
        ]
        if seen_by:
            min(seen_by, key=lambda column: _apart_deg(mount, column[0])).append(mount)
        else:
            laid.append([mount])

    cut_deg = _cut_deg([column[0].nominal_azimuth_deg for column in laid])

    def rightward(mount: Mount) -> float:
        return (mount.nominal_azimuth_deg - cut_deg) % 360.0

    laid.sort(key=lambda column: rightward(column[0]))
    return [sorted(column, key=rightward) for column in laid]


def sheet(scenario: Scenario, into: Path) -> Image.Image:
    """The frames in `into` as one picture, a clip by its middle frame: a row per band,
    a column per bearing.
    """
    bands = [band for band in scenario.outputs.bands if scenario.band_mounts(band)]
    if not bands:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")
    mounts = [m for band in bands for m in scenario.band_mounts(band)]
    typeface = brand.font()
    tiles: dict[str, Image.Image] = {}
    for mount, frame in preview.shown(scenario, into):
        # An exr is radiance; making it a picture is the render's job.
        if scenario.outputs.format == "exr" or not frame.exists():
            raise FileNotFoundError(f"{frame}: render it, as png or jpg")
        tiles[mount.name] = _tile(frame, typeface, mount, mounts)

    def cell(column: list[Mount], band: str) -> Image.Image:
        return canvas.beside(
            [tiles[m.name] for m in column if m.camera.band == band], gap_px=GUTTER
        )

    laid = columns(mounts)
    cells = [[cell(column, band) for column in laid] for band in bands]
    return canvas.table(cells, gap_px=GUTTER)


def compose(scenario: Scenario, into: Path) -> Path:
    """Write `montage.png` in `into`."""
    out = into / "montage.png"
    sheet(scenario, into).save(out)
    return out
