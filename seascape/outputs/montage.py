"""Compose a render's frames into one labelled image, for review.

No Blender: this reads the images `render` already wrote. Captions sit under each
frame, never on it: in an exr they would corrupt radiance.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from PIL.ImageFont import FreeTypeFont

from seascape.angles import wrapped_deg
from seascape.config import Mount, Scenario, sources
from seascape.outputs import agc, brand, canvas, labels, preview

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
    frame: Image.Image,
    typeface: FreeTypeFont,
    mount: Mount,
    mounts: Sequence[Mount],
) -> Image.Image:
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

    Raises:
        ValueError: No band has a camera.
        FileNotFoundError: A frame is missing, or the render is exr.
    """
    mounts = preview.cameras(scenario)
    if not mounts:
        raise ValueError(f"no frames for any of {scenario.outputs.bands} in {into}")
    pictures = {}
    for mount, frame in preview.shown(scenario, into):
        # An exr is radiance; making it a picture is the render's job.
        if scenario.outputs.format == "exr" or not frame.exists():
            raise FileNotFoundError(f"{frame}: render it, as png or jpg")
        pictures[mount.name] = _tile_high(frame, agc.Agc(mount.camera.agc))
    return _sheet(mounts, pictures)


def _tile_high(frame: Path, tone: agc.Agc) -> Image.Image:
    """`frame` as a picture `TILE_H` tall: scaled as it loads, so a sheet holds no
    full-size frame.
    """
    picture = preview.picture(frame, tone)
    return ImageOps.scale(picture, TILE_H / picture.height, Image.Resampling.LANCZOS)


def _sheet(mounts: Sequence[Mount], pictures: Mapping[str, Image.Image]) -> Image.Image:
    """`pictures`, by their mount's name, tiled a row per band and a column per
    bearing.
    """
    typeface = brand.font()
    tiles = {m.name: _tile(pictures[m.name], typeface, m, mounts) for m in mounts}

    def cell(column: list[Mount], band: str) -> Image.Image:
        return canvas.beside(
            [tiles[m.name] for m in column if m.camera.band == band], gap_px=GUTTER
        )

    laid = columns(mounts)
    bands = dict.fromkeys(m.camera.band for m in mounts)
    return canvas.table(
        [[cell(c, band) for c in laid] for band in bands], gap_px=GUTTER
    )


def _recorded(mount: Mount) -> str | None:
    """The video `mount` shows in, as `render` names it: its rig's video of it, or its
    own. None for a stitch's right camera, whose frames show in its left's.
    """
    if (recorded := mount.video) is None:
        return mount.name
    name, video = recorded
    return name if mount.camera_name == sources(video)[0] else None


def video_frames(scenario: Scenario, into: Path, folder: Path) -> list[Path]:
    """Every camera's video in `into`, frame by frame as labels.json lists them, laid
    out as `sheet` lays out its frames, as numbered pngs in `folder`, padded to even
    sides for H.264. A thermal clip is toned as its video is.

    Limitation: a stitch's tile is fanned and captioned as its left camera.

    Args:
        scenario: The scenario `into` rendered.
        into: The render's folder, holding labels.json.
        folder: Where the pngs go; made if missing.

    Returns:
        The pngs, in order.

    Raises:
        FileNotFoundError: No camera has frames in labels.json, or a frame is missing.
        ValueError: The clips differ in length.
    """
    clips = labels.cameras(into)
    shown, mounts = {}, []
    for m in preview.cameras(scenario):
        if (v := _recorded(m)) in clips:
            shown[m.name] = v
            mounts.append(m)
    if not mounts:
        raise FileNotFoundError(f"{into}: no frames of any camera")
    settings = labels.agcs(into)
    tones = {
        n: agc.Agc(settings[v], 1 / scenario.outputs.fps) for n, v in shown.items()
    }
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for i, frames in enumerate(zip(*(clips[v] for v in shown.values()), strict=True)):
        pictures = {
            n: _tile_high(into / f.file_name, tones[n])
            for n, f in zip(shown, frames, strict=True)
        }
        picture = _sheet(mounts, pictures)
        w, h = picture.size
        path = folder / f"{i:05d}.png"
        ImageOps.expand(picture, (0, 0, w % 2, h % 2), fill=canvas.MATTE).save(path)
        written.append(path)
    return written


def compose(scenario: Scenario, into: Path) -> Path:
    """Write `montage.png` in `into`."""
    out = into / "montage.png"
    sheet(scenario, into).save(out)
    return out
