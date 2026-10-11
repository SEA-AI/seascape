"""docs/outputs.md's figure: how labels.json drops a target too faint to see, checked
against the scene rendered without it.

uv run python -m docs.visibility docs/visibility.png
"""

import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import bpy
import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageText

from docs.brand import FOCUS_RED, NIGHT_BLUE, OCEAN_TEAL, SKY_GREY, font
from seascape import labels, render, scene
from seascape.config import load

SCENARIO = Path(__file__).with_name("visibility.toml")
JND = labels.JND_DELTA_E
PAD_PX = 20
THUMB_PX = (220, 132)
LINE_PX = 640
WIDTH_PX = PAD_PX + THUMB_PX[0] + 2 * PAD_PX + LINE_PX + PAD_PX + 110 + PAD_PX
ZOOM_PX = 660
DARK = np.array(ImageColor.getrgb(NIGHT_BLUE), float)
HEADLINE = (
    "Does the ship stand out from the sea or sky beside it more than that background "
    "varies on its own?"
)
METHOD = (
    "If not, it gets no box. Method: an order-statistic test after Rohling's OS-CFAR "
    "(1983), on ΔE*ab, the colour difference people perceive (CIELAB), with a "
    f"just-noticeable margin of {JND}."
)
REAL_SEA = (
    "Ship, against the real sea: the same measure, taken against the scene rendered "
    "without the ship."
)


class Example(NamedTuple):
    crop: np.ndarray  # the frame around the ship
    mask: np.ndarray
    beside: np.ndarray  # labels.flanks
    behind: np.ndarray  # labels.background
    delta_e: np.ndarray  # each pixel's ΔE*ab from its row's background
    ship: float  # labels.ricco of the ship's pixels
    sea: float  # labels.ricco of the flanks'
    seen: bool  # labels.visible
    truth: float  # labels.ricco of the ship against the render without it
    range_m: float


def shots(out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, float]]:
    """As `seascape render` renders it: the EO frame, the same with every target
    hidden, the object-index pass, all top row first, and each target's range by pass
    index.

    Hidden rather than removed: the hulls can set the far clip, which the haze spans.
    """
    scenario = load(SCENARIO)
    (mount,) = (m for m in scenario.mounts if m.camera.band == "eo")
    built = scene.build(scenario, "eo")
    render._index_output(out).file_name = "eo."
    camera = built.cameras[mount.name]
    sc = bpy.context.scene
    sc.camera = camera
    anchors = [anchor for group in built.targets.values() for anchor in group]

    def shot(name: str) -> np.ndarray:
        sc.render.filepath = str(out / name)
        bpy.ops.render.render(write_still=True)
        return render._frame(out / f"{name}.{scenario.outputs.format}")

    frame = shot("with")
    index = np.rint(render._pixels(out / "eo.index.exr")[::-1, :, 0]).astype(int)
    for anchor in anchors:
        for part in [anchor, *anchor.children_recursive]:
            part.hide_render = True
    without = shot("without")
    at = camera.matrix_world.translation
    ranges = {
        a.pass_index: (a.matrix_world.translation - at).xy.length for a in anchors
    }
    return frame, without, index, ranges


def window(mask: np.ndarray) -> tuple[slice, slice]:
    """Around the ship and its flanks, in the thumbnails' aspect."""
    ys, xs = np.nonzero(mask)
    span = np.ptp(xs) + 1
    aspect = THUMB_PX[0] / THUMB_PX[1]
    reach = labels.FLANK_GAP_PX + labels.flank_px(span)
    w = max(span + 2 * reach + 8, (np.ptp(ys) + 1 + 16) * aspect)
    h = w / aspect
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x0, y0 = max(round(cx - w / 2), 0), max(round(cy - h / 2), 0)
    return slice(y0, y0 + round(h)), slice(x0, x0 + round(w))


def example(
    frame: np.ndarray, without: np.ndarray, index: np.ndarray, p: int, range_m: float
) -> Example:
    lab = labels.cielab(frame)
    mask = index == p
    beside = labels.flanks(index, p)
    behind = labels.background(lab, mask, beside)
    if behind is None:
        raise ValueError(f"target {p} has no background to show")
    delta_e = np.linalg.norm(lab - np.nan_to_num(behind), axis=2)
    n = int(mask.sum())
    ship, sea = labels.ricco(delta_e[mask], n), labels.ricco(delta_e[beside], n)
    seen = labels.visible(index, frame, p)
    # The figure's arithmetic and the rule must not drift apart.
    if seen != (ship >= JND + sea):
        raise ValueError("labels.visible disagrees with its parts")
    truth = labels.ricco(
        np.linalg.norm(lab[mask] - labels.cielab(without)[mask], axis=1), n
    )
    crop = window(mask)
    return Example(
        frame[crop],
        mask[crop],
        beside[crop],
        behind[crop],
        delta_e[crop],
        ship,
        sea,
        seen,
        truth,
        range_m,
    )


def need(e: Example) -> float:
    return e.sea + JND


def tile(rgb: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def outline(img: Image.Image, mask: np.ndarray) -> None:
    ys, xs = np.nonzero(mask)
    sx, sy = img.width / mask.shape[1], img.height / mask.shape[0]
    ImageDraw.Draw(img).rectangle(
        [
            (xs.min() - 2) * sx,
            (ys.min() - 2) * sy,
            (xs.max() + 3) * sx,
            (ys.max() + 3) * sy,
        ],
        outline=NIGHT_BLUE,
        width=2,
    )


def paragraph(
    draw: ImageDraw.ImageDraw,
    at: tuple[int, int],
    text: str,
    width: int,
    size: int = 15,
    fill: str = NIGHT_BLUE,
) -> int:
    """`text` wrapped to `width` at `at`; the y below it."""
    wrapped = ImageText.Text(text, font(size), spacing=3)
    wrapped.wrap(width)
    draw.text(at, wrapped, fill=fill)
    return int(wrapped.get_bbox(at)[3])


def heat(
    crop: np.ndarray, where: np.ndarray, delta_e: np.ndarray, top: float
) -> np.ndarray:
    """The crop darkened, `where` from Night Blue at 0 to white at `top`."""
    out = crop.astype(float) * 0.45
    out[where] = DARK + np.clip(delta_e[where] / top, 0, 1)[:, None] * (255 - DARK)
    return out.round().astype(np.uint8)


def ringed(img: Image.Image, shape: tuple[int, ...], points: np.ndarray) -> Image.Image:
    draw = ImageDraw.Draw(img)
    sx, sy = img.width / shape[1], img.height / shape[0]
    r = max(sx, sy, 5.0) * 1.1
    for py, px in points:
        cx, cy = (px + 0.5) * sx, (py + 0.5) * sy
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=FOCUS_RED, width=2)
    return img


def strongest(
    delta_e: np.ndarray, where: np.ndarray, target_px: int
) -> tuple[np.ndarray, np.ndarray]:
    """labels.strongest of `where`'s values, largest first, and their (row, column)."""
    ys, xs = np.nonzero(where)
    picked = labels.strongest(delta_e[ys, xs], target_px)
    picked = picked[np.argsort(delta_e[ys[picked], xs[picked]])[::-1]]
    return delta_e[ys[picked], xs[picked]], np.stack([ys[picked], xs[picked]], axis=1)


def summed(values: np.ndarray) -> str:
    shown = " + ".join(f"{d:.1f}" for d in values[:6])
    return shown + (f" + … ({len(values)} values)" if len(values) > 6 else "")


def worked(sheet: Image.Image, draw: ImageDraw.ImageDraw, y: int, e: Example) -> int:
    """How `e`'s numbers are made, one step per row; the y below them."""
    n, f = int(e.mask.sum()), int(e.beside.sum())
    k = min(labels.RICCO_PX, n)
    ship_top, ship_at = strongest(e.delta_e, e.mask, n)
    sea_top, sea_at = strongest(e.delta_e, e.beside, n)
    m = len(sea_top)
    ys, xs = np.nonzero(e.mask | e.beside)
    zoom = np.s_[
        max(ys.min() - 4, 0) : ys.max() + 5, max(xs.min() - 2, 0) : xs.max() + 3
    ]
    crop, mask, beside = e.crop[zoom], e.mask[zoom], e.beside[zoom]
    delta_e, behind = e.delta_e[zoom], e.behind[zoom]
    ship_at = ship_at - [zoom[0].start, zoom[1].start]
    sea_at = sea_at - [zoom[0].start, zoom[1].start]
    size = (ZOOM_PX, round(ZOOM_PX * crop.shape[0] / crop.shape[1]))

    row = int(np.argmax(beside.sum(axis=1) * mask.any(axis=1)))
    marked = crop.copy()
    marked[row, beside[row]] = ImageColor.getrgb(OCEAN_TEAL)
    # The skipped glow, left white: between each flank and the ship on that row.
    flank, hull = np.flatnonzero(beside[row]), np.flatnonzero(mask[row])
    left, right = flank[flank < hull[0]], flank[flank > hull[-1]]
    if len(left):
        marked[row, left[-1] + 1 : hull[0]] = 255
    if len(right):
        marked[row, hull[-1] + 1 : right[0]] = 255
    one = tile(marked, size)
    sy = size[1] / crop.shape[0]
    ImageDraw.Draw(one).rectangle(
        [0, row * sy - 2, size[0] - 1, (row + 1) * sy + 1], outline=FOCUS_RED, width=1
    )
    median = cv2.cvtColor(behind[row, mask[row]][:1][None], cv2.COLOR_Lab2RGB)[0, 0]
    swatch = tuple(int(c) for c in np.clip(np.rint(255 * median), 0, 255))
    top = float(ship_top[0])
    count = (
        f"{m} = round({labels.RICCO_PX} \N{MULTIPLICATION SIGN} {f} background "
        f"pixels / {n} ship pixels): the same share of the background as "
        f"{labels.RICCO_PX} pixels are of the ship."
    )
    if k < labels.RICCO_PX:
        count += f" Scaled by {k}/{labels.RICCO_PX}, as the ship has {n} pixels."
    steps = [
        (
            "1  Background colour, row by row",
            one,
            f"The middle colour (median) of the {int(beside[row].sum())} pixels of "
            "sea or sky beside the ship on the marked row, skipping the "
            f"{labels.FLANK_GAP_PX} pixels next to the ship (its glow, left white); "
            "shown in the swatch. Every row gets its own.",
            None,
        ),
        (
            "2  Ship: how far each pixel is from its row's background",
            ringed(tile(heat(crop, mask, delta_e, top), size), crop.shape, ship_at),
            f"Its {labels.RICCO_PX} strongest pixels, ringed: "
            f"({summed(ship_top)}) / {labels.RICCO_PX} = {e.ship:.2f}",
            "ΔE*ab per pixel; lighter is more different, one scale for 2 and 3.",
        ),
        (
            "3  Background: how far it is from its own median",
            ringed(tile(heat(crop, beside, delta_e, top), size), crop.shape, sea_at),
            f"Its {m} strongest pixels, ringed: ({summed(sea_top)}) / {m} = "
            f"{e.sea:.2f}",
            count,
        ),
    ]
    text_x = PAD_PX + ZOOM_PX + PAD_PX
    text_w = WIDTH_PX - text_x - PAD_PX
    for i, (title, img, body, small) in enumerate(steps):
        sheet.paste(img, (PAD_PX, y))
        draw.text((text_x, y), title, fill=NIGHT_BLUE, font=font(17, "Medium"))
        tx = text_x
        if i == 0:
            draw.rectangle([tx, y + 28, tx + 22, y + 50], fill=swatch, outline=SKY_GREY)
            tx += 32
        bottom = paragraph(draw, (tx, y + 28), body, text_w - (tx - text_x))
        if small:
            bottom = paragraph(draw, (text_x, bottom + 6), small, text_w, 13, SKY_GREY)
        y = max(y + size[1], bottom) + PAD_PX
    sign, word = ("≥", "box") if e.seen else ("<", "no box")
    draw.text(
        (PAD_PX, y - 4),
        f"Needed to be seen = {e.sea:.2f} + {JND} (the smallest colour difference "
        f"people notice) = {need(e):.2f}.   {e.ship:.2f} {sign} {need(e):.2f}, so: "
        f"{word}.",
        fill=NIGHT_BLUE,
        font=font(18, "Medium"),
    )
    return y + 36


def number_lines(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    y: int,
    examples: list[Example],
    haze_km: float,
) -> int:
    """A thumbnail and one number line per example; the y below them."""
    x0 = PAD_PX + THUMB_PX[0] + 2 * PAD_PX
    for e in examples:
        top = 1.08 * max(e.ship, e.truth, need(e))

        def x(value: float, top: float = top) -> float:
            return x0 + LINE_PX * value / top

        img = tile(e.crop, THUMB_PX)
        outline(img, e.mask)
        sheet.paste(img, (PAD_PX, y))
        name = "Visible" if e.seen else "Lost in haze"
        below = y + THUMB_PX[1] + 4
        draw.text((PAD_PX, below), name, fill=NIGHT_BLUE, font=font(16, "Medium"))
        draw.text(
            (PAD_PX + draw.textlength(name, font=font(16, "Medium")) + 8, below + 2),
            f"{e.range_m / 1000:.1f} km away, {haze_km:g} km visibility",
            fill=SKY_GREY,
            font=font(14),
        )
        axis = y + 76
        draw.line([x0, axis, x0 + LINE_PX, axis], fill=SKY_GREY, width=1)
        draw.text((x0, axis + 8), "0", fill=SKY_GREY, font=font(13), anchor="mt")
        draw.line([x(e.sea), axis - 8, x(e.sea), axis + 8], fill=SKY_GREY, width=3)
        text = f"background varies on its own  {e.sea:.2f}"
        half = draw.textlength(text, font=font(14)) / 2
        draw.text(
            (max(x(e.sea), x0 + 16 + half), axis + 12),
            text,
            fill=SKY_GREY,
            font=font(14),
            anchor="mt",
        )
        rule = x(need(e))
        draw.line([rule, axis - 44, rule, axis + 44], fill=FOCUS_RED, width=2)
        draw.text(
            (rule, axis + 48),
            f"needed to be seen  {need(e):.2f}",
            fill=FOCUS_RED,
            font=font(14),
            anchor="mt",
        )
        for value, hollow, text, dy in (
            (e.ship, False, f"ship  {e.ship:.2f}", -38),
            (e.truth, True, f"ship, against the real sea  {e.truth:.2f}", -62),
        ):
            cx = x(value)
            draw.ellipse(
                [cx - 7, axis - 7, cx + 7, axis + 7],
                fill="white" if hollow else NIGHT_BLUE,
                outline=NIGHT_BLUE,
                width=2,
            )
            draw.text(
                (cx, axis + dy), text, fill=NIGHT_BLUE, font=font(14), anchor="mt"
            )
        word, colour = ("box", NIGHT_BLUE) if e.seen else ("no box", FOCUS_RED)
        draw.text(
            (x0 + LINE_PX + PAD_PX, axis),
            word,
            fill=colour,
            font=font(30, "Medium"),
            anchor="lm",
        )
        y += THUMB_PX[1] + 2 * PAD_PX + 6
    return y


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame, without, index, ranges = shots(Path(tmp))
    examples = [example(frame, without, index, p, r) for p, r in sorted(ranges.items())]
    sheet = Image.new("RGB", (WIDTH_PX, 3000), "white")
    draw = ImageDraw.Draw(sheet)
    y = PAD_PX
    draw.text((PAD_PX, y), HEADLINE, fill=NIGHT_BLUE, font=font(24, "Medium"))
    y = paragraph(draw, (PAD_PX, y + 32), METHOD, WIDTH_PX - 2 * PAD_PX, 16, SKY_GREY)
    y += PAD_PX
    draw.text(
        (PAD_PX, y),
        "How the numbers are made, for the visible ship",
        fill=NIGHT_BLUE,
        font=font(20, "Medium"),
    )
    y = worked(sheet, draw, y + 36, next(e for e in examples if e.seen))
    draw.line([PAD_PX, y, WIDTH_PX - PAD_PX, y], fill=SKY_GREY, width=1)
    y += PAD_PX
    draw.text((PAD_PX, y), "Both examples", fill=NIGHT_BLUE, font=font(20, "Medium"))
    haze_km = load(SCENARIO).sky.visibility_km or 0.0
    y = number_lines(sheet, draw, y + 34, examples, haze_km)
    draw.text((PAD_PX, y), REAL_SEA, fill=SKY_GREY, font=font(14))
    sheet = sheet.crop((0, 0, WIDTH_PX, y + 20 + PAD_PX))
    sheet.quantize(160, dither=Image.Dither.NONE).save(out, optimize=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
