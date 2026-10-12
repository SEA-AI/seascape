"""docs/outputs.md's figure: how labels.json drops a target too faint to see, checked
against the scene rendered without it.

uv run python -m docs.visibility docs/visibility.png
"""

import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import bpy
import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from docs.brand import FOCUS_RED, FOG_WHITE, NIGHT_BLUE, OCEAN_TEAL, SKY_GREY, font
from seascape import labels, render, scene
from seascape.config import load

SCENARIO = Path(__file__).with_name("visibility.toml")
JND, K = labels.JND_DELTA_E, labels.RICCO_PX
PAD_PX, IMAGE_PX, WIDTH_PX, THUMB_PX = 20, 640, 1080, (160, 96)
DARK = np.array(ImageColor.getrgb(NIGHT_BLUE), float)


def shots(out: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, float]]:
    """As `seascape render` renders it: the EO frame, the same with every target
    hidden, the object-index pass, all top row first, and each target's range by pass
    index. Hidden rather than removed: the hulls can set the far clip, which the haze
    spans."""
    scenario = load(SCENARIO)
    (mount,) = (m for m in scenario.mounts if m.camera.band == "eo")
    built = scene.build(scenario, "eo", cull=True)
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


def measure(frame: np.ndarray, without: np.ndarray, index: np.ndarray, p: int) -> dict:
    """The rule's quantities for target `p`, from labels' own functions."""
    lab, mask, beside = labels.cielab(frame), index == p, labels.flanks(index, p)
    behind = labels.background(lab, mask, beside)
    if behind is None:
        raise ValueError(f"target {p} has no background to show")
    raw = np.linalg.norm(lab - np.nan_to_num(behind), axis=2)
    delta_e = np.where(
        mask,
        labels.pooled(raw, mask),
        np.where(beside, labels.pooled(raw, beside), raw),
    )
    n = int(mask.sum())
    ship, sea = labels.ricco(delta_e[mask], n), labels.ricco(delta_e[beside], n)
    seen = labels.visible(index, frame, p)
    if seen != (ship >= JND + sea):
        raise ValueError("labels.visible disagrees with its parts")
    difference = np.linalg.norm(lab - labels.cielab(without), axis=2)
    real = labels.pooled(difference, mask)[mask]
    return {"frame": frame, "mask": mask, "beside": beside, "behind": behind,
            "delta_e": delta_e, "n": n, "ship": ship, "sea": sea, "seen": seen,
            "real": labels.ricco(real, n), "without": without}  # fmt: skip


def strongest(m: dict, where: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """labels.strongest of `where`'s ΔE*ab, largest first, and their (row, column)."""
    ys, xs = np.nonzero(where)
    picked = labels.strongest(m["delta_e"][ys, xs], m["n"])
    picked = picked[np.argsort(m["delta_e"][ys[picked], xs[picked]])[::-1]]
    return m["delta_e"][ys[picked], xs[picked]], np.stack([ys[picked], xs[picked]])


def panel(
    rgb: np.ndarray, rings: np.ndarray | None = None, width: int = IMAGE_PX
) -> Image.Image:
    """`rgb` enlarged to `width`, pixels kept square, `rings` circled."""
    scale = width / rgb.shape[1]
    size = (width, round(rgb.shape[0] * scale))
    img = Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)
    r = max(scale, 5.0) * 1.1
    for y, x in [] if rings is None else rings.T:
        cx, cy = (x + 0.5) * scale, (y + 0.5) * scale
        ImageDraw.Draw(img).ellipse(
            [cx - r, cy - r, cx + r, cy + r], outline=FOCUS_RED, width=2
        )
    return img


def heat(
    rgb: np.ndarray, where: np.ndarray, delta_e: np.ndarray, top: float
) -> np.ndarray:
    """`rgb` darkened, `where` from Night Blue at 0 to white at `top`."""
    out = rgb.astype(float) * 0.45
    out[where] = DARK + np.clip(delta_e[where] / top, 0, 1)[:, None] * (255 - DARK)
    return out.round().astype(np.uint8)


def thumbnail(m: dict, image: str) -> Image.Image:
    """`image` ("frame" or "without") around the ship. The frame carries the verdict,
    the box solid if labels.json keeps it and dashed if not; the image without the
    ship only marks where it was."""
    ys, xs = np.nonzero(m["mask"])
    w = max(2.5 * np.ptp(xs), 60.0)
    h = w * THUMB_PX[1] / THUMB_PX[0]
    x0, y0 = max(round(xs.mean() - w / 2), 0), max(round(ys.mean() - h / 2), 0)
    crop = m[image][y0 : y0 + round(h), x0 : x0 + round(w)]
    img = panel(crop, width=THUMB_PX[0])
    s = img.width / crop.shape[1]
    box = [(xs.min() - x0 - 2) * s, (ys.min() - y0 - 2) * s]
    box += [(xs.max() - x0 + 3) * s, (ys.max() - y0 + 3) * s]
    draw = ImageDraw.Draw(img)
    if image == "without":
        draw.rectangle(box, outline="white", width=2)
    elif m["seen"]:
        draw.rectangle(box, outline=NIGHT_BLUE, width=2)
    else:
        dashed(draw, box)
    return img


def dashed(draw: ImageDraw.ImageDraw, box: list[float], dash: int = 6) -> None:
    left, top, right, bottom = box
    edges = ((left, top, right, top), (right, top, right, bottom))
    edges += ((left, bottom, right, bottom), (left, top, left, bottom))
    for x0, y0, x1, y1 in edges:
        length = max(abs(x1 - x0), abs(y1 - y0))
        for start in range(0, round(length), 2 * dash):
            a, b = start / length, min(start + dash, length) / length
            start_at = (x0 + a * (x1 - x0), y0 + a * (y1 - y0))
            end_at = (x0 + b * (x1 - x0), y0 + b * (y1 - y0))
            draw.line([start_at, end_at], fill=FOCUS_RED, width=2)


def tick(draw: ImageDraw.ImageDraw, x: float, y0: float, y1: float) -> None:
    """The true contrast: white on the ship's stripe, outlined."""
    draw.line([x, y0, x, y1], fill=OCEAN_TEAL, width=5)
    draw.line([x, y0 + 2, x, y1 - 2], fill="white", width=1)


def reach(m: dict) -> float:
    """The longest of `m`'s bars."""
    return max(m["ship"], m["real"], m["sea"] + JND)


def runs_px(
    draw: ImageDraw.ImageDraw,
    parts: Sequence[tuple[str, str]],
    face: ImageFont.FreeTypeFont,
) -> float:
    return sum(draw.textlength(text, font=face) for text, _ in parts)


def runs(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    parts: Sequence[tuple[str, str]],
    face: ImageFont.FreeTypeFont,
) -> None:
    """`parts` left to right from `x`, each text in its colour."""
    for text, colour in parts:
        draw.text((x, y), text, fill=colour, font=face, anchor="lm")
        x += draw.textlength(text, font=face)


def race(
    draw: ImageDraw.ImageDraw, x0: int, middle: int, w: int, m: dict, top: float
) -> float:
    """The sea's stripe, its clutter and the JND stacked to the threshold, over the
    ship's, the true contrast ticked on it, on an axis from 0 to `top`; each value
    just past its stripe's end. Where the labels end."""
    small, half = font(14), 2
    # Each number in its stripe's colour, so the label is the legend.
    sea_runs = [
        (f"{m['sea']:.2f}", SKY_GREY),
        (f" + {JND}*", FOCUS_RED),
        (f" = {m['sea'] + JND:.2f}", NIGHT_BLUE),
    ]
    ship_runs = [(f"{m['ship']:.2f}", NIGHT_BLUE), ("  ·  ", SKY_GREY)]
    ship_runs += [(f"true {m['real']:.2f}", OCEAN_TEAL)]
    widths = [runs_px(draw, r, small) for r in (sea_runs, ship_runs)]
    # The longest stripe still leaves its label room.
    span = w - 40 - max(widths)

    def x(value: float) -> float:
        return x0 + 40 + (span - 12) * value / top

    sea_y, ship_y = middle - 14, middle + 14
    for label, row in (("sea", sea_y), ("ship", ship_y)):
        draw.text((x0, row), label, fill=SKY_GREY, font=small, anchor="lm")
    draw.rectangle([x(0), sea_y - half, x(m["sea"]), sea_y + half], fill=SKY_GREY)
    draw.rectangle(
        [x(m["sea"]), sea_y - half, x(m["sea"] + JND), sea_y + half], fill=FOCUS_RED
    )
    draw.rectangle([x(0), ship_y - half, x(m["ship"]), ship_y + half], fill=NIGHT_BLUE)
    tick(draw, x(m["real"]), ship_y - 8, ship_y + 8)
    sea_end, ship_end = x(m["sea"] + JND), max(x(m["ship"]), x(m["real"]))
    runs(draw, sea_end + 10, sea_y, sea_runs, small)
    runs(draw, ship_end + 10, ship_y, ship_runs, small)
    return max(sea_end + widths[0], ship_end + widths[1]) + 10


def examples(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    y: int,
    rows: tuple[tuple[str, dict, float], ...],
    km: float,
) -> int:
    """One line per example: label, both thumbnails with the verdict, the race; the
    notes boxed beside the last race. The y below them."""
    # One axis for every example, so their bars compare.
    top = 1.05 * max(reach(m) for _, m, _ in rows)
    thumbs_x = PAD_PX + 136
    line_x = thumbs_x + 2 * THUMB_PX[0] + 8 + 2 * PAD_PX
    race_w = WIDTH_PX - line_x - PAD_PX
    small = font(14)
    for name, m, r in rows:
        row_y, middle = y, y + THUMB_PX[1] // 2
        draw.text((PAD_PX, middle - 20), name, fill=NIGHT_BLUE, font=font(16, "Medium"))
        where = f"{r / 1000:.1f} km,\n{km:g} km visibility"
        draw.multiline_text((PAD_PX, middle), where, fill=SKY_GREY, font=font(13))
        for i, (image, caption) in enumerate(
            (("frame", "with ship"), ("without", "without ship"))
        ):
            x = thumbs_x + i * (THUMB_PX[0] + 8)
            sheet.paste(thumbnail(m, image), (x, y))
            draw.text((x, y + THUMB_PX[1] + 2), caption, fill=SKY_GREY, font=font(13))
        end = race(draw, line_x, middle, race_w, m, top)
        y += THUMB_PX[1] + 2 * PAD_PX + 8
    # Each key word in the colour it has in the figure.
    notes = [
        [("ship", NIGHT_BLUE), (": vs the sea beside it.", SKY_GREY)],
        [("true", OCEAN_TEAL), (": vs the image without the ship.", SKY_GREY)],
        [(f"*{JND}", FOCUS_RED), (": just noticeable (Mahy et al. 1994).", SKY_GREY)],
        [("Dashed box", FOCUS_RED), (": no label.", SKY_GREY)],
    ]
    line_px, inset = 18, 10
    width = max(runs_px(draw, note, small) for note in notes) + 2 * inset
    box = [WIDTH_PX - PAD_PX - width, row_y, WIDTH_PX - PAD_PX, 0]
    box[3] = box[1] + len(notes) * line_px + 2 * inset
    # Beside the last race: an example whose labels reach this far needs a new place.
    if box[0] < end + PAD_PX:
        raise ValueError("the notes no longer fit beside the last example's race")
    draw.rectangle(box, outline=FOG_WHITE, width=1)
    for k, note in enumerate(notes):
        runs(draw, box[0] + inset, box[1] + inset + (k + 0.5) * line_px, note, small)
    return max(y - PAD_PX, box[3])


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame, without, index, ranges = shots(Path(tmp))
    (near_m, v), (far_m, h) = (
        (r, measure(frame, without, index, p)) for p, r in sorted(ranges.items())
    )
    if not v["seen"] or h["seen"]:
        raise ValueError("the near ship must be seen and the far one not")
    ys, xs = np.nonzero(v["mask"] | v["beside"])
    top, left = max(ys.min() - 4, 0), max(xs.min() - 2, 0)
    crop = np.s_[top : ys.max() + 5, left : xs.max() + 3]
    rgb, mask, beside, delta_e = (
        v[k][crop] for k in ("frame", "mask", "beside", "delta_e")
    )

    # One row's flanks in teal, the glow skipped beside the ship in white.
    row = int(np.argmax(beside.sum(axis=1) * mask.any(axis=1)))
    marked = rgb.copy()
    hull, flank = np.flatnonzero(mask[row]), np.flatnonzero(beside[row])
    marked[row, flank.min() : hull[0]] = marked[row, hull[-1] + 1 : flank.max() + 1] = (
        255
    )
    marked[row, beside[row]] = (np.array(ImageColor.getrgb(OCEAN_TEAL)) + 255) // 2
    swatch = cv2.cvtColor(v["behind"][crop][row, hull[:1]][None], cv2.COLOR_Lab2RGB)
    ship_top, ship_at = strongest(v, v["mask"])
    sea_top, sea_at = strongest(v, v["beside"])
    corner = np.array([[top], [left]])
    hot = float(ship_top[0])
    sums = [
        " + ".join(f"{d:.2f}" for d in values[:5]) for values in (ship_top, sea_top)
    ]
    patch = f"{labels.POOL_PX}\N{MULTIPLICATION SIGN}{labels.POOL_PX} px"

    def pooled(values: np.ndarray, sum_text: str, total: float, single: str) -> str:
        """One patch whose pixels all carry its value reads as that patch."""
        if len(np.unique(np.round(values, 2))) == 1:
            return f"{single} (ringed): {total:.2f}"
        more = " + …" if len(values) > 5 else ""
        return (
            f"{len(values)} strongest, each averaged over {patch} (ringed):\n"
            f"({sum_text}{more}) / {len(values)} = {total:.2f}"
        )

    steps = [
        (panel(marked), "1  Background, row by row",
         "median of the marked pixels; white: the ship's glow, skipped"),
        (panel(heat(rgb, mask, delta_e, hot), ship_at - corner),
         "2  Ship contrast: ship vs background",
         pooled(ship_top, sums[0], v["ship"],
                f"Strongest {patch} patch, averaged as the eye does")),
        (panel(heat(rgb, beside, delta_e, hot), sea_at - corner),
         "3  Clutter: background vs itself",
         pooled(sea_top, sums[1], v["sea"],
                f"Strongest {patch} patch at the ship's share")),
    ]  # fmt: skip
    sheet = Image.new("RGB", (WIDTH_PX, 2000), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (PAD_PX, PAD_PX),
        "Does the ship stand out more than the sea varies on its own?",
        fill=NIGHT_BLUE,
        font=font(24, "Medium"),
    )
    draw.text(
        (PAD_PX, PAD_PX + 34),
        f"If not, it gets no box. Differences in ΔE*ab (CIELAB), each pooled over the "
        f"{labels.POOL_PX}\N{MULTIPLICATION SIGN}{labels.POOL_PX} px a viewer's eye "
        "sums, as in Ricco's law.",
        fill=SKY_GREY,
        font=font(16),
    )
    y, text_x = PAD_PX + 74, 2 * PAD_PX + IMAGE_PX
    for i, (img, title, body) in enumerate(steps):
        sheet.paste(img, (PAD_PX, y))
        draw.text((text_x, y), title, fill=NIGHT_BLUE, font=font(17, "Medium"))
        if i == 0:
            colour = tuple(int(c) for c in np.rint(255 * swatch[0, 0].clip(0, 1)))
            draw.rectangle([text_x, y + 30, text_x + 22, y + 52], fill=colour)
        at = (text_x + (32 if i == 0 else 0), y + 28)
        draw.multiline_text(at, body, fill=NIGHT_BLUE, font=font(15), spacing=4)
        y += max(img.height, 60) + PAD_PX
    rows = (("Visible", v, near_m), ("Lost in haze", h, far_m))
    y = examples(sheet, draw, y, rows, load(SCENARIO).sky.visibility_km or 0.0)
    sheet = sheet.crop((0, 0, WIDTH_PX, y + PAD_PX))
    sheet.quantize(160, dither=Image.Dither.NONE).save(out, optimize=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
