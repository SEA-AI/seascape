"""docs/outputs.md's figure: how labels.json drops a target too faint to see, checked
against the scene rendered without it.

uv run python -m docs.visibility docs/visibility.png
"""

import sys
import tempfile
from pathlib import Path

import bpy
import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw

from docs.brand import FOCUS_RED, NIGHT_BLUE, OCEAN_TEAL, SKY_GREY, font
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
    """`image` ("frame" or "without") around the ship, the ship boxed."""
    ys, xs = np.nonzero(m["mask"])
    w = max(2.5 * np.ptp(xs), 60.0)
    h = w * THUMB_PX[1] / THUMB_PX[0]
    x0, y0 = max(round(xs.mean() - w / 2), 0), max(round(ys.mean() - h / 2), 0)
    crop = m[image][y0 : y0 + round(h), x0 : x0 + round(w)]
    img = panel(crop, width=THUMB_PX[0])
    s = img.width / crop.shape[1]
    box = [(xs.min() - x0 - 2) * s, (ys.min() - y0 - 2) * s]
    box += [(xs.max() - x0 + 3) * s, (ys.max() - y0 + 3) * s]
    ImageDraw.Draw(img).rectangle(box, outline=NIGHT_BLUE, width=2)
    return img


def number_line(draw: ImageDraw.ImageDraw, x0: int, axis: int, w: int, m: dict) -> None:
    """Ship contrast, true contrast and clutter on one axis from 0; the threshold in
    red."""
    top = 1.3 * max(m["ship"], m["real"], m["sea"] + JND)

    def x(value: float) -> float:
        return x0 + w * value / top

    small = font(14)
    draw.line([x0, axis, x0 + w, axis], fill=SKY_GREY)
    draw.text((x0, axis + 8), "0", fill=SKY_GREY, font=small, anchor="mt")
    draw.line([x(m["sea"]), axis - 8, x(m["sea"]), axis + 8], fill=SKY_GREY, width=3)
    text = f"clutter  {m['sea']:.2f}"
    at = max(x(m["sea"]), x0 + 16 + draw.textlength(text, font=small) / 2)
    draw.text((at, axis + 16), text, fill=SKY_GREY, font=small, anchor="mt")
    rule = x(m["sea"] + JND)
    draw.line([rule, axis - 14, rule, axis + 14], fill=FOCUS_RED, width=3)
    text = f"threshold  {m['sea'] + JND:.2f}"
    draw.text((rule, axis + 34), text, fill=FOCUS_RED, font=small, anchor="mt")
    dots = [(m["ship"], False, "ship contrast"), (m["real"], True, "true contrast")]
    # Each label on the dot's outer side, so the two never stack.
    for (value, hollow, label), anchor in zip(sorted(dots), ("rb", "lb"), strict=True):
        cx = x(value)
        fill = "white" if hollow else NIGHT_BLUE
        draw.ellipse(
            [cx - 7, axis - 7, cx + 7, axis + 7], fill=fill, outline=NIGHT_BLUE, width=2
        )
        at = (cx + (8 if anchor == "lb" else -8), axis - 12)
        text = f"{label}  {value:.2f}"
        draw.text(at, text, fill=NIGHT_BLUE, font=small, anchor=anchor)
    word, colour = ("box", NIGHT_BLUE) if m["seen"] else ("no box", FOCUS_RED)
    draw.text(
        (x0 + w + 24, axis), word, fill=colour, font=font(28, "Medium"), anchor="lm"
    )


def examples(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    y: int,
    rows: tuple[tuple[str, dict, float], ...],
    km: float,
) -> int:
    """One line per example: label, both thumbnails, the number line, the verdict;
    then the legend. The y below them."""
    thumbs_x = PAD_PX + 136
    line_x = thumbs_x + 2 * THUMB_PX[0] + 8 + 2 * PAD_PX
    for name, m, r in rows:
        middle = y + THUMB_PX[1] // 2
        draw.text((PAD_PX, middle - 20), name, fill=NIGHT_BLUE, font=font(16, "Medium"))
        where = f"{r / 1000:.1f} km,\n{km:g} km visibility"
        draw.multiline_text((PAD_PX, middle), where, fill=SKY_GREY, font=font(13))
        for i, (image, caption) in enumerate(
            (("frame", "with ship"), ("without", "without ship"))
        ):
            x = thumbs_x + i * (THUMB_PX[0] + 8)
            sheet.paste(thumbnail(m, image), (x, y))
            draw.text((x, y + THUMB_PX[1] + 2), caption, fill=SKY_GREY, font=font(13))
        number_line(draw, line_x, middle, WIDTH_PX - line_x - 130, m)
        y += THUMB_PX[1] + 2 * PAD_PX + 8
    small = font(14)
    for hollow, text, x in (
        (False, "ship contrast: against the sea beside it", PAD_PX),
        (True, "true contrast: against the image without the ship", 300),
    ):
        fill = "white" if hollow else NIGHT_BLUE
        draw.ellipse([x, y + 4, x + 10, y + 14], fill=fill, outline=NIGHT_BLUE, width=2)
        draw.text((x + 16, y), text, fill=NIGHT_BLUE, font=small)
    draw.text(
        (PAD_PX, y + 24),
        "Values in ΔE*ab, colour difference as people see it. Threshold: clutter + "
        f"{JND}, the smallest difference people notice.",
        fill=SKY_GREY,
        font=small,
    )
    return y + 44


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
    steps = [
        (panel(marked), "1  Background, row by row",
         "median of the marked pixels; white: the ship's glow, skipped"),
        (panel(heat(rgb, mask, delta_e, hot), ship_at - corner),
         "2  Ship contrast: ship vs background",
         f"{K} strongest (ringed): ({sums[0]}) / {K} = {v['ship']:.2f}"),
        (panel(heat(rgb, beside, delta_e, hot), sea_at - corner),
         "3  Clutter: background vs itself",
         f"{len(sea_top)} strongest (ringed): ({sums[1]}"
         f"{' + …' if len(sea_top) > 5 else ''}) / {len(sea_top)} = {v['sea']:.2f}"),
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
        (PAD_PX, PAD_PX + 34), "If not, it gets no box.", fill=SKY_GREY, font=font(16)
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
