"""docs/outputs.md's figure: how labels.json drops a target too faint to see.

uv run python -m docs.visibility docs/visibility.png
"""

import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from PIL import Image, ImageDraw

from docs.brand import FOCUS_RED, FOG_WHITE, NIGHT_BLUE, OCEAN_TEAL, SKY_GREY, font
from seascape import labels, render, scene
from seascape.config import load

SCENARIO = Path(__file__).with_name("visibility.toml")
PAD_PX, LEFT_PX, WIDTH_PX, THUMB_PX = 24, 620, 1000, (300, 150)


def shots(out: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """As `seascape render` renders it: the EO frame and its object-index pass, top
    row first, and the targets' pass indices, nearest first."""
    scenario = load(SCENARIO)
    (mount,) = (m for m in scenario.mounts if m.camera.band == "eo")
    built = scene.build(scenario, "eo", cull=True)
    render._index_output(out).file_name = "eo."
    camera = built.cameras[mount.name]
    sc = bpy.context.scene
    sc.camera = camera
    sc.render.filepath = str(out / "frame")
    bpy.ops.render.render(write_still=True)
    frame = render._frame(out / f"frame.{scenario.outputs.format}")
    index = np.rint(render._pixels(out / "eo.index.exr")[::-1, :, 0]).astype(int)
    at = camera.matrix_world.translation
    anchors = [anchor for group in built.targets.values() for anchor in group]
    anchors.sort(key=lambda a: (a.matrix_world.translation - at).length)
    return frame, index, [a.pass_index for a in anchors]


def excess(frame: np.ndarray, index: np.ndarray, p: int) -> float:
    value = labels.excess_de(index, frame, p)
    if value is None:
        raise ValueError(f"target {p} has no background to show")
    return value


def row_through(frame: np.ndarray, index: np.ndarray, p: int) -> dict:
    """Around target `p`: the crop, its row with the most flank, and that row's ΔE*ab
    from the background line, with which columns are ship and which flank."""
    mask, beside = index == p, labels.flanks(index, p)
    behind = labels.background(labels.cielab(frame), mask, beside)
    if behind is None:
        raise ValueError(f"target {p} has no background to show")
    ys, xs = np.nonzero(mask | beside)
    crop = np.s_[ys.min() - 6 : ys.max() + 7, xs.min() : xs.max() + 1]
    delta_e = np.linalg.norm(labels.cielab(frame) - behind, axis=2)[crop]
    row = int(np.argmax(beside[crop].sum(axis=1) * mask[crop].any(axis=1)))
    return {
        "rgb": frame[crop],
        "row": row,
        "delta_e": np.nan_to_num(delta_e[row]),
        "ship": mask[crop][row],
        "flank": beside[crop][row],
    }


def runs(columns: np.ndarray) -> list[np.ndarray]:
    xs = np.flatnonzero(columns)
    return np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)


def zoomed(r: dict) -> tuple[Image.Image, float]:
    """The crop enlarged to the left column, pixels square, the row's parts outlined."""
    scale = LEFT_PX / r["rgb"].shape[1]
    size = (LEFT_PX, round(r["rgb"].shape[0] * scale))
    img = Image.fromarray(r["rgb"]).resize(size, Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(img)
    y0, y1 = r["row"] * scale, (r["row"] + 1) * scale
    for part, colour in (("flank", OCEAN_TEAL), ("ship", FOCUS_RED)):
        for run in runs(r[part]):
            box = [run[0] * scale, y0, (run[-1] + 1) * scale - 1, y1]
            draw.rectangle(box, outline=colour, width=3)
    return img, scale


def chart(r: dict, scale: float, height: int) -> Image.Image:
    """Each column of the row as a bar of its ΔE*ab, under its own pixel, with the
    ship's and the flanks' means as lines."""
    img = Image.new("RGB", (LEFT_PX, height), "white")
    draw = ImageDraw.Draw(img)
    base = height - 24
    top = 1.1 * r["delta_e"].max()

    def y(value: float) -> float:
        return base - (base - 8) * value / top

    for x, value in enumerate(r["delta_e"]):
        colour = (
            FOCUS_RED if r["ship"][x] else OCEAN_TEAL if r["flank"][x] else FOG_WHITE
        )
        draw.rectangle(
            [x * scale + 1, y(value), (x + 1) * scale - 2, base], fill=colour
        )
    draw.line([0, base, LEFT_PX, base], fill=SKY_GREY, width=1)
    for part, colour in (("ship", FOCUS_RED), ("flank", OCEAN_TEAL)):
        mean = y(float(r["delta_e"][r[part]].mean()))
        for x in range(0, LEFT_PX, 10):
            draw.line([x, mean, x + 5, mean], fill=colour, width=2)
    draw.text(
        (0, base + 4), "0 = the sea's colour on this row", fill="black", font=font(14)
    )
    draw.text(
        (LEFT_PX, base + 4), "dashed: means", fill="black", font=font(14), anchor="ra"
    )
    return img


def thumbnail(frame: np.ndarray, index: np.ndarray, p: int, seen: bool) -> Image.Image:
    """The frame around target `p`, boxed solid if labelled, dashed if dropped."""
    ys, xs = np.nonzero(index == p)
    w = int(max(4 * np.ptp(xs), 80))
    h = w * THUMB_PX[1] // THUMB_PX[0]
    x0, y0 = int(xs.mean() - w / 2), int(ys.mean() - h / 2)
    s = THUMB_PX[0] / w
    crop = frame[y0 : y0 + h, x0 : x0 + w]
    img = Image.fromarray(crop).resize(THUMB_PX, Image.Resampling.LANCZOS)
    left, top = (xs.min() - x0 - 2) * s, (ys.min() - y0 - 2) * s
    right, bottom = (xs.max() - x0 + 3) * s, (ys.max() - y0 + 3) * s
    draw = ImageDraw.Draw(img)
    if seen:
        draw.rectangle([left, top, right, bottom], outline=FOCUS_RED, width=2)
        return img
    for a in np.arange(left, right, 10):
        for edge in (top, bottom):
            draw.line([a, edge, min(a + 5, right), edge], fill=FOCUS_RED, width=2)
    for a in np.arange(top, bottom, 10):
        for edge in (left, right):
            draw.line([edge, a, edge, min(a + 5, bottom)], fill=FOCUS_RED, width=2)
    return img


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame, index, (near, far) = shots(Path(tmp))
    found = [excess(frame, index, p) for p in (near, far)]
    if not found[0] >= labels.MIN_EXCESS_DE > found[1]:
        raise ValueError("the near ship must be seen and the far one not")
    r = row_through(frame, index, near)
    zoom, scale = zoomed(r)
    body, label = font(18), font(18, "Medium")
    sheet = Image.new("RGB", (WIDTH_PX, 2000), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (PAD_PX, PAD_PX),
        "Does the ship differ from the sea more than the sea does from itself?",
        fill=NIGHT_BLUE,
        font=font(24, "Medium"),
    )

    # The verdicts first: the chart stretches to end level with them.
    x, y = PAD_PX + LEFT_PX + 2 * PAD_PX, PAD_PX + 48
    draw.text((x, y), "3  Ship mean minus flank mean", fill=NIGHT_BLUE, font=label)
    y += 28
    for p, value in zip((near, far), found, strict=True):
        seen = value >= labels.MIN_EXCESS_DE
        sheet.paste(thumbnail(frame, index, p, seen), (x, y))
        y += THUMB_PX[1] + 6
        verdict = "boxed" if seen else "dropped"
        draw.text((x, y), f"{value:.1f} ΔE*ab: {verdict}", fill="black", font=body)
        y += 36
    draw.text(
        (x, y), f"Under {labels.MIN_EXCESS_DE:g}: no box", fill=FOCUS_RED, font=label
    )
    bottom = y + 24

    x, y = PAD_PX, PAD_PX + 48
    draw.text((x, y), "1  One row through the ship", fill=NIGHT_BLUE, font=label)
    right = x + LEFT_PX
    for text, colour in (("ship", FOCUS_RED), ("flank", OCEAN_TEAL)):
        right -= draw.textlength(text, font=body)
        draw.text((right, y + 2), text, fill=colour, font=body)
        right -= 16
    y += 28
    sheet.paste(zoom, (x, y))
    y += zoom.height + 20
    draw.text((x, y), "2  Each pixel's ΔE*ab from the sea", fill=NIGHT_BLUE, font=label)
    y += 28
    sheet.paste(chart(r, scale, bottom - y), (x, y))
    sheet = sheet.crop((0, 0, WIDTH_PX, bottom + PAD_PX))
    sheet.quantize(160, dither=Image.Dither.NONE).save(out, optimize=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
