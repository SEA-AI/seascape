"""docs/outputs.md's figure: how labels.json drops a target too faint to see, checked
against the scene rendered without it.

uv run python -m docs.visibility docs/visibility.png
"""

import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

import bpy
import numpy as np
from PIL import Image, ImageDraw

from docs.brand import FOCUS_RED, NIGHT_BLUE, OCEAN_TEAL, SKY_GREY, font
from seascape import labels, render, scene
from seascape.config import load

SCENARIO = Path(__file__).with_name("visibility.toml")
TILE_PX = (264, 158)
PAD_PX = 20
# Column widths: row label, frame, flanks, bars, verdict.
COLUMNS_PX = (118, TILE_PX[0], TILE_PX[0], 290, 200)
JND = labels.JND_DELTA_E
STEPS = [
    ("1  What the camera delivers", "the ship outlined"),
    (
        "2  The sea beside it",
        f"{labels.FLANK_PX}+ px left and right on every row; their median is the "
        "background",
    ),
    (
        "3  How different, how noisy",
        f"colour difference (ΔE*ab), strongest {labels.RICCO_PX} pixels",
    ),
    ("4  Verdict", ""),
]
HEADLINE = (
    "Does the ship stand out from the sea beside it more than that sea varies on its "
    "own?"
)
LEGEND = (
    "ΔE*ab: colour difference as people perceive it (CIELAB). "
    f"{JND}: the smallest difference people notice."
)


class Example(NamedTuple):
    crop: np.ndarray  # the frame around the ship
    without: np.ndarray  # the same, rendered without it
    mask: np.ndarray
    beside: np.ndarray
    ship: float  # labels.ricco of the ship's ΔE*ab
    sea: float  # labels.clutter of the flanks'
    seen: bool
    truth: float  # labels.ricco of the ship's ΔE*ab against the render without it
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
    """Around the ship and its flanks, in the tiles' aspect."""
    ys, xs = np.nonzero(mask)
    span = np.ptp(xs) + 1
    aspect = TILE_PX[0] / TILE_PX[1]
    w = max(span + 2 * labels.flank_px(span) + 8, (np.ptp(ys) + 1 + 16) * aspect)
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
    ship = labels.ricco(np.linalg.norm(lab[mask] - behind[mask], axis=1))
    texture = np.linalg.norm(lab[beside] - behind[beside], axis=1)
    sea = labels.clutter(texture, int(mask.sum()))
    seen = labels.visible(index, frame, p)
    # The figure's arithmetic and the rule must not drift apart.
    if seen != (ship >= labels.JND_DELTA_E + sea):
        raise ValueError("labels.visible disagrees with its parts")
    truth = labels.ricco(
        np.linalg.norm(lab[mask] - labels.cielab(without)[mask], axis=1)
    )
    crop = window(mask)
    return Example(
        frame[crop],
        without[crop],
        mask[crop],
        beside[crop],
        ship,
        sea,
        seen,
        truth,
        range_m,
    )


def tile(rgb: np.ndarray, size: tuple[int, int] = TILE_PX) -> Image.Image:
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def outlined(e: Example) -> Image.Image:
    img = tile(e.crop)
    ys, xs = np.nonzero(e.mask)
    sx, sy = TILE_PX[0] / e.mask.shape[1], TILE_PX[1] / e.mask.shape[0]
    box = [(xs.min() - 2) * sx, (ys.min() - 2) * sy, (xs.max() + 3) * sx]
    ImageDraw.Draw(img).rectangle(
        [*box, (ys.max() + 3) * sy], outline=NIGHT_BLUE, width=2
    )
    return img


def flanked(e: Example) -> Image.Image:
    teal = np.array(Image.new("RGB", (1, 1), OCEAN_TEAL))[0, 0]
    shaded = e.crop.astype(float)
    shaded[e.beside] = 0.6 * shaded[e.beside] + 0.4 * teal
    return tile(shaded.round().astype(np.uint8))


def wrap(draw: ImageDraw.ImageDraw, text: str, size: int, width: int) -> str:
    lines = [""]
    for word in text.split():
        line = f"{lines[-1]} {word}".strip()
        if lines[-1] and draw.textlength(line, font=font(size)) > width:
            lines.append(word)
        else:
            lines[-1] = line
    return "\n".join(lines)


def bars(e: Example) -> Image.Image:
    """Both differences on one axis from zero, and the red rule a ship must pass."""
    w, h = COLUMNS_PX[3], TILE_PX[1]
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    zero, need = 104, e.sea + JND
    top = 1.1 * max(e.ship, need)

    def x(value: float) -> float:
        return zero + (w - zero - 4) * value / top

    for i, (name, value, colour) in enumerate(
        (("ship vs sea", e.ship, NIGHT_BLUE), ("sea vs itself", e.sea, SKY_GREY))
    ):
        y = 24 + 50 * i
        draw.text((0, y), name, fill=NIGHT_BLUE, font=font(15))
        draw.text((0, y + 18), f"{value:.2f}", fill=SKY_GREY, font=font(14))
        draw.rectangle([zero, y + 4, max(x(value), zero + 1), y + 30], fill=colour)
    draw.line([zero, 14, zero, 124], fill=NIGHT_BLUE, width=1)
    draw.text((zero, 128), "0", fill=SKY_GREY, font=font(13), anchor="mt")
    rule = x(need)
    draw.line([rule, 0, rule, 124], fill=FOCUS_RED, width=2)
    label = f"needed to be seen  {need:.2f}"
    if rule + 4 + draw.textlength(label, font=font(14)) > w:
        draw.text((rule - 4, 0), label, fill=FOCUS_RED, font=font(14), anchor="rt")
    else:
        draw.text((rule + 4, 0), label, fill=FOCUS_RED, font=font(14), anchor="lt")
    return img


def verdict(e: Example) -> Image.Image:
    w = COLUMNS_PX[4]
    img = Image.new("RGB", (w, TILE_PX[1]), "white")
    draw = ImageDraw.Draw(img)
    draw.text(
        (0, 10),
        "box" if e.seen else "no box",
        fill=NIGHT_BLUE if e.seen else FOCUS_RED,
        font=font(34, "Medium"),
    )
    words = (
        f"stands out by more than the sea's own variation + {JND}"
        if e.seen
        else "does not"
    )
    text = wrap(draw, words, 16, w)
    draw.multiline_text((0, 60), text, fill=NIGHT_BLUE, font=font(16), spacing=3)
    bottom = draw.multiline_textbbox((0, 60), text, font=font(16), spacing=3)[3]
    sign = "≥" if e.seen else "<"
    draw.text(
        (0, bottom + 10),
        f"{e.ship:.2f} {sign} {e.sea:.2f} + {JND}",
        fill=SKY_GREY,
        font=font(14),
    )
    return img


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame, without, index, ranges = shots(Path(tmp))
    examples = [example(frame, without, index, p, r) for p, r in sorted(ranges.items())]
    haze_km = load(SCENARIO).sky.visibility_km
    xs = np.cumsum([PAD_PX, *(c + PAD_PX for c in COLUMNS_PX)])
    width = int(xs[-1])
    sheet = Image.new("RGB", (width, 2000), "white")
    draw = ImageDraw.Draw(sheet)
    y = PAD_PX
    draw.text((PAD_PX, y), HEADLINE, fill=NIGHT_BLUE, font=font(24, "Medium"))
    y += 32
    draw.text((PAD_PX, y), "If not, it gets no box.", fill=SKY_GREY, font=font(17))
    y += 40
    for x, (title, caption), column in zip(
        xs[1:-1], STEPS, COLUMNS_PX[1:], strict=True
    ):
        draw.text((x, y), title, fill=NIGHT_BLUE, font=font(18, "Medium"))
        draw.multiline_text(
            (x, y + 24), wrap(draw, caption, 14, column), fill=SKY_GREY, font=font(14)
        )
    y += 66
    for e in examples:
        draw.text(
            (PAD_PX, y + 4),
            "Visible" if e.seen else "Lost in haze",
            fill=NIGHT_BLUE,
            font=font(18, "Medium"),
        )
        draw.multiline_text(
            (PAD_PX, y + 30),
            f"{e.range_m / 1000:.1f} km away,\n{haze_km:g} km visibility",
            fill=SKY_GREY,
            font=font(14),
        )
        panels = (outlined(e), flanked(e), bars(e), verdict(e))
        for x, panel in zip(xs[1:-1], panels, strict=True):
            sheet.paste(panel, (int(x), y))
        y += TILE_PX[1] + PAD_PX
    y += 4
    draw.text(
        (PAD_PX, y),
        "Check: the same scene rendered without the ship",
        fill=NIGHT_BLUE,
        font=font(18, "Medium"),
    )
    y += 30
    small = (TILE_PX[0] // 2, TILE_PX[1] // 2)
    x = PAD_PX
    for e in examples:
        sheet.paste(tile(e.without, small), (x, y))
        same = (e.truth >= JND + e.sea) == e.seen
        text = f"true difference {e.truth:.2f} · " + (
            "same verdict" if same else "a different verdict"
        )
        draw.text(
            (x + small[0] + 12, y + small[1] // 2),
            text,
            fill=NIGHT_BLUE,
            font=font(16),
            anchor="lm",
        )
        x += small[0] + 12 + int(draw.textlength(text, font=font(16))) + 3 * PAD_PX
    y += small[1] + PAD_PX
    draw.text((PAD_PX, y), LEGEND, fill=SKY_GREY, font=font(14))
    y += 18 + PAD_PX
    sheet = sheet.crop((0, 0, width, y))
    sheet.quantize(256, dither=Image.Dither.NONE).save(out, optimize=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
