"""How labels.json decides a target is too faint for a box, and the check against the
same scene rendered without it.

uv run python -m docs.visibility docs/visibility.png
"""

import sys
import tempfile
from pathlib import Path

import bpy
import cv2
import numpy as np
from PIL import Image, ImageDraw

from docs.brand import FOCUS_RED, FOG_WHITE, NIGHT_BLUE, OCEAN_TEAL, SKY_GREY, font
from seascape import labels, render, scene
from seascape.config import load

SCENARIO = Path(__file__).with_name("visibility.toml")
TILE_PX = (260, 200)
PAD_PX, GAP_PX = 18, 20
# The colour scale's top: the JND sits mid-scale.
DELTA_E_MAX = 2 * labels.JND_DELTA_E
MASK, FLANK, TOP, UNLIT = (213, 122, 46), (46, 134, 193), (36, 199, 150), (34, 34, 38)
type Tile = tuple[Image.Image, str, tuple[str, str] | None]


def shots(
    out: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, float]]:
    """As `seascape render` renders it: the EO frame, the same with every target
    hidden, the object-index pass, all top row first, and each target's range by pass
    index.

    Hidden rather than removed: the hulls set the far clip, which the haze spans.
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
    """Around the target and its flanks, in the tiles' aspect."""
    ys, xs = np.nonzero(mask)
    aspect = TILE_PX[0] / TILE_PX[1]
    w = max(np.ptp(xs) + 1 + 4 * labels.FLANK_PX, (np.ptp(ys) + 1 + 16) * aspect)
    h = w / aspect
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    x0, y0 = max(round(cx - w / 2), 0), max(round(cy - h / 2), 0)
    return slice(y0, y0 + round(h)), slice(x0, x0 + round(w))


def tile(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb).resize(TILE_PX, Image.Resampling.NEAREST)


def tinted(
    rgb: np.ndarray, where: np.ndarray, colour: tuple[int, int, int]
) -> Image.Image:
    out = rgb.copy()
    out[where] = colour
    return tile(out)


def heat(delta_e: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.nan_to_num(delta_e) / DELTA_E_MAX, 0, 1)
    rgb = cv2.applyColorMap((255 * scaled).astype(np.uint8), cv2.COLORMAP_INFERNO)[
        ..., ::-1
    ]
    rgb[~mask] = UNLIT
    return rgb


def verdict(score: float, seen: bool) -> tuple[str, str]:
    return (
        (f"{score:.2f} ΔE: box", OCEAN_TEAL)
        if seen
        else (f"{score:.2f} ΔE: no box", FOCUS_RED)
    )


def steps(
    frame: np.ndarray, index: np.ndarray, without: np.ndarray, p: int, caption: str
) -> list[Tile]:
    """One target's tile in each step."""
    mask = index == p
    lab, lab_without = labels.cielab(frame), labels.cielab(without)
    behind = labels.background(index, lab, p)
    if behind is None:
        raise ValueError(f"target {p} has no background to show")
    delta_e = np.where(mask, np.linalg.norm(lab - behind, axis=2), np.nan)
    true_delta_e = np.where(mask, np.linalg.norm(lab - lab_without, axis=2), np.nan)
    score, true_score = labels.ricco(delta_e[mask]), labels.ricco(true_delta_e[mask])
    seen = labels.visible(index, frame, p)
    if seen != (score >= labels.JND_DELTA_E):
        raise ValueError("labels.visible and labels.ricco disagree")
    filled = np.where(mask[..., None], np.nan_to_num(behind), lab)
    rgb = cv2.cvtColor(filled.astype(np.float32), cv2.COLOR_Lab2RGB)
    background = (255 * np.clip(rgb, 0, 1)).round().astype(np.uint8)
    ys, xs = np.nonzero(mask)
    largest = np.argsort(delta_e[mask])[-labels.RICCO_PX :]
    marked = heat(delta_e, mask)
    marked[ys[largest], xs[largest]] = TOP
    crop = window(mask)

    def at(image: np.ndarray) -> np.ndarray:
        return image[crop]

    return [
        (tile(at(frame)), caption, None),
        (tinted(at(frame), at(mask), MASK), caption, None),
        (tinted(at(frame), at(labels.flanks(index, p)), FLANK), caption, None),
        (tile(at(background)), caption, None),
        (tile(at(heat(delta_e, mask))), caption, None),
        (tile(at(marked)), caption, verdict(score, seen)),
        (tile(at(without)), caption, None),
        (
            tile(at(heat(true_delta_e, mask))),
            caption,
            verdict(true_score, true_score >= labels.JND_DELTA_E),
        ),
    ]


def card(title: str, tiles: list[Tile], note: str | None = None) -> Image.Image:
    w = 3 * PAD_PX + 2 * TILE_PX[0]
    h = 2 * PAD_PX + 34 + TILE_PX[1] + 56
    img = Image.new("RGB", (w, h), FOG_WHITE)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [0, 0, w - 1, h - 1], radius=10, fill="white", outline=SKY_GREY
    )
    draw.text((PAD_PX, PAD_PX), title, fill=NIGHT_BLUE, font=font(20, "Bold"))
    y = PAD_PX + 34
    for i, (picture, caption, said) in enumerate(tiles):
        x = PAD_PX + i * (TILE_PX[0] + PAD_PX)
        img.paste(picture, (x, y))
        centre = x + TILE_PX[0] / 2
        draw.text(
            (centre, y + TILE_PX[1] + 8),
            caption,
            fill=NIGHT_BLUE,
            font=font(16),
            anchor="mt",
        )
        if said is not None:
            text, colour = said
            draw.text(
                (centre, y + TILE_PX[1] + 32),
                text,
                fill=colour,
                font=font(18, "Bold"),
                anchor="mt",
            )
    if note:
        draw.text(
            (w / 2, y + TILE_PX[1] + 34),
            note,
            fill=SKY_GREY,
            font=font(15),
            anchor="mt",
        )
    return img


TITLES = [
    "1. The 8-bit image, as delivered",
    "2. The target's pixels, from the object-index pass",
    f"3. Each row: up to {labels.FLANK_PX} px left and right of it",
    "4. Their median is the background, row by row",
    "5. ΔE*ab per pixel against it (brighter is more)",
    f"6. Do the {labels.RICCO_PX} largest average ≥ {labels.JND_DELTA_E} ΔE?",
    "7. Check: the same scene, rendered without the targets",
    "8. ΔE against the true background, pooled alike",
]


def main(out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        frame, without, index, ranges = shots(Path(tmp))
    columns = [
        steps(frame, index, without, p, f"{r / 1000:.1f} km away")
        for p, r in sorted(ranges.items())
    ]
    clear = cv2.dilate((index > 0).astype(np.uint8), np.ones((41, 41), np.uint8)) == 0
    shift = np.linalg.norm(labels.cielab(frame) - labels.cielab(without), axis=2)
    note = (
        f"Away from the targets the frames differ by ΔE {np.median(shift[clear]):.2f}."
    )
    cards = [
        card(title, [column[i] for column in columns], note if i == 6 else None)
        for i, title in enumerate(TITLES)
    ]
    w, h = cards[0].size
    sheet = Image.new("RGB", (2 * w + 3 * GAP_PX, 4 * h + 5 * GAP_PX), FOG_WHITE)
    for i, c in enumerate(cards):
        sheet.paste(
            c, (GAP_PX + (i % 2) * (w + GAP_PX), GAP_PX + (i // 2) * (h + GAP_PX))
        )
    sheet.quantize(128, dither=Image.Dither.NONE).save(out, optimize=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
