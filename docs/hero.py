"""The README's hero: EO above LWIR under each sky, with labels.json drawn on.

uv run --with supervision python -m docs.hero docs/hero.jpg
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw

from docs.brand import FOG_WHITE, MASK_COLOURS, NIGHT_BLUE, font

SCENARIO = Path(__file__).with_name("hero.toml")
SKIES = {
    "clear": ["sky.sun_elevation_deg = 35"],
    # The turn brings the photo's broken cloud into view.
    "cumulus": ['sky.hdri = "sunflowers"', "sky.sun_bearing_deg = -83.8"],
    "haze": ['sky.hdri = "overcast_soil"', "sky.visibility_km = 3"],
}
GAP_PX = 8


def render(scenario: Path, overrides: list[str], out: Path) -> None:
    sets = [arg for o in overrides for arg in ("--set", o)]
    subprocess.run(
        ["seascape", "render", str(scenario), "-o", str(out), *sets], check=True
    )


def tag(
    draw: ImageDraw.ImageDraw, at: tuple[float, float], text: str, fill: str, size: int
) -> None:
    tb = draw.textbbox(at, text, font=font(size))
    draw.rectangle([tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3], fill=fill)
    draw.text(at, text, fill="white", font=font(size))


def captioned(img: Image.Image, text: str, size: int = 15) -> Image.Image:
    tag(ImageDraw.Draw(img), (10, 8), text, NIGHT_BLUE, size)
    return img


def masks(
    image: dict,
    annotations: list[dict],
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> np.ndarray:
    """Each mask, cropped to `box` and resized to `size`."""
    resolution = (image["width"], image["height"])
    return np.stack(
        [
            np.asarray(
                Image.fromarray(
                    sv.rle_to_mask(a["segmentation"]["counts"], resolution).astype(
                        np.uint8
                    )
                    * 255
                ).resize(size, Image.Resampling.BILINEAR, box=box)
            )
            > 127
            for a in annotations
        ]
    )


def outlined(pic: np.ndarray, found: sv.Detections) -> np.ndarray:
    """`pic` with each target haloed, boxed and outlined, one colour per target."""
    colours = sv.ColorPalette.from_hex(MASK_COLOURS)
    by_index = sv.ColorLookup.INDEX
    grown = sv.Detections(
        xyxy=found.xyxy,
        mask=np.stack(
            [cv2.dilate(m.view(np.uint8), np.ones((3, 3))) > 0 for m in found.mask]
        ),
    )
    pic = sv.HaloAnnotator(
        color=colours, opacity=0.6, kernel_size=25, color_lookup=by_index
    ).annotate(pic, found)
    pic = sv.PolygonAnnotator(
        color=colours, thickness=1, color_lookup=by_index
    ).annotate(pic, grown)
    return sv.BoxAnnotator(color=colours, thickness=1, color_lookup=by_index).annotate(
        pic, found
    )


def lit(img: Image.Image, found: np.ndarray) -> Image.Image:
    """`img` with the targets in `found` drawn on."""
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(found), mask=found)
    pic = outlined(np.asarray(img)[..., ::-1].copy(), detections)
    return Image.fromarray(pic[..., ::-1])


def frame(
    run: Path,
    image: dict,
    annotations: list[dict],
    window: tuple[float, float, float, float],
    size: tuple[int, int],
    caption: str,
) -> Image.Image:
    """`window` is (left, top, width, height) in source pixels."""
    left, top, width, height = window
    s = size[0] / width
    img = Image.open(run / image["file_name"]).convert("RGB")
    box = (left, top, left + width, top + height)
    img = img.resize(size, Image.Resampling.LANCZOS, box=box)
    if annotations:
        img = lit(img, masks(image, annotations, box, size))
    draw = ImageDraw.Draw(img)

    def at(x: float, y: float) -> tuple[float, float]:
        return (x - left) * s, (y - top) * s

    draw.line([at(x, y) for x, y in image["horizon_px"]], fill=FOG_WHITE)
    for a in annotations:
        x0, y0 = at(*a["bbox"][:2])
        r = a["range_m"]
        dist = f"{r / 1000:.1f} km" if r >= 1000 else f"{r:.0f} m"
        text = f"{a['name']}  {dist}  {a['bearing_deg']:.1f}°"
        # Off the right edge a label shifts left, off the top it drops below the edge.
        text_w = draw.textlength(text, font=font(15))
        x = max(4, min(x0 + 4, img.width - text_w - 8))
        tag(draw, (x, max(4, y0 - 20)), text, NIGHT_BLUE, 15)
    return captioned(img, caption)


def column(run: Path) -> tuple[Image.Image, Image.Image]:
    """EO cropped to the LWIR field of view, and LWIR, both at LWIR size."""
    labels = json.loads((run / "labels.json").read_text())
    calibration = json.loads((run / "calibration.json").read_text())
    by_band = {im["band"]: im for im in labels["images"]}
    if len(by_band) != len(labels["images"]):
        raise ValueError(f"{run}: one camera per band, got {len(labels['images'])}")
    fx = {c["band"]: c["K"][0][0] for c in calibration["cameras"]}
    eo, ir = by_band["eo"], by_band["ir"]
    size = (ir["width"], ir["height"])
    # Assumes coaxial cameras.
    w, h = size[0] * fx["eo"] / fx["ir"], size[1] * fx["eo"] / fx["ir"]

    def boxes(im: dict) -> list[dict]:
        return [a for a in labels["annotations"] if a["image_id"] == im["id"]]

    return (
        frame(
            run,
            eo,
            boxes(eo),
            ((eo["width"] - w) / 2, (eo["height"] - h) / 2, w, h),
            size,
            "EO",
        ),
        frame(run, ir, boxes(ir), (0.0, 0.0, *size), size, "LWIR"),
    )


def main(hero: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        columns = []
        for name, overrides in SKIES.items():
            run = Path(tmp) / name
            render(SCENARIO, overrides, run)
            columns.append(column(run))
    w, h = columns[0][0].size
    sheet = Image.new(
        "RGB", (len(columns) * (w + GAP_PX) - GAP_PX, 2 * h + GAP_PX), "white"
    )
    for i, pair in enumerate(columns):
        for j, img in enumerate(pair):
            sheet.paste(img, (i * (w + GAP_PX), j * (h + GAP_PX)))
    sheet.save(hero, quality=85)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
