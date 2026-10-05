"""The README's hero: EO above LWIR under each sky, with labels.json drawn on.

uv run python docs/hero.py docs/hero.jpg
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SCENARIO = Path(__file__).with_name("hero.toml")
SKIES = {
    "clear": ["sky.sun_elevation_deg = 35"],
    # The turn brings the photo's broken cloud into view.
    "cumulus": ['sky.hdri = "sunflowers"', "sky.sun_bearing_deg = -83.8"],
    "haze": ['sky.hdri = "overcast_soil"', "sky.visibility_km = 3"],
}
GAP_PX = 8
BOX = (255, 214, 0)
HORIZON = (80, 220, 255)


def render(scenario: Path, overrides: list[str], out: Path) -> None:
    sets = [arg for o in overrides for arg in ("--set", o)]
    subprocess.run(
        ["seascape", "render", str(scenario), "-o", str(out), *sets], check=True
    )


def captioned(img: Image.Image, text: str, size: int = 14) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=size)
    tb = draw.textbbox((10, 8), text, font=font)
    draw.rectangle([tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3], fill="black")
    draw.text((10, 8), text, fill="white", font=font)
    return img


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
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=14)

    def at(x: float, y: float) -> tuple[float, float]:
        return (x - left) * s, (y - top) * s

    draw.line([at(x, y) for x, y in image["horizon_px"]], fill=HORIZON)
    for a in annotations:
        x, y, w, h = a["bbox"]
        (x0, y0), (x1, y1) = at(x, y), at(x + w, y + h)
        draw.rectangle([x0, y0, x1, y1], outline=BOX, width=2)
        r = a["range_m"]
        dist = f"{r / 1000:.1f} km" if r >= 1000 else f"{r:.0f} m"
        text = f"{a['name']}  {dist}  {a['bearing_deg']:.1f}°"
        tb = draw.textbbox((x0, y0 - 18), text, font=font)
        draw.rectangle([tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2], fill="black")
        draw.text((x0, y0 - 18), text, fill=BOX, font=font)
    return captioned(img, caption)


def column(run: Path, caption: str) -> tuple[Image.Image, Image.Image]:
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
            caption,
        ),
        frame(run, ir, boxes(ir), (0.0, 0.0, *map(float, size)), size, "LWIR"),
    )


def main(hero: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        columns = []
        for name, overrides in SKIES.items():
            run = Path(tmp) / name
            render(SCENARIO, overrides, run)
            columns.append(column(run, ", ".join(overrides)))
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
