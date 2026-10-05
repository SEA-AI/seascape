"""The README's hero: one scene under each sky in SKIES, EO above LWIR, with
labels.json drawn on.

uv run seascape render docs/hero.toml -o out/clear/
uv run seascape render docs/hero.toml -o out/cumulus/ \
    --set 'sky.hdri = "sunflowers"' --set 'sky.sun_bearing_deg = -83.8'
uv run seascape render docs/hero.toml -o out/haze/ \
    --set 'sky.hdri = "overcast_soil"' --set 'sky.visibility_km = 3'
uv run python docs/hero.py out/ docs/hero.jpg
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SKIES = {
    "clear": "sun_elevation_deg = 35",
    "cumulus": 'hdri = "sunflowers"',
    "haze": 'hdri = "overcast_soil", visibility_km = 3',
}
GAP_PX = 8
BOX = (255, 214, 0)
HORIZON = (80, 220, 255)


def frame(
    run: Path,
    image: dict,
    annotations: list[dict],
    window: tuple[float, float, float, float],
    size: tuple[int, int],
    caption: str,
) -> Image.Image:
    """`window` (left, top, width, height) of the frame, scaled to `size`."""
    left, top, width, height = window
    s = size[0] / width
    img = Image.open(run / image["file_name"]).convert("RGB")
    img = img.crop((round(left), round(top), round(left + width), round(top + height)))
    img = img.resize(size, Image.Resampling.LANCZOS)
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
        box = draw.textbbox((x0, y0 - 18), text, font=font)
        draw.rectangle([box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2], fill="black")
        draw.text((x0, y0 - 18), text, fill=BOX, font=font)
    box = draw.textbbox((10, 8), caption, font=font)
    draw.rectangle([box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3], fill="black")
    draw.text((10, 8), caption, fill="white", font=font)
    return img


def column(run: Path) -> tuple[Image.Image, Image.Image]:
    """EO cut to the LWIR camera's field of view, and LWIR, as one size."""
    labels = json.loads((run / "labels.json").read_text())
    by_band = {im["band"]: im for im in labels["images"]}
    calibration = json.loads((run / "calibration.json").read_text())
    cameras = {c["band"]: c for c in calibration["cameras"]}
    eo, ir = by_band["eo"], by_band["ir"]
    size = (ir["width"], ir["height"])
    # Same centre, same angle: the window spans fx_eo / fx_ir of the LWIR frame.
    k = cameras["eo"]["K"][0][0] / cameras["ir"]["K"][0][0]
    w, h = size[0] * k, size[1] * k
    windows = {
        "eo": ((eo["width"] - w) / 2, (eo["height"] - h) / 2, w, h),
        "ir": (0.0, 0.0, *map(float, size)),
    }
    return tuple(
        frame(
            run,
            im,
            [a for a in labels["annotations"] if a["image_id"] == im["id"]],
            windows[band],
            size,
            SKIES[run.name] if band == "eo" else "LWIR",
        )
        for band, im in (("eo", eo), ("ir", ir))
    )


def main(out: Path, hero: Path) -> None:
    columns = [column(out / sky) for sky in SKIES]
    w, h = columns[0][0].size
    sheet = Image.new(
        "RGB", (len(columns) * (w + GAP_PX) - GAP_PX, 2 * h + GAP_PX), "white"
    )
    for i, pair in enumerate(columns):
        for j, img in enumerate(pair):
            sheet.paste(img, (i * (w + GAP_PX), j * (h + GAP_PX)))
    sheet.save(hero, quality=85)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
