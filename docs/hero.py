"""The README's hero: each frame of a render, side by side, with labels.json drawn on.

uv run seascape render docs/hero.toml -o out/
uv run python docs/hero.py out/ docs/hero.jpg
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HEIGHT_PX = 540
GAP_PX = 8
BOX = (255, 214, 0)
HORIZON = (80, 220, 255)


def frame(run: Path, image: dict, annotations: list[dict]) -> Image.Image:
    img = Image.open(run / image["file_name"]).convert("RGB")
    s = HEIGHT_PX / img.height
    img = img.resize((round(img.width * s), HEIGHT_PX), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=15)
    draw.line([(x * s, y * s) for x, y in image["horizon_px"]], fill=HORIZON)
    for a in annotations:
        x, y, w, h = (v * s for v in a["bbox"])
        draw.rectangle([x, y, x + w, y + h], outline=BOX, width=2)
        r = a["range_m"]
        dist = f"{r / 1000:.1f} km" if r >= 1000 else f"{r:.0f} m"
        text = f"{a['name']}  {dist}  {a['bearing_deg']:.1f}°"
        left, top, right, bottom = draw.textbbox((x, y - 20), text, font=font)
        draw.rectangle([left - 3, top - 2, right + 3, bottom + 2], fill="black")
        draw.text((x, y - 20), text, fill=BOX, font=font)
    draw.text((10, 8), image["band"].upper(), fill="white", font=font)
    return img


def main(run: Path, out: Path) -> None:
    labels = json.loads((run / "labels.json").read_text())
    frames = [
        frame(run, im, [a for a in labels["annotations"] if a["image_id"] == im["id"]])
        for im in labels["images"]
    ]
    width = sum(f.width for f in frames) + GAP_PX * (len(frames) - 1)
    sheet = Image.new("RGB", (width, HEIGHT_PX), "white")
    x = 0
    for f in frames:
        sheet.paste(f, (x, 0))
        x += f.width + GAP_PX
    sheet.save(out, quality=85)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
