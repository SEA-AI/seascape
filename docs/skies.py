"""The sky library: each photo down to `LOWEST_DEG`, its measured sun circled.

uv run python docs/skies.py docs/skies.jpg
"""

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from seascape import skies
from seascape.assets import download

TILE = (512, 150)
COLUMNS = 3
LABEL_PX = 20
LOWEST_DEG = -5.0
SUN = (255, 60, 40)


def tile(name: str, photo: skies.Photo) -> Image.Image:
    image = bpy.data.images.load(str(download(name, photo.url, photo.sha256)))
    w, h = image.size
    pixels = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    rgb = pixels.reshape(h, w, 4)[::-1, :, :3][: round(h * (90 - LOWEST_DEG) / 180)]
    rgb = rgb / (4 * np.median(rgb @ skies.LUMINANCE))
    ldr = (np.clip(rgb / (1 + rgb), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    return Image.fromarray(ldr).resize(TILE, Image.Resampling.LANCZOS)


def main(out: Path) -> None:
    photos = skies.library()
    rows = math.ceil(len(photos) / COLUMNS)
    sheet = Image.new("RGB", (TILE[0] * COLUMNS, (TILE[1] + LABEL_PX) * rows), "black")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    # Highest sun first, then the skies whose sun shows no disc.
    order = sorted(
        photos.items(),
        key=lambda p: (
            90.0 if p[1].sun_elevation_deg is None else -p[1].sun_elevation_deg
        ),
    )
    for k, (name, photo) in enumerate(order):
        x, y = (k % COLUMNS) * TILE[0], (k // COLUMNS) * (TILE[1] + LABEL_PX)
        sheet.paste(tile(name, photo), (x, y + LABEL_PX))
        elevation = photo.sun_elevation_deg
        sun = "no disc" if elevation is None else f"sun {elevation:.1f}°"
        draw.text((x + 6, y + 3), f"{name}  {sun}", fill="white", font=font)
        if elevation is not None:
            cx = x + (photo.sun_bearing_deg + 90) / 360 % 1 * TILE[0]
            cy = y + LABEL_PX + (90 - elevation) / (90 - LOWEST_DEG) * TILE[1]
            draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=SUN, width=2)
    sheet.save(out, quality=85)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
