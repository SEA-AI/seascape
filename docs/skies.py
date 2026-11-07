"""The sky library: each photo down to `LOWEST_DEG`, its measured sun circled.

uv run python docs/skies.py docs/skies.jpg
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from seascape.assets import download, skies
from seascape.blender import blend
from seascape.outputs import canvas

TILE = (512, 150)
COLUMNS = 3
LABEL_PX = 20
LOWEST_DEG = -5.0
SUN = (255, 60, 40)


def tile(name: str, photo: skies.Photo) -> Image.Image:
    pixels = blend.read(download(name, photo.url, photo.sha256))
    h = len(pixels)
    rgb = pixels[::-1, :, :3][: round(h * (90 - LOWEST_DEG) / 180)]
    rgb = rgb / (4 * np.median(rgb @ skies.LUMINANCE))
    ldr = (np.clip(rgb / (1 + rgb), 0, 1) ** (1 / skies.GAMMA) * 255).astype(np.uint8)
    return Image.fromarray(ldr).resize(TILE, Image.Resampling.LANCZOS)


def main(out: Path) -> None:
    photos = skies.library()
    font = canvas.font(14)
    cells = []
    # Highest sun first, then the skies whose sun shows no disc.
    order = sorted(
        photos.items(),
        key=lambda p: (
            90.0 if p[1].sun_elevation_deg is None else -p[1].sun_elevation_deg
        ),
    )
    for name, photo in order:
        cell = ImageOps.expand(
            tile(name, photo), border=(0, LABEL_PX, 0, 0), fill="black"
        )
        draw = ImageDraw.Draw(cell)
        elevation = photo.sun_elevation_deg
        sun = "no disc" if elevation is None else f"sun {elevation:.1f}°"
        draw.text((6, 3), f"{name}  {sun}", fill="white", font=font)
        if elevation is not None:
            cx = (photo.sun_bearing_deg + 90) / 360 % 1 * TILE[0]
            cy = LABEL_PX + (90 - elevation) / (90 - LOWEST_DEG) * TILE[1]
            draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline=SUN, width=2)
        cells.append(cell)
    canvas.grid(cells, COLUMNS, matte="black").save(out, quality=85)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
