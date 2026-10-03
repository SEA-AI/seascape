"""Every mesh rendered by seascape, side on and three-quarter on, for docs/assets.md.

uv run python docs/assets.py docs/assets.jpg
"""

import math
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from seascape import assets, render
from seascape.config import load

OPEN_SEA = Path(__file__).parent.parent / "scenarios" / "open-sea.toml"
TILE = (640, 360)
LABEL_PX = 24
HFOV_DEG = 45.0
# The hull spans this much of the frame side on.
FILL = 0.7
# Side on, then three-quarter on.
HEADINGS_DEG = (90.0, 210.0)


def shot(name: str, length_m: float, heading_deg: float, into: Path) -> Image.Image:
    range_m = length_m / (FILL * math.radians(HFOV_DEG))
    camera = (
        f'{{ kind = "eo", hfov_deg = {HFOV_DEG}, width_px = {TILE[0]}, '
        f"height_px = {TILE[1]} }}"
    )
    scenario = load(
        OPEN_SEA,
        [
            f"rig.height_m = {0.1 * length_m}",
            f'rig.pods = [{{ name = "bow", yaw_deg = 0.0, cameras = [{camera}] }}]',
            f'objects = [{{ asset = "{name}", range_m = {range_m}, bearing_deg = 0.0, '
            f"heading_deg = {heading_deg} }}]",
            'outputs.bands = ["eo"]',
        ],
    )
    return Image.open(render.render(scenario, into)[0]).convert("RGB")


def main(out: Path) -> None:
    meshes = assets.manifest()
    sheet = Image.new(
        "RGB", (TILE[0] * len(HEADINGS_DEG), (TILE[1] + LABEL_PX) * len(meshes))
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    with tempfile.TemporaryDirectory() as tmp:
        for row, (name, mesh) in enumerate(meshes.items()):
            y = row * (TILE[1] + LABEL_PX)
            textures = f"{len(mesh.texture_px)} textures" if mesh.texture_px else "flat"
            label = (
                f"{name}  {mesh.length_m:.0f} m  {mesh.triangles:,} tris  {textures}"
            )
            draw.text((6, y + 4), label, fill="white", font=font)
            for col, heading_deg in enumerate(HEADINGS_DEG):
                into = Path(tmp) / f"{name}-{col}"
                tile = shot(name, mesh.length_m, heading_deg, into)
                sheet.paste(tile, (col * TILE[0], y + LABEL_PX))
    sheet.save(out, quality=85)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
