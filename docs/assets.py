"""The asset sheet in docs/assets.md, and the measured lines of a new manifest entry.

uv run python docs/assets.py sheet docs/assets.jpg
uv run python docs/assets.py measure hull.glb
"""

import math
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from seascape import assets, render, scene
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


def sheet(out: Path) -> None:
    meshes = assets.manifest()
    page = Image.new(
        "RGB", (TILE[0] * len(HEADINGS_DEG), (TILE[1] + LABEL_PX) * len(meshes))
    )
    draw = ImageDraw.Draw(page)
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
                page.paste(tile, (col * TILE[0], y + LABEL_PX))
    page.save(out, quality=85)


def measure(path: Path) -> None:
    triangles, texture_px = scene.measure(path)
    print(f'sha256 = "{assets.digest(path)}"')
    print(f"triangles = {triangles}\ntexture_px = {list(texture_px)}")


if __name__ == "__main__":
    commands = {"sheet": sheet, "measure": measure}
    if len(sys.argv) != 3 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](Path(sys.argv[2]))
