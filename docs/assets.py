"""The asset sheets in docs/assets.md, and the measured lines of a new manifest entry.

uv run python docs/assets.py sheet docs/assets_eo.jpg
uv run python docs/assets.py sheet docs/assets_ir.jpg ir
uv run python docs/assets.py measure hull.glb
"""

import json
import math
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from seascape import assets, render, scene
from seascape.config import Band, load

OPEN_SEA = Path(__file__).parent.parent / "scenarios" / "open-sea.toml"
TILE = (480, 270)
COLUMNS = 4
# Buoys and debris through a long lens, as a ship sees them.
HFOV_DEG = {"hull": 45.0, "buoy": 10.0, "debris": 10.0}
# A hull or debris spans this much of the tile's width, a buoy's freeboard this much
# of its height.
FILL = 0.7
# Three-quarter on, bow towards the camera's left.
HEADING_DEG = 210.0


def shot(name: str, mesh: assets.Asset, band: Band, into: Path) -> Image.Image:
    """The tile, cut from a frame twice its size around the mesh's label."""
    hfov_deg = HFOV_DEG[mesh.kind]
    if isinstance(mesh, assets.Hull | assets.Debris):
        size_m, across_rad = mesh.length_m, math.radians(hfov_deg)
    else:
        size_m = mesh.height_m - mesh.draught_m
        across_rad = math.radians(hfov_deg) * TILE[1] / TILE[0]
    range_m = size_m / (FILL * across_rad)
    # Twice the pixels over twice the tangent: the tile's pixel, with room to centre.
    wide_deg = math.degrees(2 * math.atan(2 * math.tan(math.radians(hfov_deg) / 2)))
    camera = (
        f'{{ kind = "{band}", hfov_deg = {wide_deg}, width_px = {2 * TILE[0]}, '
        f"height_px = {2 * TILE[1]} }}"
    )
    scenario = load(
        OPEN_SEA,
        [
            # A tenth of the range up, aimed at the waterline.
            f"rig.height_m = {0.1 * range_m}",
            f"rig.pitch_deg = {-math.degrees(math.atan(0.1))}",
            # The default clips the water in front of a camera this low.
            f"rig.near_clip_m = {0.01 * size_m}",
            f'rig.pods = [{{ name = "bow", yaw_deg = 0.0, cameras = [{camera}] }}]',
            f'objects = [{{ asset = "{name}", range_m = {range_m}, bearing_deg = 0.0, '
            f"heading_deg = {HEADING_DEG} }}]",
            f'outputs.bands = ["{band}"]',
        ],
    )
    frame = Image.open(render.render(scenario, into)[0]).convert("RGB")
    (label,) = json.loads((into / "labels.json").read_text())["annotations"]
    x, y, w, h = label["bbox"]
    left = min(max(x + w // 2 - TILE[0] // 2, 0), TILE[0])
    top = min(max(y + h // 2 - TILE[1] // 2, 0), TILE[1])
    return frame.crop((left, top, left + TILE[0], top + TILE[1]))


def sheet(out: Path, band: Band = "eo") -> None:
    meshes = assets.manifest()
    rows = math.ceil(len(meshes) / COLUMNS)
    page = Image.new("RGB", (TILE[0] * COLUMNS, TILE[1] * rows))
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=15)
    with tempfile.TemporaryDirectory() as tmp:
        for k, (name, mesh) in enumerate(meshes.items()):
            x, y = (k % COLUMNS) * TILE[0], (k // COLUMNS) * TILE[1]
            page.paste(shot(name, mesh, band, Path(tmp) / name), (x, y))
            n = len(mesh.texture_px)
            textures = f"{n} texture{'s' * (n > 1)}" if n else "flat"
            # The LWIR build replaces every material, so textures show only in EO.
            label = f"{name}  {mesh.size}  {mesh.triangles:,} tris"
            label += f"  {textures}" if band == "eo" else ""
            draw.rectangle((x, y, x + TILE[0], y + 22), fill="black")
            draw.text((x + 6, y + 3), label, fill="white", font=font)
    page.save(out, quality=85)


def measure(path: Path) -> None:
    triangles, texture_px = scene.measure(path)
    print(f'sha256 = "{assets.digest(path)}"')
    print(f"triangles = {triangles}\ntexture_px = {list(texture_px)}")
    if path.suffix == ".gltf":
        gltf = json.loads(path.read_text())
        named = [*gltf.get("buffers", []), *gltf.get("images", [])]
        uris = (item.get("uri", "") for item in named)
        for uri in sorted(u for u in uris if u and not u.startswith("data:")):
            sha256 = assets.digest(path.parent / uri)
            print(f'files."{uri}" = {{ url = "", sha256 = "{sha256}" }}')


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["sheet", out]:
            sheet(Path(out))
        case ["sheet", out, "eo" | "ir" as band]:
            sheet(Path(out), band)
        case ["measure", mesh]:
            measure(Path(mesh))
        case _:
            sys.exit(__doc__)
