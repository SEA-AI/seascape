"""Asset sheets, and the manifest lines measured from a mesh file."""

import json
import math
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from seascape import assets, render, scene
from seascape.config import Band, load

TILE = (480, 270)
COLUMNS = 4
# Buoys and debris through a long lens, as a ship sees them.
HFOV_DEG = {"hull": 45.0, "buoy": 10.0, "debris": 10.0}
# A hull or debris spans this much of the tile's width, a buoy's freeboard this much
# of its height.
FILL = 0.7
# Three-quarter on, bow towards the camera's left.
HEADING_DEG = 210.0


def shot(
    name: str, mesh: assets.Asset, scenario: Path, band: Band, into: Path
) -> Image.Image:
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
    built = load(
        scenario,
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
    frame = Image.open(render.render(built, into)[0]).convert("RGB")
    (label,) = json.loads((into / "labels.json").read_text())["annotations"]
    x, y, w, h = label["bbox"]
    left = min(max(x + w // 2 - TILE[0] // 2, 0), TILE[0])
    top = min(max(y + h // 2 - TILE[1] // 2, 0), TILE[1])
    return frame.crop((left, top, left + TILE[0], top + TILE[1]))


def sheet(scenario: Path, out: Path, band: Band = "eo") -> None:
    meshes = assets.manifest()
    rows = math.ceil(len(meshes) / COLUMNS)
    page = Image.new("RGB", (TILE[0] * COLUMNS, TILE[1] * rows))
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=15)
    with tempfile.TemporaryDirectory() as tmp:
        for k, (name, mesh) in enumerate(meshes.items()):
            x, y = (k % COLUMNS) * TILE[0], (k // COLUMNS) * TILE[1]
            page.paste(shot(name, mesh, scenario, band, Path(tmp) / name), (x, y))
            n = len(mesh.texture_px)
            textures = f"{n} texture{'s' * (n > 1)}" if n else "flat"
            # The LWIR build replaces every material, so textures show only in EO.
            label = f"{name}  {mesh.size}  {mesh.triangles:,} tris"
            label += f"  {textures}" if band == "eo" else ""
            draw.rectangle((x, y, x + TILE[0], y + 22), fill="black")
            draw.text((x + 6, y + 3), label, fill="white", font=font)
    page.save(out, quality=85)


def measure(path: Path) -> str:
    triangles, texture_px = scene.measure(path)
    lines = [
        f'sha256 = "{assets.digest(path)}"',
        f"triangles = {triangles}",
        f"texture_px = {list(texture_px)}",
    ]
    if path.suffix == ".gltf":
        gltf = json.loads(path.read_text())
        named = [*gltf.get("buffers", []), *gltf.get("images", [])]
        uris = (item.get("uri", "") for item in named)
        for uri in sorted(u for u in uris if u and not u.startswith("data:")):
            sha256 = assets.digest(path.parent / uri)
            lines.append(f'files."{uri}" = {{ url = "", sha256 = "{sha256}" }}')
    return "\n".join(lines)
