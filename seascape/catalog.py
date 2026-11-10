"""Asset sheets, and the manifest lines measured from a mesh file."""

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from seascape import assets, render
from seascape.blender import meshes
from seascape.config import Band, Scenario, load
from seascape.outputs import brand, canvas
from seascape.outputs.labels import Labels

TILE = (480, 270)
COLUMNS = 4
# Buoys, debris and turbines through a long lens, as a ship sees them.
HFOV_DEG = {"hull": 45.0, "buoy": 10.0, "debris": 10.0, "turbine": 10.0}
# A hull or debris spans this much of the tile's width, anything else's freeboard
# this much of its height.
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
    rig = {
        # A tenth of the range up, aimed at the waterline.
        "height_m": 0.1 * range_m,
        "pitch_deg": -math.degrees(math.atan(0.1)),
        "cameras": {
            band: {
                "band": band,
                "hfov_deg": wide_deg,
                "width_px": 2 * TILE[0],
                "height_px": 2 * TILE[1],
            }
        },
    }
    at = {"asset": name, "range_m": range_m, "bearing_deg": 0.0}
    if isinstance(mesh, assets.Turbine):
        # It faces the wind, so the wind turns it.
        placed = {"farms": [at], "sea": {"wind_from_deg": HEADING_DEG}}
    else:
        placed = {"objects": [at | {"heading_deg": HEADING_DEG}]}
    built = load(
        scenario,
        [placed, {"outputs": {"bands": [band], "duration_s": 0.0}}],
    )
    # The mesh alone, through one camera: TOML cannot set the ring or the ownship's
    # hull to none, and the scenario's rigs merge with any it is given.
    alone: dict[str, Any] = built.model_dump(mode="json") | {
        "targets": None,
        "rigs": {"bow": rig},
    }
    alone["ownship"] |= {"asset": None}
    built = Scenario.model_validate(alone)
    frame = Image.open(render.render(built, into)[0]).convert("RGB")
    labelled = Labels.read(into)
    (label,) = labelled.annotations
    x, y, w, h = label.bbox
    left = min(max(x + w // 2 - TILE[0] // 2, 0), TILE[0])
    top = min(max(y + h // 2 - TILE[1] // 2, 0), TILE[1])
    return frame.crop((left, top, left + TILE[0], top + TILE[1]))


def sheet(scenario: Path, out: Path, band: Band = "eo") -> None:
    """Write a sheet of every asset in the manifest, shot in `scenario`, to `out`."""
    font = brand.font(15)
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, mesh in assets.manifest().items():
            tile = shot(name, mesh, scenario, band, Path(tmp) / name)
            n = len(mesh.texture_px)
            textures = f"{n} texture{'s' * (n > 1)}" if n else "flat"
            # The LWIR build replaces every material, so textures show only in EO.
            label = f"{name}  {mesh.size}  {mesh.triangles:,} tris"
            label += f"  {textures}" if band == "eo" else ""
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, TILE[0], 22), fill="black")
            draw.text((6, 3), label, fill="white", font=font)
            tiles.append(tile)
    # Judgement: a contact sheet to look at, so a smaller file over fidelity.
    canvas.grid(tiles, COLUMNS, matte="black").save(out, quality=85)


def measure(path: Path) -> str:
    """The manifest lines for the mesh at `path`."""
    triangles, texture_px = meshes.measure(path)
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
