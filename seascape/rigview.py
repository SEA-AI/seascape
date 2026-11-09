"""Where a scenario's rigs sit on its ownship, to confirm a mount."""

import math
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw

from seascape.blender import workbench
from seascape.blender.meshes import box_corners, mesh_objects
from seascape.blender.rigs import add_rigs, boresight_deg
from seascape.blender.vessels import Hulls, add_ownship
from seascape.config import Mount, Scenario
from seascape.outputs import canvas
from seascape.physics import earth

VIEWS = (
    workbench.Aim("starboard beam", 90.0, 0.0),
    workbench.Aim("top", 90.0, 89.0),
    workbench.Aim("starboard bow", 45.0, 25.0),
)
TILE = (800, 600)
# A frustum's length, as a share of the hull's diagonal.
REACH = 0.15
# A frustum's edge, as a share of its length.
WIRE = 1 / 30
# tan(hfov / 2) blows up at 180 deg and flips past it.
WIDEST_DEG = 160.0
FONT_PX = 18
LINE_PX = FONT_PX + 4

type Line = tuple[str, tuple[int, ...]]


def _frustum(
    camera: bpy.types.Object, mount: Mount, length_m: float
) -> bpy.types.Object:
    """A wireframe pyramid from `camera` along its view: nested ones stay visible."""
    spec = mount.camera
    aspect = spec.width_px / spec.height_px
    across = math.radians(min(spec.hfov_deg, WIDEST_DEG)) / 2
    # A fisheye's angle grows with the pixel radius, a pinhole's tangent does.
    up = across / aspect if spec.lens.fisheye else math.atan(math.tan(across) / aspect)
    half_w, half_h = length_m * math.tan(across), length_m * math.tan(up)
    far = [
        (sx * half_w, sy * half_h, -length_m)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    mesh = bpy.data.meshes.new(camera.name)
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), *far],
        [],
        [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 3, 2)],
    )
    obj = bpy.data.objects.new(camera.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = camera  # in the camera's frame, which looks down its -Z
    edges = obj.modifiers.new("edges", "WIREFRAME")
    edges.thickness = WIRE * length_m
    return obj


def _hit_m(origin: Vector, up: bool) -> float | None:
    """Height of the first surface straight above or below `origin`, if any."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hit, location, *_ = bpy.context.scene.ray_cast(
        depsgraph, origin, Vector((0.0, 0.0, 1.0 if up else -1.0))
    )
    return location.z if hit else None


def _rig_line(name: str, scenario: Scenario, rig: bpy.types.Object) -> Line:
    """The rig's numbers and what of the hull lies straight below and above it. Cast
    before the frusta exist, which a ray would hit.
    """
    spec = scenario.rigs[name]
    at = rig.matrix_world.translation
    text = (
        f"rig {name}: height_m {spec.height_m:g}, offset_y_m {spec.offset_y_m:g}, "
        f"offset_x_m {spec.offset_x_m:g}"
    )
    below_m = _hit_m(at, up=False)
    if below_m is None:
        text += "; no hull below it"
    else:
        text += f"; hull below it at {below_m:.2f} m, {at.z - below_m:.2f} m clear"
    if (above_m := _hit_m(at, up=True)) is not None:
        text += f"; hull ABOVE it at {above_m:.2f} m"
    return text, canvas.INK


def _tiles(objects: list[bpy.types.Object]) -> list[Image.Image]:
    """`objects` from each of `VIEWS`, in Workbench's object colours."""
    workbench.setup()
    data = bpy.data.cameras.new("rigview")
    data.type = "ORTHO"
    sc = bpy.context.scene
    sc.camera = bpy.data.objects.new("rigview", data)
    sc.collection.objects.link(sc.camera)
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for pose in VIEWS:
            workbench.aim(objects, pose.bearing_deg, pose.elevation_deg)
            path = Path(tmp) / f"{pose.name}.png"
            workbench.shoot(path, "OBJECT", "STUDIO", "FXAA", exr=False)
            tile = canvas.on_backdrop(Image.open(path))
            tile = tile.resize(TILE, Image.Resampling.LANCZOS)
            ImageDraw.Draw(tile).text(
                (10, 8), pose.name, font=canvas.font(FONT_PX + 2), fill=canvas.INK
            )
            tiles.append(tile)
    return tiles


def _caption(lines: list[Line], width_px: int) -> Image.Image:
    """`lines` top to bottom, each in its own ink, on the matte."""
    caption = Image.new("RGB", (width_px, 16 + LINE_PX * len(lines)), canvas.MATTE)
    pen = ImageDraw.Draw(caption)
    for k, (text, ink) in enumerate(lines):
        pen.text((10, 8 + LINE_PX * k), text, font=canvas.font(FONT_PX), fill=ink)
    return caption


def draw(scenario: Scenario, out: Path) -> None:
    """Draw the ownship at the scenario's start, a frustum per camera and the rigs'
    numbers, to `out`. The ownship needs an asset.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    built = add_rigs(scenario)
    vessel = add_ownship(
        scenario.ownship,
        built.root,
        Hulls(scenario.sky, None),  # Workbench draws no wear
        scenario.outputs,
        (),
        earth.earth_radius_m(scenario.sea.refraction_k),
    )
    bpy.context.scene.frame_set(0)  # runs the depsgraph: matrix_world is stale before
    hull = mesh_objects([vessel])
    lines = [_rig_line(name, scenario, rig) for name, rig in built.rigs.items()]

    corners = np.array([tuple(c) for c in box_corners(hull)])
    length_m = REACH * float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0)))
    colours = canvas.palette(len(scenario.mounts))
    frusta = []
    for mount, rgb in zip(scenario.mounts, colours, strict=True):
        camera = built.cameras[mount.name]
        frustum = _frustum(camera, mount, length_m)
        frustum.color = (*(rgb / 255), 1.0)
        frusta.append(frustum)
        bearing, elevation = boresight_deg(camera)
        lines.append(
            (
                f"{mount.name}: bearing {bearing:+.1f} deg, "
                f"elevation {elevation:+.1f} deg, hfov {mount.camera.hfov_deg:g} deg",
                tuple(int(c) for c in rgb),
            )
        )
    bpy.context.view_layer.update()

    tiles = _tiles(hull + frusta)
    out.parent.mkdir(parents=True, exist_ok=True)
    caption = _caption(lines, TILE[0] * len(tiles))
    canvas.above([canvas.beside(tiles), caption]).save(out)
