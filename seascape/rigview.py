"""Where a scenario's rigs sit on its ownship, to confirm a mount."""

import math
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw

from seascape.blender import workbench
from seascape.blender.blend import BLENDER_TO_CV, mesh_objects
from seascape.blender.meshes import box_corners
from seascape.blender.rigs import add_rigs, boresight_deg, stabilise
from seascape.blender.vessels import Hulls, add_ownship
from seascape.config import Mount, Scenario
from seascape.motion import ownship_up_m, tilted, track
from seascape.outputs import brand, canvas
from seascape.physics import earth

VIEWS = (
    workbench.Aim("starboard beam", 90.0, 0.0),
    workbench.Aim("top", 90.0, 89.0),
    workbench.Aim("astern", 150.0, 20.0),
)
TILE = (800, 600)
# A frustum's length, as a share of the hull's diagonal.
REACH = 0.15
# A frustum's edge, as a share of its length.
WIRE = 1 / 30
FONT_PX = 18
LINE_PX = FONT_PX + 4
# Left of the views' names, so the columns line up with them.
PAD_PX = 10
COLUMN_GAP_PX = 48
# Brand Book 2026: a red line structures the page, with visible weight.
RULE_PX = 3
WHITE = (255, 255, 255)

type Line = tuple[str, tuple[int, ...]]


def _frustum(
    camera: bpy.types.Object, mount: Mount, length_m: float
) -> bpy.types.Object:
    """A wireframe pyramid from `camera` along its view: nested ones stay visible."""
    spec = mount.camera
    # In image axes: x right, y down.
    corners_px = np.array([(-1, 1), (1, 1), (1, -1), (-1, -1)]) / 2
    corners_px = corners_px * (spec.width_px, spec.height_px)
    rays = spec.rays(corners_px[:, 0], corners_px[:, 1])
    far_cv = rays * length_m / rays[:, 2:]
    # Parented to the camera, so in Blender's camera axes.
    far = far_cv @ np.array(BLENDER_TO_CV.to_3x3()).T
    mesh = bpy.data.meshes.new(camera.name)
    mesh.from_pydata(
        [(0.0, 0.0, 0.0), *map(tuple, far)],
        [],
        [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 3, 2)],
    )
    obj = bpy.data.objects.new(camera.name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = camera  # in the camera's frame, which looks down its -Z
    edges = obj.modifiers.new("edges", "WIREFRAME")
    edges.thickness = WIRE * length_m
    return obj


def _hit_m(frame: bpy.types.Object, origin: Vector, up: bool) -> float | None:
    """Height in `frame` of the first surface along its vertical above or below
    `origin`, if any.
    """
    T_frame_world = frame.matrix_world.inverted()
    axis = frame.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0 if up else -1.0))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hit, location, *_ = bpy.context.scene.ray_cast(depsgraph, origin, axis)
    return (T_frame_world @ location).z if hit else None


def _rig_lines(
    name: str, scenario: Scenario, rig: bpy.types.Object, frame: bpy.types.Object
) -> list[Line]:
    """The rig's numbers and what of the hull lies straight below and above it in
    `frame`, the hull's, so a trimmed or heeled ownship measures along its own mast.
    Cast before the frusta exist, which a ray would hit.
    """
    spec = scenario.rigs[name]
    at = rig.matrix_world.translation
    height_m = (frame.matrix_world.inverted() @ at).z
    texts = [
        f"height_m {spec.height_m:g}, offset_y_m {spec.offset_y_m:g}, "
        f"offset_x_m {spec.offset_x_m:g}"
    ]
    below_m = _hit_m(frame, at, up=False)
    if below_m is None:
        texts.append("no hull below it")
    else:
        texts.append(
            f"hull below it at {below_m:.2f} m, {height_m - below_m:.2f} m clear"
        )
    if (above_m := _hit_m(frame, at, up=True)) is not None:
        texts.append(f"hull ABOVE it at {above_m:.2f} m")
    return [(text, WHITE) for text in texts]


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
            tile = canvas.on_backdrop(Image.open(path), brand.SKY_GREY)
            tile = tile.resize(TILE, Image.Resampling.LANCZOS)
            ImageDraw.Draw(tile).text(
                (10, 8),
                pose.name.upper(),
                font=brand.font(FONT_PX + 2, "Medium"),
                fill=brand.NIGHT_BLUE,
            )
            tiles.append(tile)
    return tiles


def _column(rig: str, lines: list[Line], width_px: int) -> Image.Image:
    """The rig's name over `lines`, each in its own ink."""
    column = Image.new("RGB", (width_px, LINE_PX * (1 + len(lines))), brand.NIGHT_BLUE)
    pen = ImageDraw.Draw(column)
    pen.text(
        (0, 0), f"RIG {rig.upper()}", font=brand.font(FONT_PX, "Medium"), fill=WHITE
    )
    for k, (text, ink) in enumerate(lines, start=1):
        pen.text((0, LINE_PX * k), text, font=brand.font(FONT_PX), fill=ink)
    return column


def _caption(rigs: dict[str, list[Line]], width_px: int) -> Image.Image:
    """A column per rig, a whole number of tiles wide, as many to a row as fit, on
    Night Blue under a red rule.
    """
    face = brand.font(FONT_PX)
    widest = max(face.getlength(text) for lines in rigs.values() for text, _ in lines)
    column_px = TILE[0] * math.ceil((widest + COLUMN_GAP_PX) / TILE[0])
    fit = max(1, width_px // column_px)
    columns = [_column(rig, lines, column_px) for rig, lines in rigs.items()]
    body = canvas.grid(columns, min(fit, len(columns)), matte=brand.NIGHT_BLUE)
    height_px = RULE_PX + 2 * PAD_PX + body.height
    caption = Image.new("RGB", (width_px, height_px), brand.NIGHT_BLUE)
    caption.paste(body, (PAD_PX, RULE_PX + PAD_PX))
    ImageDraw.Draw(caption).rectangle(
        (0, 0, width_px, RULE_PX - 1), fill=brand.FOCUS_RED
    )
    return caption


def draw(scenario: Scenario, out: Path) -> None:
    """Draw the ownship at the scenario's start, a frustum per camera and the rigs'
    numbers, to `out`. The ownship needs an asset.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    built = add_rigs(scenario)
    path = track(scenario.ownship, scenario.outputs, scenario.turn)
    # The hull does not ride here, so the deck tilts by its script alone.
    stabilise(built.heads, scenario, tilted(path, None, (), ()))
    vessel = add_ownship(
        scenario.ownship,
        path,
        ownship_up_m(
            scenario.ownship,
            scenario.outputs,
            (),
            earth.earth_radius_m(scenario.sea.refraction_k),
        ),
        built.root,
        Hulls(scenario.sky, None),  # Workbench draws no wear
        scenario.outputs.times_s,
        [],
    )
    bpy.context.scene.frame_set(0)  # runs the depsgraph: matrix_world is stale before
    hull = mesh_objects([vessel])
    rigs = {
        name: _rig_lines(name, scenario, rig, built.root)
        for name, rig in built.heads.items()
    }

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
        rigs[mount.rig_name].append(
            (
                f"{mount.name}: bearing {bearing:+.1f} deg, "
                f"elevation {elevation:+.1f} deg, hfov {mount.camera.hfov_deg:g} deg",
                tuple(int(c) for c in rgb),
            )
        )
    bpy.context.view_layer.update()

    tiles = _tiles(hull + frusta)
    out.parent.mkdir(parents=True, exist_ok=True)
    caption = _caption(rigs, TILE[0] * len(tiles))
    canvas.above([canvas.beside(tiles), caption]).save(out)
