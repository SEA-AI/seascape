"""Build a Blender scene from a scenario. Nothing here renders."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

from seascape import lwir, sea, waves
from seascape.assets import Asset, fetch, manifest
from seascape.blend import CURVE_SAMPLES, animate, curve_image, place, sine, yaw
from seascape.calibration import CameraCalibration, Matrix4
from seascape.config import (
    Band,
    Drift,
    ImageFormat,
    Mount,
    Object,
    Outputs,
    Ownship,
    Rig,
    Scenario,
    Sky,
    Targets,
)

# Flat paint over steel, 8-14 um. Paints sit at 0.94-0.96 across this band and the
# colour does not matter, only how flat the finish is.
PAINT_EMISSIVITY = 0.94


def _substream(seed: int, name: str) -> np.random.Generator:
    """A named substream, so adding a component cannot perturb an existing one."""
    return np.random.default_rng([seed, *name.encode()])


def wave_field(scenario: Scenario) -> tuple[waves.Wave, ...]:
    return waves.components(
        scenario.sea.wind_speed_mps,
        scenario.sea.wind_from_deg,
        scenario.outputs.period_s,
        _substream(scenario.seed, "sea/surface"),
    )


def _sky(sky: Sky, band: Band) -> bpy.types.World:
    world = bpy.data.worlds.new("sky")
    # Nonzero, EEVEE turns world light above it into a sun a mirror cannot see.
    world.sun_threshold = 0.0
    tree = world.node_tree
    if band == "ir":
        return _thermal_sky(world, sky.t_air_k)
    node = tree.nodes.new("ShaderNodeTexSky")
    # `turbidity` belongs to Preetham and Hosek-Wilkie and is silently ignored here.
    node.sky_type = "MULTIPLE_SCATTERING"
    node.sun_elevation = math.radians(sky.sun_elevation_deg)
    # An azimuth, clockwise from +Y, though Blender calls it a rotation.
    node.sun_rotation = math.radians(sky.sun_bearing_deg)
    node.aerosol_density = sky.aerosol_density
    tree.links.new(node.outputs["Color"], tree.nodes["Background"].inputs["Color"])
    return world


def _sky_image(t_air_k: float) -> bpy.types.Image:
    """`lwir.sky_radiance` baked against sin(elevation), which is what the shader has.

    A world shader's ray direction is a unit vector, so its Z is already sin(elevation)
    and no arcsine node is needed. Below the horizon Z is clamped to 0, where the curve
    holds at ambient.
    """
    return curve_image(
        "sky_radiance",
        lwir.sky_radiance(np.arcsin(np.linspace(0.0, 1.0, CURVE_SAMPLES)), t_air_k),
    )


def _thermal_sky(world: bpy.types.World, t_air_k: float) -> bpy.types.World:
    """Downwelling radiance against elevation, as raw W m^-2 sr^-1."""
    tree = world.node_tree
    tree.nodes.clear()
    link = tree.links.new
    coord = tree.nodes.new("ShaderNodeTexCoord")
    height = tree.nodes.new("ShaderNodeSeparateXYZ")
    above = tree.nodes.new("ShaderNodeMath")
    above.operation = "MAXIMUM"
    above.use_clamp = True
    above.inputs[1].default_value = 0.0
    lookup = tree.nodes.new("ShaderNodeCombineXYZ")
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = _sky_image(t_air_k)
    texture.extension = "EXTEND"
    background = tree.nodes.new("ShaderNodeBackground")
    output = tree.nodes.new("ShaderNodeOutputWorld")

    link(coord.outputs["Generated"], height.inputs["Vector"])
    link(height.outputs["Z"], above.inputs[0])
    link(above.outputs["Value"], lookup.inputs["X"])
    link(lookup.outputs["Vector"], texture.inputs["Vector"])
    link(texture.outputs["Color"], background.inputs["Color"])
    link(background.outputs["Background"], output.inputs["Surface"])
    return world


def _sun_vector(sky: Sky) -> tuple[float, float, float]:
    """Unit vector towards the sun. A direction, so the bearing is not negated: that
    belongs to rotations, and `_pose` places by the same sin/cos."""
    elevation = math.radians(sky.sun_elevation_deg)
    bearing = math.radians(sky.sun_bearing_deg)
    return (
        math.cos(elevation) * math.sin(bearing),
        math.cos(elevation) * math.cos(bearing),
        math.sin(elevation),
    )


def _thermal_skin(name: str, t_k: float, sky: Sky) -> bpy.types.Material:
    """eps of a painted hull emitted, the remaining 1 - eps reflected from the sky.

    Diffuse: flat marine paint is near-Lambertian in this band.
    """
    material = bpy.data.materials.new(name)
    tree = material.node_tree
    tree.nodes.clear()
    # White: the Mix Shader already applies the 1 - eps weighting, so a grey here would
    # absorb part of the reflected sky a second time.
    scatter = tree.nodes.new("ShaderNodeBsdfDiffuse")
    scatter.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    mix = tree.nodes.new("ShaderNodeMixShader")
    # Mix Shader names both shader inputs "Shader", so they can only be indexed.
    mix.inputs["Factor"].default_value = PAINT_EMISSIVITY
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    link = tree.links.new
    link(scatter.outputs["BSDF"], mix.inputs[1])
    link(_sunlit_emission(tree, t_k, sky), mix.inputs[2])
    link(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _sunlit_emission(
    tree: bpy.types.NodeTree, t_k: float, sky: Sky
) -> bpy.types.NodeSocket:
    """Emission graded from shaded to sunlit by Lambert's cosine on the real sun.

    Two emissions mixed by `max(0, n . sun)`, so both ends are the exact band radiance
    and only the middle interpolates: no shader node can evaluate the Planck integral
    at the blended temperature.
    """
    shaded = tree.nodes.new("ShaderNodeEmission")
    shaded.inputs["Strength"].default_value = lwir.band_radiance(t_k)
    sunlit = tree.nodes.new("ShaderNodeEmission")
    sunlit.inputs["Strength"].default_value = lwir.band_radiance(t_k + sky.solar_gain_k)

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = tree.nodes.new("ShaderNodeVectorMath")
    facing.operation = "DOT_PRODUCT"
    facing.inputs[1].default_value = _sun_vector(sky)
    # Clamped, or a surface turned away from the sun would cool below shaded.
    lit = tree.nodes.new("ShaderNodeMath")
    lit.operation = "MAXIMUM"
    lit.inputs[1].default_value = 0.0

    grade = tree.nodes.new("ShaderNodeMixShader")
    link = tree.links.new
    link(geometry.outputs["Normal"], facing.inputs[0])
    link(facing.outputs["Value"], lit.inputs[0])
    link(lit.outputs["Value"], grade.inputs["Factor"])
    link(shaded.outputs["Emission"], grade.inputs[1])
    link(sunlit.outputs["Emission"], grade.inputs[2])
    return grade.outputs["Shader"]


class _RigObjects(NamedTuple):
    root: bpy.types.Object
    pods: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]


def _rig(rig: Rig, far_m: float) -> _RigObjects:
    # Blender takes an inverted frustum without complaint and renders nothing.
    if rig.near_clip_m >= far_m:
        raise ValueError(
            f"near clip {rig.near_clip_m} m is past the far plane at {far_m:.0f} m"
        )
    root = bpy.data.objects.new("rig", None)
    bpy.context.collection.objects.link(root)
    place(root, 0.0, 0.0, rig.height_m)

    pods: dict[str, bpy.types.Object] = {}
    for pod in rig.pods:
        empty = bpy.data.objects.new(f"pod_{pod.name}", None)
        bpy.context.collection.objects.link(empty)
        empty.parent = root
        place(empty, pod.offset_x_m, pod.offset_y_m, 0.0)
        # XYZ euler is Rz @ Ry @ Rx: yaw, then pitch about the pod's own transverse
        # axis, so a pitched pod rolls the horizon of its off-axis cameras.
        empty.rotation_euler = (
            math.radians(rig.pitch_deg),
            0.0,
            yaw(pod.yaw_deg),
        )
        pods[pod.name] = empty

    cameras: dict[str, bpy.types.Object] = {}
    for mount in rig.mounts:
        data = bpy.data.cameras.new(mount.name)
        # AUTO fits the field of view to whichever image dimension is larger, so a
        # portrait sensor would silently reinterpret hfov as a vertical angle.
        data.sensor_fit = "HORIZONTAL"
        data.angle_x = math.radians(mount.camera.hfov_deg)
        data.clip_start = rig.near_clip_m
        # The default 1000 m renders a target past it as sky, and the clip boundary
        # reads as the horizon. Nothing warns.
        data.clip_end = far_m
        camera = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(camera)
        camera.parent = pods[mount.pod.name]
        camera.rotation_mode = "XYZ"
        # A camera looks down its local -Z; +90 deg about X aims it at the horizon.
        camera.rotation_euler = (
            math.radians(90.0 + mount.camera.pitch_deg),
            0.0,
            yaw(mount.camera.yaw_deg),
        )
        cameras[mount.name] = camera
    return _RigObjects(root, pods, cameras)


def boresight_deg(
    camera: bpy.types.Object, frame: bpy.types.Object | None = None
) -> tuple[float, float]:
    """Bearing and elevation a built camera points at, in degrees, in the world or in
    `frame`'s axes. `matrix_world` is stale until the depsgraph runs, so build first.
    """
    rotation = camera.matrix_world.to_3x3()
    if frame is not None:
        rotation = frame.matrix_world.to_3x3().inverted() @ rotation
    forward = rotation @ Vector((0.0, 0.0, -1.0))
    forward.normalize()
    return (
        math.degrees(math.atan2(forward.x, forward.y)),
        math.degrees(math.asin(min(1.0, max(-1.0, forward.z)))),
    )


# Blender's camera looks down -Z with +Y up; OpenCV's looks down +Z with +Y down.
_BLENDER_TO_CV = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))


@dataclass(frozen=True)
class Built:
    """What `build` made, by role, valid until the next build. A name in `bpy.data`
    belongs to whoever took it first, and an imported asset can carry any name."""

    vessel: bpy.types.Object
    pods: dict[str, bpy.types.Object]
    cameras: dict[str, bpy.types.Object]
    targets: dict[str, list[bpy.types.Object]]  # by asset


def calibrate(built: Built, mount: Mount, image: str) -> CameraCalibration:
    """A built camera's geometry, read off the scene."""
    camera = built.cameras[mount.name]
    world = camera.matrix_world @ _BLENDER_TO_CV
    vessel = built.vessel.matrix_world
    pod = built.pods[mount.pod.name].matrix_world
    width, height = mount.camera.width_px, mount.camera.height_px
    f = (width / 2) / math.tan(camera.data.angle_x / 2)
    return CameraCalibration(
        name=mount.name,
        band=mount.camera.kind,
        image=image,
        pod=mount.pod.name,
        width_px=width,
        height_px=height,
        # Blender's frame spans pixel edges, so its centre is half a pixel past
        # OpenCV's.
        K=((f, 0.0, (width - 1) / 2), (0.0, f, (height - 1) / 2), (0.0, 0.0, 1.0)),
        extrinsics={
            "world": _rows(world),
            "vessel": _rows(vessel.inverted() @ world),
            "pod": _rows(pod.inverted() @ world),
        },
    )


def waterline_m(anchor: bpy.types.Object) -> np.ndarray:
    """Where a hull's edges cross its waterline, as world east and north, (N, 2).

    The waterline is the anchor's level: `_fit` puts the keel a draught below it and
    `_pose` never tilts it.
    """
    # ponytail: the crossings, not the segments between them, so the nearest one to a
    # camera at range r is up to side^2 / 2r further than a side facing it.
    level = anchor.matrix_world.translation.z
    crossings = []
    for part in _meshes([anchor]):
        mesh = part.data
        local = np.empty(3 * len(mesh.vertices), dtype=np.float32)
        mesh.vertices.foreach_get("co", local)
        ends = np.empty(2 * len(mesh.edges), dtype=np.int32)
        mesh.edges.foreach_get("vertices", ends)
        m = np.array(part.matrix_world)
        world = local.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        a, b = world[ends.reshape(-1, 2)].transpose(1, 0, 2)
        cross = (a[:, 2] - level) * (b[:, 2] - level) < 0
        a, b = a[cross], b[cross]
        t = (level - a[:, 2]) / (b[:, 2] - a[:, 2])
        crossings.append((a + t[:, None] * (b - a))[:, :2])
    return np.concatenate(crossings)


def _rows(m: Matrix) -> Matrix4:
    x, y, z, w = map(tuple, m)
    return x, y, z, w


def _corners(objects: Iterable[bpy.types.Object]) -> list[Vector]:
    """World-space bounding corners of the meshes in `objects`.

    An empty's `bound_box` is a point at its origin, and an import is mostly empties,
    so including them silently inflates the extent.
    """
    return [
        o.matrix_world @ Vector(corner)
        for o in objects
        if o.type == "MESH"
        for corner in o.bound_box
    ]


def _fit(corners: Iterable[Vector], asset: Asset) -> Matrix:
    """Bow to +Y, scaled to the manifest length, centred, keel at the draught.

    Turned first so the length is measured bow to stern whatever the authored axis.
    """
    turn = Matrix.Rotation(yaw(-asset.bow_deg), 4, "Z")
    axes = list(zip(*(turn @ c for c in corners), strict=True))
    low, high = Vector([min(a) for a in axes]), Vector([max(a) for a in axes])
    scale = asset.length_m / (high.y - low.y)
    low, high = low * scale, high * scale
    return (
        Matrix.Translation(
            (-(low.x + high.x) / 2, -(low.y + high.y) / 2, -low.z - asset.draught_m)
        )
        @ Matrix.Scale(scale, 4)
        @ turn
    )


def _meshes(parts: Iterable[bpy.types.Object]) -> list[bpy.types.Object]:
    return [
        obj
        for part in parts
        for obj in [part, *part.children_recursive]
        if obj.type == "MESH"
    ]


def _import(name: str, band: Band) -> list[bpy.types.Object]:
    """Import an asset and fit it; returns the root parts.

    An asset arrives in its author's units, off-origin, in many parts.
    """
    before = set(bpy.data.objects)
    path = fetch(name)
    importer = {".fbx": bpy.ops.import_scene.fbx, ".glb": bpy.ops.import_scene.gltf}
    importer[path.suffix](filepath=str(path))
    imported = set(bpy.data.objects) - before
    # A Poly glTF brings its viewer's camera and lights, which would light the scene.
    for obj in [o for o in imported if o.type in {"CAMERA", "LIGHT"}]:
        imported.remove(obj)
        bpy.data.objects.remove(obj)
    # Measure everything, move the roots. An FBX keeps meshes under empties, and
    # measuring only the roots would leave them out of the fit.
    parts = [o for o in imported if o.parent is None]

    fit = _fit(_corners(imported), manifest()[name])
    for part in parts:
        part.matrix_world = fit @ part.matrix_world

    if band == "ir":
        # The asset's own materials are albedo, which says nothing about 8-14 um.
        # One empty slot, which each hull fills with its own skin.
        for mesh in _meshes(parts):
            mesh.data.materials.clear()
            mesh.data.materials.append(None)
    return parts


def _vessel(
    name: str,
    t_k: float,
    band: Band,
    sky: Sky,
    hulls: dict[str, list[bpy.types.Object]],
) -> bpy.types.Object:
    """A hull fitted and anchored at the origin under an empty.

    `hulls` holds each asset's first import for the rest of the build: the FBX
    importer slows with every material already in the file.
    """
    if name in hulls:
        parts = [_copy_tree(part, None) for part in hulls[name]]
    else:
        parts = hulls[name] = _import(name, band)

    if band == "ir":
        skin = _thermal_skin(f"{name}_ir", t_k, sky)
        for mesh in _meshes(parts):
            # Copies share mesh data, and hulls differ in temperature.
            slot = mesh.material_slots[0]
            slot.link = "OBJECT"
            slot.material = skin

    anchor = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(anchor)
    for part in parts:
        part.parent = anchor
    place(anchor, 0.0, 0.0, 0.0)
    return anchor


def _ownship(
    ownship: Ownship,
    band: Band,
    sky: Sky,
    rig: bpy.types.Object,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
) -> bpy.types.Object:
    """At the origin, bow to +Y, carrying the rig: its offsets are in this frame."""
    if ownship.asset is None:
        anchor = bpy.data.objects.new("ownship", None)
        bpy.context.collection.objects.link(anchor)
        place(anchor, 0.0, 0.0, 0.0)
    else:
        anchor = _vessel(ownship.asset, ownship.t_k, band, sky, hulls)
    anchor.name = "ownship"
    rig.parent = anchor
    # YXZ euler is Rz @ Rx @ Ry: roll about the keel, innermost.
    anchor.rotation_mode = "YXZ"
    anchor.rotation_euler = (
        math.radians(ownship.pitch_deg),
        math.radians(ownship.roll_deg),
        0.0,
    )
    for index, mean_deg, swing in (
        (0, ownship.pitch_deg, ownship.pitch),
        (1, ownship.roll_deg, ownship.roll),
    ):
        if swing is not None:
            value_at = sine(
                math.radians(mean_deg),
                math.radians(swing.amplitude_deg),
                outputs.period_s(swing.period_s),
            )
            animate(anchor, "rotation_euler", outputs.times_s, value_at, index)
    if (heave := ownship.heave) is not None:
        value_at = sine(
            anchor.location.z, heave.amplitude_m, outputs.period_s(heave.period_s)
        )
        animate(anchor, "location", outputs.times_s, value_at, index=2)
    return anchor


def _pose(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    heading_deg: float,
    speed_mps: float,
    drift: Drift | None,
    radius_m: float,
    outputs: Outputs,
) -> None:
    """Put a hull on the sea at a bearing and range at t = 0, underway along its
    heading and drifting about that pose.

    A hull left at z = 0 flies above the curved sea.

    Not tilted to the local vertical: at any range the sea reaches, range / R moves a
    hull's ends far less than its draught.
    """
    bearing, heading = math.radians(bearing_deg), math.radians(heading_deg)
    sway = surge = sine(0.0, 0.0, 1.0)
    if drift is not None:
        period_s = outputs.period_s(drift.period_s)
        sway = sine(0.0, drift.sway_m, period_s)
        surge = sine(0.0, drift.surge_m, period_s / 2.0)

    def at(t_s: float) -> tuple[float, float, float]:
        along, across = speed_mps * t_s + surge(t_s), sway(t_s)
        # Starboard of the heading is (cos, -sin).
        east = (
            range_m * math.sin(bearing)
            + along * math.sin(heading)
            + across * math.cos(heading)
        )
        north = (
            range_m * math.cos(bearing)
            + along * math.cos(heading)
            - across * math.sin(heading)
        )
        return east, north, waves.sea_z_m(east, north, radius_m)

    place(anchor, *at(0.0))
    animate(anchor, "location", outputs.times_s, at)
    anchor.rotation_euler = (0.0, 0.0, yaw(heading_deg))


def _orbit(
    anchor: bpy.types.Object,
    range_m: float,
    bearing_deg: float,
    lap_s: float,
    radius_m: float,
    times_s: Sequence[float],
) -> None:
    def bearing_at_deg(t_s: float) -> float:
        return bearing_deg + 360.0 * t_s / lap_s

    def at(t_s: float) -> tuple[float, float, float]:
        bearing = math.radians(bearing_at_deg(t_s))
        east, north = range_m * math.sin(bearing), range_m * math.cos(bearing)
        return east, north, waves.sea_z_m(east, north, radius_m)

    place(anchor, *at(0.0))
    animate(anchor, "location", times_s, at)
    # Clockwise, the bow runs a right angle ahead of the bearing.
    animate(
        anchor,
        "rotation_euler",
        times_s,
        lambda t: yaw(bearing_at_deg(t) + 90),
        2,
    )


def _copy_tree(
    obj: bpy.types.Object, parent: bpy.types.Object | None
) -> bpy.types.Object:
    """Duplicate an object tree. `copy()` shares `data`, so N targets cost one mesh.

    The parent inverse copies too, so the tree keeps its shape rather than flattening.
    """
    clone = obj.copy()
    bpy.context.collection.objects.link(clone)
    clone.parent = parent
    for child in obj.children:
        _copy_tree(child, clone)
    return clone


def _targets(
    spec: Targets,
    band: Band,
    radius_m: float,
    sky: Sky,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
) -> list[bpy.types.Object]:
    first = _vessel(spec.asset, spec.t_k, band, sky, hulls)
    poses = spec.poses()
    anchors = [first, *(_copy_tree(first, None) for _ in poses[1:])]
    for i, (anchor, (bearing_deg, heading_deg)) in enumerate(
        zip(anchors, poses, strict=True)
    ):
        anchor.name = f"target_{i}"
        _pose(
            anchor,
            spec.range_m,
            bearing_deg,
            heading_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
    return anchors


def _object(
    spec: Object,
    band: Band,
    radius_m: float,
    sky: Sky,
    hulls: dict[str, list[bpy.types.Object]],
    outputs: Outputs,
) -> list[bpy.types.Object]:
    anchor = _vessel(spec.asset, spec.t_k, band, sky, hulls)
    orbit = spec.orbit
    if orbit is None:
        _pose(
            anchor,
            spec.range_m,
            spec.bearing_deg,
            spec.heading_deg,
            spec.speed_mps,
            spec.drift,
            radius_m,
            outputs,
        )
        return [anchor]
    anchors = [anchor, *(_copy_tree(anchor, None) for _ in range(orbit.count - 1))]
    lap_s = orbit.count * outputs.period_s(orbit.period_s / orbit.count)
    for i, hull in enumerate(anchors):
        bearing_deg = spec.bearing_deg + 360.0 * i / orbit.count
        _orbit(hull, spec.range_m, bearing_deg, lap_s, radius_m, outputs.times_s)
    return anchors


JPEG_QUALITY = 95

# Blender's identifier and bit depth.
FORMATS: dict[ImageFormat, tuple[str, str]] = {
    "exr": ("OPEN_EXR", "32"),
    "png": ("PNG", "8"),
    "jpg": ("JPEG", "8"),
}


def _enable_gpu() -> bool:
    """Point Cycles at a GPU. Without `refresh_devices()` it stays on the CPU."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            preferences.compute_device_type = backend
        except TypeError:
            continue  # not compiled into this build
        preferences.refresh_devices()
        if any(device.type != "CPU" for device in preferences.devices):
            for device in preferences.devices:
                # CPU alongside the GPU wins nothing here.
                device.use = device.type != "CPU"
            return True
    return False


def _output(outputs: Outputs, band: Band) -> None:
    """Render and display settings, in the .blend, so F12 renders what `render` does."""
    sc = bpy.context.scene
    # Not EEVEE: it darkens rough reflections at grazing view.
    sc.render.engine = "CYCLES"
    # Caps reflected light at 10, and LWIR radiance is tens of W m^-2 sr^-1.
    sc.eevee.clamp_surface_indirect = 0.0
    sc.cycles.device = "GPU" if _enable_gpu() else "CPU"
    sc.cycles.samples = getattr(outputs.samples, band)
    # On by default. OIDN breaks the ir frame's R=G=B and blurs sub-pixel waves.
    sc.cycles.use_denoising = False
    view = sc.view_settings
    if band == "eo":
        view.exposure = outputs.exposure_ev
    else:
        # Radiance in W m^-2 sr^-1, not a picture; the default AgX film curve bends it.
        view.view_transform, view.look = "Standard", "None"
        view.exposure, view.gamma = 0.0, 1.0
    # 8-bit radiance is not radiance; `render` maps 8-bit ir from the exr.
    file_format, depth = FORMATS[outputs.format if band == "eo" else "exr"]
    sc.render.image_settings.file_format = file_format
    sc.render.image_settings.color_depth = depth
    sc.render.image_settings.quality = JPEG_QUALITY
    sc.render.use_persistent_data = True
    # A fixed seed would hold the sample noise still while the scene moves under it.
    sc.cycles.use_animated_seed = True


def _viewport(near_m: float, far_m: float) -> None:
    """Blender's view clip defaults to 0.01-1000 m, which cuts a sea reaching tens
    of km."""
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.clip_start, space.clip_end = near_m, far_m


def build(scenario: Scenario, band: Band = "eo") -> Built:
    """Replace the current Blender session's contents with `scenario` in one band."""
    if not any(mount.camera.kind == band for mount in scenario.rig.mounts):
        raise ValueError(f"the rig has no {band} camera to build a {band} scene for")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _output(scenario.outputs, band)
    bpy.context.scene.world = _sky(scenario.sky, band)
    reach_m = sea.sea_reach_m(scenario.rig, scenario.sea)
    far_m = 1.5 * reach_m  # the sea's corner is reach * sqrt(2) away
    outputs = scenario.outputs
    mounts = [m for m in scenario.rig.mounts if m.camera.kind == band]
    # The finest pixel in the band, so its sharpest camera does not blur.
    pixel_rad = min(math.radians(m.camera.hfov_deg) / m.camera.width_px for m in mounts)
    sea.water(scenario.sea, wave_field(scenario), reach_m, band, outputs, pixel_rad)
    rig = _rig(scenario.rig, far_m)
    hulls: dict[str, list[bpy.types.Object]] = {}
    vessel = _ownship(scenario.ownship, band, scenario.sky, rig.root, hulls, outputs)
    radius_m = waves.earth_radius_m(scenario.sea.refraction_k)
    targets: dict[str, list[bpy.types.Object]] = {}
    for spec in scenario.objects:
        anchors = _object(spec, band, radius_m, scenario.sky, hulls, outputs)
        targets.setdefault(spec.asset, []).extend(anchors)
    if scenario.targets is not None:
        anchors = _targets(
            scenario.targets, band, radius_m, scenario.sky, hulls, outputs
        )
        targets.setdefault(scenario.targets.asset, []).extend(anchors)
    # The object-index pass reads 0 for everything else: sky, sea and ownship.
    for index, anchor in enumerate(chain(*targets.values()), start=1):
        for part in [anchor, *anchor.children_recursive]:
            part.pass_index = index
    # The band's first camera, not the rig's: an IR build would otherwise open on a
    # camera whose optics belong to the other band.
    first = next(mount for mount in scenario.rig.mounts if mount.camera.kind == band)
    sc = bpy.context.scene
    sc.camera = rig.cameras[first.name]
    sc.render.resolution_x, sc.render.resolution_y = (
        first.camera.width_px,
        first.camera.height_px,
    )
    _viewport(scenario.rig.near_clip_m, far_m)
    # After the last import, which sets fps and fps_base to the file's own. The
    # factory scene starts at frame 1.
    sc.frame_start = sc.frame_current = 0
    sc.frame_end = len(outputs.times_s) - 1
    sc.render.fps, sc.render.fps_base = outputs.fps, 1.0
    # Until the depsgraph runs, every child still reports its pre-parenting
    # matrix_world, so anything measuring the scene reads the wrong place.
    bpy.context.view_layer.update()
    return Built(vessel, rig.pods, rig.cameras, targets)
