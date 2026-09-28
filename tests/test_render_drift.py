"""Properties of a rendered frame that a refactor must not shift.

Skipped unless `--render` is given.
"""

import math
from pathlib import Path

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape import lwir, scene, waves
from seascape.config import Band, Scenario, load

pytestmark = pytest.mark.render

SCENARIO = load(Path(__file__).parent.parent / "scenarios" / "baseline.toml")
SAMPLES = 48


def shoot(size: tuple[int, int], name: str) -> np.ndarray:
    """Render the current scene and return its radiance, top row first."""
    sc = bpy.context.scene
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.filepath = str(Path(bpy.app.tempdir) / name)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(sc.render.filepath + ".exr")
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(size[1], size[0], 4)[::-1, :, 0]


def radiance(band: Band, kind: str, size: tuple[int, int]) -> np.ndarray:
    """Build for a band, render the named camera, return its radiance."""
    scene.build(SCENARIO, band)
    sc = bpy.context.scene
    sc.camera = next(
        o for o in bpy.data.objects if o.type == "CAMERA" and f"_{kind}_" in o.name
    )
    # Build already set the band's engine and denoiser; only the sample count is the
    # test's own.
    sc.cycles.samples = SAMPLES
    return shoot(size, f"drift_{band}")


def texture(rows: np.ndarray) -> float:
    """Spread within each row, robust to a target sitting in the band.

    Median absolute deviation, not standard deviation: a hot hull is a handful of very
    bright pixels and would otherwise swamp the sea it sits on.
    """
    return float(np.median(np.abs(rows - np.median(rows, axis=1, keepdims=True))))


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    return radiance("ir", "ir", (640, 512))


def test_the_sky_runs_from_cold_overhead_to_ambient_at_the_horizon(frame) -> None:
    """Medians, not means: a target's superstructure stands above the horizon and lands
    in the band this samples, and a hot hull is far enough off ambient to drag it.
    """
    horizon = frame.shape[0] // 2
    ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
    just_above = float(np.median(frame[horizon - 6 : horizon - 1]))
    assert just_above == pytest.approx(ambient, rel=0.02)
    assert 0.80 <= float(np.median(frame[:5])) / ambient <= 0.95


def test_sea_texture_fades_with_range(frame) -> None:
    """Displaced geometry inverts this: sub-pixel geometry aliases, not averages.

    Not strict monotonicity across all four bands: that holds for a high eye but not
    a low one, where a foreground row spans less than one wavelength.
    """
    sea = frame[frame.shape[0] // 2 + 4 :]
    band = len(sea) // 4
    far_to_near = [texture(sea[i * band : (i + 1) * band]) for i in range(4)]
    assert far_to_near[0] == min(far_to_near), (
        f"the far field has to settle, not sparkle: {far_to_near}"
    )


def sea_of(scenario: Scenario, waves: bool) -> np.ndarray:
    """The near sea of an ir frame, optionally with the wave relief flattened."""
    scene.build(scenario, "ir")
    if not waves:
        tree = bpy.data.materials["sea"].node_tree
        flat = tree.nodes.new("ShaderNodeNewGeometry").outputs["Normal"]
        tree.links.new(flat, tree.nodes["wave_normal"].inputs["Vector"])
    sc = bpy.context.scene
    sc.camera = next(
        o for o in bpy.data.objects if o.type == "CAMERA" and "_ir_" in o.name
    )
    # Flat sea is the control, so grain must sit well under the relief.
    sc.cycles.samples = 256
    frame = shoot((320, 256), "isothermal")
    horizon = frame.shape[0] // 2
    return frame[horizon + 30 : horizon + 80]


@pytest.mark.render
def test_waves_survive_a_sea_at_air_temperature() -> None:
    """A tilted facet reflects a different sky elevation, cold overhead to ambient at
    the horizon, so relief shows without `t_sea_k - t_air_k`.
    """
    isothermal = SCENARIO.model_copy(
        update={
            "sea": SCENARIO.sea.model_copy(update={"t_sea_k": SCENARIO.sky.t_air_k})
        }
    )

    rippled, flat = sea_of(isothermal, True), sea_of(isothermal, False)

    assert texture(rippled) > 2.5 * texture(flat)


LOOP = load(
    Path(__file__).parent.parent / "scenarios" / "baseline.toml",
    ["outputs.duration_s = 30", "outputs.fps = 1", "outputs.loop = true"],
)


@pytest.mark.render
@pytest.mark.parametrize(
    ("scenario", "frame"),
    # Past half-way, a loop's atan2 clock reads t - span.
    [(SCENARIO, 0), (LOOP, 20)],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_the_shader_tilts_the_sea_by_the_field_s_slope(
    scenario: Scenario, frame: int, axis: int
) -> None:
    span_m, px = 8.0, 64
    scene.build(scenario, "eo")
    bpy.context.scene.frame_set(frame)
    tree = bpy.data.materials["sea"].node_tree
    component = tree.nodes.new("ShaderNodeSeparateXYZ")
    shown = tree.nodes.new("ShaderNodeMath")
    shown.operation = "MULTIPLY_ADD"  # (n + 1) / 2, so the emission stays positive
    shown.inputs["Value_001"].default_value = 0.5
    shown.inputs["Value_002"].default_value = 0.5
    emission = tree.nodes.new("ShaderNodeEmission")
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    link = tree.links.new
    link(tree.nodes["wave_normal"].outputs["Vector"], component.inputs["Vector"])
    link(component.outputs["XY"[axis]], shown.inputs["Value"])
    link(shown.outputs["Value"], emission.inputs["Color"])
    link(emission.outputs["Emission"], output.inputs["Surface"])

    lens = bpy.data.cameras.new("probe")
    lens.type, lens.ortho_scale = "ORTHO", span_m
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01  # the pixel centre, as the numpy grid
    sc.view_settings.view_transform = "Standard"
    rendered = 2 * shoot((px, px), "slope_probe") - 1

    centres = (np.arange(px) + 0.5) * span_m / px - span_m / 2
    east, north = np.meshgrid(centres, centres[::-1])
    t_s = scenario.outputs.times_s[frame]
    slope = waves.slope(scene.wave_field(scenario), east, north, t_s)
    expected = -slope[axis] / np.sqrt(1 + slope[0] ** 2 + slope[1] ** 2)
    assert np.abs(rendered - expected).max() < 2e-3


@pytest.mark.render
def test_a_pixel_takes_as_roughness_the_slope_it_does_not_draw() -> None:
    """Top down from high enough that the footprint sits among the waves' fades."""
    height_m, span_m, px = 1500.0, 600.0, 32
    scene.build(SCENARIO, "eo")
    tree = bpy.data.materials["sea"].node_tree
    emission = tree.nodes.new("ShaderNodeEmission")
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    tree.links.new(
        tree.nodes["sea_roughness"].outputs["Value"], emission.inputs["Color"]
    )
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    lens = bpy.data.cameras.new("probe")
    lens.type, lens.ortho_scale = "ORTHO", span_m
    lens.clip_end = 2 * height_m  # the default far plane stops short of the sea
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    camera.location = (0.0, 0.0, height_m)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    sc.view_settings.view_transform = "Standard"
    rendered = shoot((px, px), "roughness_probe")

    mount = next(m for m in SCENARIO.rig.mounts if m.camera.kind == "eo")
    pixel_rad = math.radians(mount.camera.hfov_deg) / mount.camera.width_px
    centres = (np.arange(px) + 0.5) * span_m / px - span_m / 2
    east, north = np.meshgrid(centres, centres[::-1])
    distance = np.sqrt(east**2 + north**2 + height_m**2)
    field, speed = scene.wave_field(SCENARIO), SCENARIO.sea.wind_speed_mps
    variance = np.array(
        [
            waves.unresolved_slope_variance(speed, field, d * pixel_rad)
            for d in distance.ravel()
        ]
    ).reshape(distance.shape)
    # Top down, both footprints agree: alpha = sqrt(variance) on each axis.
    expected = np.minimum(variance**0.25, 1.0)
    assert np.abs(rendered - expected).max() < 5e-3


@pytest.mark.render
def test_a_grazing_pixel_stretches_its_lobe_along_the_view() -> None:
    """Principled's Anisotropic, read back at a grazing view, against the footprints
    the pixel covers: 10 m up, looking 3 deg down, the sea from ~100 m to ~1 km."""
    scene.build(SCENARIO, "eo")
    tree = bpy.data.materials["sea"].node_tree
    anisotropic = (
        tree.nodes["Principled BSDF"].inputs["Anisotropic"].links[0].from_socket
    )
    emission = tree.nodes.new("ShaderNodeEmission")
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    tree.links.new(anisotropic, emission.inputs["Strength"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    lens = bpy.data.cameras.new("probe")
    lens.clip_end = 1e5
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_euler = (math.radians(87.0), 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    sc.view_settings.view_transform = "Standard"
    px = (32, 24)
    rendered = shoot(px, "anisotropy_probe")

    bpy.context.view_layer.update()
    corners = [camera.matrix_world @ c for c in lens.view_frame(scene=sc)]
    top_right, bottom_right, bottom_left, top_left = (np.array(c) for c in corners)
    origin = np.array(camera.matrix_world.translation)
    mount = next(m for m in SCENARIO.rig.mounts if m.camera.kind == "eo")
    pixel_rad = math.radians(mount.camera.hfov_deg) / mount.camera.width_px
    field, speed = scene.wave_field(SCENARIO), SCENARIO.sea.wind_speed_mps
    checked = 0
    for row in range(px[1]):
        for col in range(px[0]):
            u, v = (col + 0.5) / px[0], (row + 0.5) / px[1]
            left = top_left + v * (bottom_left - top_left)
            right = top_right + v * (bottom_right - top_right)
            ray = left + u * (right - left) - origin
            ray /= np.linalg.norm(ray)
            if ray[2] > -0.02:  # the horizon's rows and the sky
                continue
            distance = origin[2] / -ray[2]
            across = distance * pixel_rad
            along = across / -ray[2]
            v_along = waves.unresolved_slope_variance(speed, field, along)
            v_across = waves.unresolved_slope_variance(speed, field, across)
            expected = (1 - math.sqrt(v_across / v_along)) / 0.9
            assert rendered[row, col] == pytest.approx(expected, abs=0.02), (row, col)
            checked += 1
    assert checked > 100
    assert rendered.max() > 0.2, "a grazing view is anisotropic"


@pytest.mark.render
def test_the_rendered_whitecaps_cover_what_monahan_measured() -> None:
    # Nothing but sea in view: a lit hull would count as foam.
    windy = SCENARIO.model_copy(
        update={
            "sea": SCENARIO.sea.model_copy(update={"wind_speed_mps": 15.0}),
            "objects": [],
            "ownship": SCENARIO.ownship.model_copy(update={"asset": None}),
        }
    )
    scene.build(windy, "eo")
    tree = bpy.data.materials["sea"].node_tree
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes["Material Output"]
    tree.links.new(tree.nodes["whitecaps"].outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    lens = bpy.data.cameras.new("probe")
    lens.type, lens.ortho_scale = "ORTHO", 1500.0
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    sc.view_settings.view_transform = "Standard"
    covered = float(np.mean(shoot((400, 400), "whitecap_probe")))
    assert covered == pytest.approx(waves.whitecap_fraction(15.0), rel=0.2)


def test_the_sky_draws_its_sun_where_the_sun_vector_points() -> None:
    """The hull's heating takes the sun from that vector and the sky draws it from the
    bearing; a sign between them mirrors the disc and its glitter east to west."""
    scene.build(SCENARIO, "eo")
    sc = bpy.context.scene
    camera = sc.camera
    camera.parent = None
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = Vector(scene._sun_vector(SCENARIO.sky)).to_track_quat(
        "-Z", "Y"
    )
    sc.cycles.samples = 4
    sc.cycles.use_denoising = False
    frame = shoot((200, 150), "sun")
    row, col = np.unravel_index(frame.argmax(), frame.shape)
    assert abs(row - 75) <= 2 and abs(col - 100) <= 2
