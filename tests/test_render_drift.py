"""Properties of a rendered frame that a refactor must not shift."""

import math
import shutil
from collections import defaultdict
from pathlib import Path

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape import render, skies
from seascape.assets import Buoy, download, fetch, manifest
from seascape.blender import scene, sea
from seascape.blender.blend import _math
from seascape.blender.compositor import (
    GLARE_SHARE,
    GLARE_SHOULDER,
    GLARE_SLOPE,
    MID_GREY,
    respond,
)
from seascape.blender.meshes import _import, _meshes, measure
from seascape.blender.sea.daylight import SEAWATER_IOR, SUN_SLOPE_RADIUS
from seascape.blender.sea.foam import LATTICE_WALL_M, _frames_folder
from seascape.blender.sea.geometry import DICED_EDGE_M, OFFSCREEN_DICING, SEA_CELLS
from seascape.blender.sky import _sky_texture, _sun_vector, haze
from seascape.config import Band, Camera, Mount, Scenario, Sky, load, substream
from seascape.physics import colour, lwir, wakes, waves
from tests.scenarios import BASELINE, OPEN_SEA, preset_target, target, variant

pytestmark = pytest.mark.render


def _clear(scenario: Scenario) -> Scenario:
    """The probes read the sea's shader: haze would add its airlight to every one, and
    an 8-bit format the camera's glare, blur and exposure."""
    sky = scenario.sky.model_copy(update={"visibility_km": None})
    outputs = scenario.outputs.model_copy(update={"format": "exr"})
    return scenario.model_copy(update={"sky": sky, "outputs": outputs})


SCENARIO = _clear(load(OPEN_SEA))
SAMPLES = 48


def shoot(size: tuple[int, int], name: str) -> np.ndarray:
    """Render the current scene and return its radiance, top row first."""
    return shoot_rgb(size, name)[:, :, 0]


def shoot_rgb(size: tuple[int, int], name: str) -> np.ndarray:
    """Render the current scene and return its RGB radiance, top row first."""
    sc = bpy.context.scene
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.resolution_x, sc.render.resolution_y = size
    sc.render.filepath = str(Path(bpy.app.tempdir) / name)
    bpy.ops.render.render(write_still=True)

    image = bpy.data.images.load(sc.render.filepath + ".exr")
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(size[1], size[0], 4)[::-1, :, :3]


def eo_pixel_rad(scenario: Scenario) -> float:
    mount = next(m for m in scenario.mounts if m.camera.band == "eo")
    return math.radians(mount.camera.hfov_deg) / mount.camera.width_px


def overhead(span_m: float, pixel_rad: float, px: int) -> bpy.types.Camera:
    """An orthographic lens `span_m` across and `px` wide whose pixel the sea reads as
    `pixel_rad`: the sea takes its pixel from the camera's angle, which an orthographic
    projection otherwise ignores."""
    lens = bpy.data.cameras.new("probe")
    lens.type, lens.ortho_scale = "ORTHO", span_m
    lens.angle_x = pixel_rad * px
    return lens


def _camera(built: scene.Built, scenario: Scenario, band: Band) -> bpy.types.Object:
    return built.cameras[next(m.name for m in scenario.mounts if m.camera.band == band)]


def radiance(band: Band, size: tuple[int, int]) -> np.ndarray:
    built = scene.build(SCENARIO, band)
    sc = bpy.context.scene
    sc.camera = _camera(built, SCENARIO, band)
    # Build already set the band's engine and denoiser; only the sample count is the
    # test's own.
    sc.cycles.samples = SAMPLES
    return shoot(size, f"drift_{band}")


def texture(rows: np.ndarray) -> float:
    """Spread within each row, as median absolute deviation."""
    return float(np.median(np.abs(rows - np.median(rows, axis=1, keepdims=True))))


def test_the_ir_sky_falls_to_ambient_and_the_far_sea_settles() -> None:
    """One frame serves both checks: a module fixture renders once per test worker."""
    frame = radiance("ir", (640, 512))
    _sky_runs_from_cold_overhead_to_ambient_at_the_horizon(frame)
    _sea_texture_fades_with_range(frame)


def _sky_runs_from_cold_overhead_to_ambient_at_the_horizon(frame: np.ndarray) -> None:
    horizon = frame.shape[0] // 2
    ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
    just_above = float(np.median(frame[horizon - 6 : horizon - 1]))
    assert just_above == pytest.approx(ambient, rel=0.02)
    # The top five rows' centre, on a level camera whose horizon is the middle row.
    height, width = frame.shape
    ir = next(m for m in SCENARIO.mounts if m.camera.band == "ir")
    half = math.tan(math.radians(ir.camera.hfov_deg) / 2) * height / width
    top = math.atan(half * (1 - 5 / height))
    sky = SCENARIO.sky
    expected = float(lwir.sky_radiance(top, sky.t_air_k, sky.atmosphere))
    assert float(np.median(frame[:5])) == pytest.approx(expected, rel=0.02)


def _sea_texture_fades_with_range(frame: np.ndarray) -> None:
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
    """The near sea of an ir frame, optionally with the wave relief flattened: the
    shaded waves and the displaced ones, whose tilt the Geometry node's normal holds."""
    built = scene.build(scenario, "ir")
    if not waves:
        tree = bpy.data.materials["sea"].node_tree
        output = next(
            n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
        )
        for link in output.inputs["Displacement"].links:
            tree.links.remove(link)
        flat = tree.nodes.new("ShaderNodeNewGeometry").outputs["Normal"]
        tree.links.new(flat, tree.nodes["wave_normal"].inputs["Vector"])
    sc = bpy.context.scene
    sc.camera = _camera(built, scenario, "ir")
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


LOOP = _clear(variant(outputs={"duration_s": 30.0, "fps": 1, "loop": True}))


@pytest.mark.render
@pytest.mark.parametrize(
    ("scenario", "frame"),
    # Past half-way, a loop's atan2 clock reads t - span.
    [(SCENARIO, 0), (LOOP, 20)],
)
def test_the_shader_tilts_the_sea_by_the_field_s_slope(
    scenario: Scenario, frame: int
) -> None:
    span_m, px = 0.25, 64
    scene.build(scenario, "eo")
    bpy.context.scene.frame_set(frame)
    tree = bpy.data.materials["sea"].node_tree
    shown = tree.nodes.new("ShaderNodeVectorMath")
    shown.operation = "MULTIPLY_ADD"  # (v + 1) / 2, so the emission stays positive
    shown.inputs[2].default_value = (0.5, 0.5, 0.5)
    emission = tree.nodes.new("ShaderNodeEmission")
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    link = tree.links.new
    link(shown.outputs["Vector"], emission.inputs["Color"])
    link(emission.outputs["Emission"], output.inputs["Surface"])

    # A pixel fine enough to draw every wave, as far as its box passes it, and far
    # enough that every ray runs straight down to within a ripple's phase. Not
    # orthographic: Cycles dices what that camera sees as off-screen.
    height_m = 5000.0
    lens = bpy.data.cameras.new("probe")
    lens.angle_x, lens.clip_end = span_m / height_m, 2 * height_m
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    camera.location = (0.0, 0.0, height_m)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01  # the pixel centre, as the numpy grid
    # The build's finest edge; a pixel this fine would dice to it anyway.
    water = bpy.data.objects["sea"]
    dice_m = water.dimensions.x / SEA_CELLS / 2**sc.cycles.max_subdivisions
    assert dice_m <= DICED_EDGE_M
    water.modifiers["dice"].adaptive_pixel_size = dice_m * px / span_m
    # Dicing the kilometres of sea off-screen is most of the render. This much coarser
    # than the build leaves every pixel as it was.
    sc.cycles.offscreen_dicing_scale = 4 * OFFSCREEN_DICING

    def read(socket: bpy.types.NodeSocket, unit: float) -> np.ndarray:
        """`socket`, east, north and up, in units of `unit`, which keep it in -1..1."""
        shown.inputs[1].default_value = (0.5 / unit,) * 3
        link(socket, shown.inputs[0])
        return np.moveaxis(2 * unit * shoot_rgb((px, px), "slope_probe") - unit, 2, 0)

    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facet = read(geometry.outputs["Normal"], 1.0)
    rest = read(tree.nodes["sea_rest"].outputs["Vector"], 8.0)[:2]
    normal = read(tree.nodes["wave_normal"].outputs["Vector"], 1.0)

    # The rest of the waves tilt the facet where its water was at rest.
    tall, shaded = sea.displaced(
        (scene.wind_waves(scenario), scene.swell_waves(scenario))
    )
    t_s = scenario.outputs.times_s[frame]
    seen = waves.filtered(shaded[0] + shaded[1], span_m / px)
    slope = waves.slope(seen, *rest, t_s)
    tilted = facet - np.array([slope[0], slope[1], np.zeros_like(slope[0])])
    tilted /= np.linalg.norm(tilted, axis=0)
    assert np.abs(normal - tilted).max() < 2e-3

    # The facets lie on the tall waves' Gerstner surface, to what linear interpolation
    # over a facet misses (h^2 / 8 of the second derivative) and, for the normal, what
    # the difference across neighbouring facets adds (h^2 / 6 of the third).
    centres = (np.arange(px) + 0.5) * span_m / px - span_m / 2
    east, north = np.meshgrid(centres, centres[::-1])
    at_rest = np.array(waves.gerstner_rest_m(tall, east, north, t_s))
    curved = sum(w.amplitude_m * w.k_rad_m**2 for w in tall)
    assert np.abs(rest - at_rest).max() < dice_m**2 / 8 * curved
    bent = sum(w.amplitude_m * w.k_rad_m**3 for w in tall)
    surface = np.array(waves.gerstner_normal(tall, *at_rest, t_s))
    assert np.abs(facet - surface).max() < dice_m**2 * (1 / 8 + 1 / 6) * bent


@pytest.mark.render
def test_the_shader_reads_the_foam_memory_where_the_water_is() -> None:
    """Straight down on the undisplaced sea, at the clip's first and last frames:
    each pixel over a cell of the finest grid reads that cell of that frame, the
    water having drifted since."""
    clip = _clear(variant(OPEN_SEA, outputs={"duration_s": 1.0, "fps": 4}))
    stale = _frames_folder() / "0.25m" / "foam_0001.exr"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.touch()
    built = scene.build(clip, "eo")
    # A build clears the last one's frames.
    assert not stale.exists()
    grid = scene._foam_grids(clip, "eo", built.cameras, scene.wind_waves(clip))[0]
    tree = bpy.data.materials["sea"].node_tree
    texture = tree.nodes[f"sea_foam_{grid.cell_m:g}m"]
    texture.interpolation = "Closest"
    emission = tree.nodes.new("ShaderNodeEmission")
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    tree.links.new(
        tree.nodes["whitecap_bubbles"].outputs["Value"], emission.inputs["Color"]
    )
    # Flat: a pixel straight down then sees the water at rest under it.
    tree.links.remove(output.inputs["Displacement"].links[0])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    px = 32
    row, column = grid.rows // 2, grid.columns // 2
    camera = bpy.data.objects.new(
        "probe", overhead(px * grid.cell_m, eo_pixel_rad(clip), px)
    )
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01  # the pixel centre, a cell's centre
    sc.view_settings.view_transform = "Standard"
    drift = sea.surface_drift(clip.sea, clip.outputs)
    assert drift != (0.0, 0.0)
    first = Path(bpy.path.abspath(texture.image.filepath))
    for frame in (0, len(clip.outputs.times_s) - 1):
        sc.frame_set(frame)
        t_s = clip.outputs.times_s[frame]
        # The cell's water has drifted v t downwind.
        camera.location = (
            grid.west_m + (column + px / 2) * grid.cell_m + drift[0] * t_s,
            grid.south_m + (row + px / 2) * grid.cell_m + drift[1] * t_s,
            10.0,
        )
        rendered = shoot((px, px), f"memory_probe_{frame}")
        stored = render._pixels(first.with_name(f"foam_{frame + 1:04d}.exr"))[..., 0]
        expected = stored[row : row + px, column : column + px][::-1]
        assert expected.max() > 0.0
        np.testing.assert_allclose(rendered, expected, atol=2e-3)


@pytest.mark.render
def test_a_pixel_takes_as_roughness_the_slope_it_does_not_draw() -> None:
    """Top down from high enough that the footprint sits among the waves' fades, under
    a sun overhead, so every facet can glint."""
    height_m, span_m, px = 1500.0, 600.0, 32
    scene.build(_clear(variant(sky={"sun_elevation_deg": 90.0})), "eo")
    tree = bpy.data.materials["sea"].node_tree
    emission = tree.nodes.new("ShaderNodeEmission")
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    tree.links.new(
        tree.nodes["sea_roughness"].outputs["Value"], emission.inputs["Color"]
    )
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    # Straight down, an orthographic ray has no horizontal to turn against the wind:
    # held at 45 degrees, where the wind's axes mix evenly.
    facing = tree.nodes["sea_wind_facing"]
    for socket in facing.inputs:
        for link in socket.links:
            tree.links.remove(link)
        socket.default_value = math.sqrt(0.5)
    pixel_rad = eo_pixel_rad(SCENARIO)
    lens = overhead(span_m, pixel_rad, px)
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

    centres = (np.arange(px) + 0.5) * span_m / px - span_m / 2
    east, north = np.meshgrid(centres, centres[::-1])
    distance = np.sqrt(east**2 + north**2 + height_m**2)
    wind = scene.wind_waves(SCENARIO)
    footprints = distance.ravel() * pixel_rad
    lobes = [
        lobe(
            *unresolved(SCENARIO, f, f, 0.5), f, f, wind, sc.cycles.samples, glints=True
        )
        for f in footprints
    ]
    # Top down, both footprints agree.
    expected = np.array([min((a * c) ** 0.125, 1.0) for a, c in lobes]).reshape(
        distance.shape
    )
    assert np.abs(rendered - expected).max() < 5e-3


def unresolved(
    scenario: Scenario, along_m: float, across_m: float, cos2: float
) -> tuple[float, float]:
    """The pixel's unresolved variance along and across a view at cos^2 `cos2` to the
    wind, each twice its axis's."""
    wind, swell = scene.wind_waves(scenario), scene.swell_waves(scenario)
    speed, axis = scenario.sea.wind_speed_mps, math.radians(scenario.sea.wind_from_deg)
    up_a, cross_a = waves.unresolved_slope_variances(speed, axis, wind, swell, along_m)
    up_c, cross_c = waves.unresolved_slope_variances(speed, axis, wind, swell, across_m)
    return (
        2 * (up_a * cos2 + cross_a * (1 - cos2)),
        2 * (up_c * (1 - cos2) + cross_c * cos2),
    )


def lobe(
    along: float,
    across: float,
    along_m: float,
    across_m: float,
    wind: tuple[waves.Wave, ...],
    samples: int,
    glints: bool,
) -> tuple[float, float]:
    """The unresolved variance left to the lobe, wider axis first, once the glitter
    takes its share of the narrower, where the sun can glint; all of it where it
    cannot."""
    wide, narrow = max(along, across), min(along, across)
    cells = along_m * across_m / waves.specular_cell_m2(wind)
    widen = max(cells / samples - 1, 0.0) * SUN_SLOPE_RADIUS**2
    carried = max(narrow - widen, 0.0) if glints else 0.0
    left = max(wide - carried, 0.0)
    return max(left, 1e-12), max(narrow - carried, left / 100, 1e-12)


def _met(
    tree: bpy.types.NodeTree, emission: bpy.types.Node, px: tuple[int, int]
) -> np.ndarray:
    """Where each pixel's ray met the waves, rendered through `emission`: its depth in
    km, and the cosine it met them at."""
    camera = tree.nodes.new("ShaderNodeCameraData")
    geometry = tree.nodes.new("ShaderNodeNewGeometry")
    facing = tree.nodes.new("ShaderNodeVectorMath")
    facing.operation = "DOT_PRODUCT"
    tree.links.new(geometry.outputs["Incoming"], facing.inputs[0])
    tree.links.new(geometry.outputs["Normal"], facing.inputs[1])
    km = tree.nodes.new("ShaderNodeMath")
    km.operation = "DIVIDE"
    km.inputs[1].default_value = 1000.0
    tree.links.new(camera.outputs["View Distance"], km.inputs[0])
    hit = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(km.outputs["Value"], hit.inputs["X"])
    tree.links.new(facing.outputs["Value"], hit.inputs["Y"])
    tree.links.new(hit.outputs["Vector"], emission.inputs["Color"])
    for link in list(emission.inputs["Strength"].links):
        tree.links.remove(link)
    return shoot_rgb(px, "hit_probe")[:, :, :2]


@pytest.mark.render
def test_a_grazing_pixel_stretches_its_lobe_along_the_view() -> None:
    """The specular's Anisotropy, read back at a grazing view, against the footprints
    the pixel covers: 10 m up, looking 3 deg down, the sea from ~100 m to ~1 km. The
    sun has set and the gusts are stilled: the lobe holds the footprints' variance
    alone."""
    scene.build(_clear(variant(sky={"sun_elevation_deg": -5.0})), "eo")
    tree = bpy.data.materials["sea"].node_tree
    tree.nodes["sea_gust_variance"].inputs["Value_001"].default_value = 0.0
    anisotropy = tree.nodes["sea_specular"].inputs["Anisotropy"].links[0].from_socket
    emission = tree.nodes.new("ShaderNodeEmission")
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    tree.links.new(anisotropy, emission.inputs["Strength"])
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
    met = _met(tree, emission, px)

    bpy.context.view_layer.update()
    corners = [camera.matrix_world @ c for c in lens.view_frame(scene=sc)]
    top_right, bottom_right, bottom_left, top_left = (np.array(c) for c in corners)
    origin = np.array(camera.matrix_world.translation)
    pixel_rad = lens.angle_x / px[0]
    wind = scene.wind_waves(SCENARIO)
    checked = []
    for row in range(px[1]):
        for col in range(px[0]):
            u, v = (col + 0.5) / px[0], (row + 0.5) / px[1]
            left = top_left + v * (bottom_left - top_left)
            right = top_right + v * (bottom_right - top_right)
            ray = left + u * (right - left) - origin
            ray /= np.linalg.norm(ray)
            if ray[2] > -0.02:  # the horizon's rows and the sky
                continue
            across = 1000.0 * met[row, col, 0] * pixel_rad
            along = across / max(abs(met[row, col, 1]), 1e-3)
            cos2 = ray[1] ** 2 / (ray[0] ** 2 + ray[1] ** 2)
            v_along, v_across = lobe(
                *unresolved(SCENARIO, along, across, cos2),
                along,
                across,
                wind,
                sc.cycles.samples,
                glints=False,
            )
            expected = (1 - math.sqrt(v_across / v_along)) / 0.9
            assert rendered[row, col] == pytest.approx(expected, abs=0.02), (row, col)
            checked.append(expected)
    assert len(checked) > 100
    # Well clear of the tolerance, so a flat lobe cannot pass.
    assert max(checked) > 0.05, "a grazing view is anisotropic"


def _glitter_off_changes(sun_elevation_deg: float, sun_bearing_deg: float) -> float:
    """How much of a frame the glitter's tilt moves, built with the sun ahead and then
    given the sun through the world, as a scene's sun is set after its build."""
    ahead = {"sun_elevation_deg": 12.0, "sun_bearing_deg": 0.0}
    scene.build(_clear(variant(sky=ahead)), "eo")
    sc = bpy.context.scene
    sc.world["sun_elevation"] = math.radians(sun_elevation_deg)
    sc.world["sun_rotation"] = math.radians(sun_bearing_deg)
    sc.world.update_tag()
    lens = bpy.data.cameras.new("probe")
    lens.clip_end, lens.angle_x = 1e5, math.radians(20.0)
    camera = bpy.data.objects.new("probe", lens)
    sc.collection.objects.link(camera)
    sc.camera = camera
    camera.location = (0.0, 0.0, 15.0)
    camera.rotation_euler = (math.radians(78.0), 0.0, 0.0)
    glittering = shoot((96, 64), "glitter_on")
    tree = bpy.data.materials["sea"].node_tree
    for link in list(tree.nodes["glitter_tilt"].outputs["Vector"].links):
        link.to_socket.default_value = (0.0, 0.0, 0.0)
        tree.links.remove(link)
    plain = shoot((96, 64), "glitter_off")
    return float(np.abs(glittering - plain).mean() / plain.mean())


@pytest.mark.parametrize(
    ("sun_elevation_deg", "sun_bearing_deg", "glints"),
    [
        pytest.param(12.0, 0.0, True, id="ahead"),
        pytest.param(12.0, 180.0, False, id="behind"),
        pytest.param(-5.0, 0.0, False, id="set"),
    ],
)
def test_the_glitter_tilts_only_facets_that_can_mirror_the_sun(
    sun_elevation_deg: float, sun_bearing_deg: float, glints: bool
) -> None:
    changed = _glitter_off_changes(sun_elevation_deg, sun_bearing_deg)

    assert (changed > 0.01) == glints, changed


PROBE_HEIGHT_M, PROBE_FOV_DEG, PROBE_PX = 15.0, 4.0, (64, 16)


def _mirrored(cosine: np.ndarray) -> np.ndarray:
    """Seawater's Fresnel reflectance at incidence `cosine`."""
    theta = np.arccos(np.clip(cosine, 0.0, 1.0))
    return 1 - lwir.fresnel_emissivity(theta, SEAWATER_IOR, 0.0)


def _sea_reflectance(scenario: Scenario, depression_deg: float) -> float:
    """What the sea reflects of a uniform sky: rays traced to the tall waves' Gerstner
    surface, and around each hit's normal facets of the rest of Cox & Munk's Gaussian
    slopes. Each facet mirrors F of the sky, weighted by the area it shows the view,
    less the share of its mirror ray that Smith's Beckmann shadowing (Walter et al.
    2007) sends into another wave. A hit whose own mirror ray meets another tall wave
    sees that wave's F of the sky instead.

    Crests hide the faces turned away, whose grazing F is the highest, more than
    Smith's uncorrelated masking does. Only the slope the probe's centre pixel leaves
    unresolved shadows, along the ray's azimuth: the drawn waves are normals, which
    occlude nothing.
    """
    speed = scenario.sea.wind_speed_mps
    wind_axis = math.radians(scenario.sea.wind_from_deg)
    d = math.radians(depression_deg)
    # The probe looks north; x east, y north.
    up, cross = waves.cox_munk_variances(speed)
    c2 = math.cos(wind_axis) ** 2
    sides = np.array((up * (1 - c2) + cross * c2, up * c2 + cross * (1 - c2)))
    wind, swell = scene.wind_waves(scenario), scene.swell_waves(scenario)
    tall, _ = sea.displaced((wind, swell))
    geometry = np.array(
        [
            sum(w.amplitude_m**2 * getattr(w, k) ** 2 / 2 for w in tall)
            for k in ("k_east_rad_m", "k_north_rad_m")
        ]
    )

    rays = 2000
    rng = np.random.default_rng(0)
    origin = np.vstack(
        [rng.uniform(-5000.0, 5000.0, (2, rays)), np.full(rays, PROBE_HEIGHT_M)]
    )
    view = np.array([0.0, -math.cos(d), math.sin(d)])  # toward the camera
    ahead = np.repeat(-view[:, None], rays, axis=1)
    rest = waves.gerstner_hit(tall, origin, ahead, 0.0, 0.25)[:2]
    moved = np.array(waves.gerstner_offset_m(tall, *rest, 0.0))
    normal = np.array(waves.gerstner_normal(tall, *rest, 0.0))
    toward = view @ normal
    bounced = 2 * toward * normal - view[:, None]
    hit = np.array([*rest, np.zeros(rays)]) + moved + 1e-3 * normal
    *beyond, blocked = waves.gerstner_hit(tall, hit, bounced, 0.0, 0.25)
    met = np.abs(
        np.sum(bounced * np.array(waves.gerstner_normal(tall, *beyond, 0.0)), 0)
    )
    sky = np.where(
        blocked,
        1 - lwir.fresnel_emissivity(np.arccos(np.clip(met, 0, 1)), SEAWATER_IOR, 0.0),
        1.0,
    )

    unit = lwir._facets(1.0)
    tilt = -unit[:, :2] / unit[:, 2:] * np.sqrt(np.maximum(sides - geometry, 0.0))
    # Slopes add: the hit's, then each facet's about it.
    slope = (-normal[:2] / normal[2])[:, :, None] + tilt.T[:, None, :]
    facet = np.stack([-slope[0], -slope[1], np.ones_like(slope[0])])
    facet /= np.linalg.norm(facet, axis=0)
    facing = np.einsum("i,inm->nm", view, facet)
    # A hit is a pixel's share of the view; its facets split it by the area each shows.
    area = np.maximum(facing, 0.0) / facet[2] * (normal[2] / toward)[:, None]
    mirror = 2 * facing * facet - view[:, None, None]
    across_m = PROBE_HEIGHT_M / math.sin(d) * math.radians(PROBE_FOV_DEG) / PROBE_PX[0]
    left = [
        waves.unresolved_slope_variances(speed, wind_axis, wind, swell, f)
        for f in (across_m, across_m / math.sin(d))
    ]
    # East takes the across footprint, north the along one.
    east, north = (
        up * w + cross * (1 - w)
        for (up, cross), w in zip(left, (1 - c2, c2), strict=True)
    )
    run = np.maximum(np.hypot(mirror[0], mirror[1]), 1e-12)
    sigma = np.sqrt(east * mirror[0] ** 2 + north * mirror[1] ** 2) / run
    a = np.tan(np.arcsin(np.clip(mirror[2], 1e-6, 1.0))) / (math.sqrt(2) * sigma)
    lam = (np.exp(-a * a) / (a * math.sqrt(math.pi)) - np.vectorize(math.erfc)(a)) / 2
    lit = np.where(mirror[2] > 0.0, 1 / (1 + lam), 0.0)
    return float(np.sum(area * _mirrored(facing) * lit * sky[:, None]) / np.sum(area))


def _uniform_sky_reflectance(scenario: Scenario, depression_deg: float) -> float:
    """The sea's reflection alone of a white sky, looking down at `depression_deg`. A
    steep frame sees a few metres of sea tilted by the long wave under it; frames a
    long wave apart average to the sea's."""
    scene.build(scenario, "eo")
    world = bpy.context.scene.world.node_tree
    background = world.nodes["Background"]
    for link in list(background.inputs["Color"].links):
        world.links.remove(link)
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    tree = bpy.data.materials["sea"].node_tree
    tree.nodes["sea_body_color"].inputs["Vector"].default_value = (0, 0, 0)
    # The sea's reflection alone: no whitecap foam.
    for link in list(tree.nodes["whitecap_foam"].outputs["Value"].links):
        mixed = link.to_socket
        tree.links.remove(link)
        mixed.default_value = 0.0
    lens = bpy.data.cameras.new("probe")
    lens.clip_end, lens.angle_x = 1e5, math.radians(PROBE_FOV_DEG)
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    sc.camera = camera
    camera.rotation_euler = (math.radians(90.0 - depression_deg), 0.0, 0.0)
    frames = []
    for i in range(32):
        camera.location = (97.0 * i, 59.0 * i, PROBE_HEIGHT_M)
        frames.append(shoot(PROBE_PX, f"reflectance_{i}").mean())
    return float(np.mean(frames))


@pytest.mark.parametrize("depression_deg", [3.0, 10.0, 40.0])
@pytest.mark.parametrize("wind_from_deg", [0.0, 90.0])
def test_the_sea_reflects_a_uniform_sky_as_its_waves_and_slopes_do(
    wind_from_deg: float, depression_deg: float
) -> None:
    """The lobe's distribution sets how much sky the sea returns near the horizon,
    looking upwind and across the wind."""
    scenario = _clear(variant(sea={"wind_from_deg": wind_from_deg}))
    expected = _sea_reflectance(scenario, depression_deg)

    rendered = _uniform_sky_reflectance(scenario, depression_deg)

    # A judgement.
    assert rendered == pytest.approx(expected, rel=0.06)


@pytest.mark.render
def test_the_rendered_whitecaps_cover_what_monahan_measured() -> None:
    # Nothing but sea in view: a lit hull would count as foam.
    windy = SCENARIO.model_copy(
        update={"sea": SCENARIO.sea.model_copy(update={"wind_speed_mps": 15.0})}
    )
    scene.build(windy, "eo")
    tree = bpy.data.materials["sea"].node_tree
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes["Material Output"]
    tree.links.new(tree.nodes["whitecaps"].outputs["Value"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    px = 400
    lens = overhead(1500.0, eo_pixel_rad(windy), px)
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    sc.view_settings.view_transform = "Standard"
    covered = float(np.mean(shoot((px, px), "whitecap_probe")))
    assert covered == pytest.approx(waves.whitecap_fraction(15.0), rel=0.2)


@pytest.mark.render
def test_a_pixel_wider_than_the_foam_lattice_s_walls_draws_their_mean() -> None:
    """Straight down, a pixel half a wall across and one three walls across: the
    first draws the lattice, the second its mean, aged foam's grey, at fresh foam's
    albedo."""
    windy = SCENARIO.model_copy(
        update={"sea": SCENARIO.sea.model_copy(update={"wind_speed_mps": 15.0})}
    )
    scene.build(windy, "eo")
    tree = bpy.data.materials["sea"].node_tree
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes["Material Output"]
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    sc = bpy.context.scene
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    sc.view_settings.view_transform = "Standard"
    height_m, span_m, px = 10.0, 40.0, 64

    def read(footprint_m: float, name: str) -> np.ndarray:
        lens = overhead(span_m, footprint_m / height_m, px)
        camera = bpy.data.objects.new("probe", lens)
        sc.collection.objects.link(camera)
        camera.location = (0.0, 0.0, height_m)
        sc.camera = camera
        tree.links.new(tree.nodes[name].outputs["Value"], emission.inputs["Color"])
        return shoot((px, px), f"{name}_{footprint_m:g}")

    def grey(footprint_m: float) -> np.ndarray:
        return read(footprint_m, "whitecaps") * read(footprint_m, "whitecap_bright")

    wide_m = 3 * LATTICE_WALL_M
    foamed = read(wide_m, "whitecaps") > 0.0
    assert foamed.mean() > 0.01, "no foam in view would pass vacuously"
    np.testing.assert_allclose(read(wide_m, "whitecap_foam"), grey(wide_m), atol=1e-3)
    np.testing.assert_allclose(
        read(wide_m, "whitecap_albedo"), wakes.FRESH_FOAM_REFLECTANCE, atol=1e-3
    )
    narrow_m = LATTICE_WALL_M / 2
    assert np.abs(read(narrow_m, "whitecap_foam") - grey(narrow_m)).max() > 0.1


def test_the_sky_draws_its_sun_where_the_sun_vector_points() -> None:
    """The hull's heating takes the sun from that vector and the sky draws it from the
    bearing; a sign between them mirrors the disc and its glitter east to west."""
    scene.build(SCENARIO, "eo")
    sc = bpy.context.scene
    camera = sc.camera
    camera.parent = None
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = Vector(_sun_vector(SCENARIO.sky)).to_track_quat(
        "-Z", "Y"
    )
    sc.cycles.samples = 4
    sc.cycles.use_denoising = False
    frame = shoot((200, 150), "sun")
    row, col = np.unravel_index(frame.argmax(), frame.shape)
    assert abs(row - 75) <= 2 and abs(col - 100) <= 2


@pytest.mark.render
@pytest.mark.parametrize(
    ("band", "ranges_m"),
    [("eo", (1000.0, 5000.0, 25000.0, 28000.0)), ("ir", (1000.0, 5000.0, 25000.0))],
)
def test_haze_leaves_a_black_card_its_share_of_the_sky(
    band: Band, ranges_m: tuple[float, ...]
) -> None:
    """Looking just above the horizon, so without the card the ray reaches the sky.
    One build and one sky per band, a card per range."""
    hazy = load(OPEN_SEA)
    hazy = hazy.model_copy(update={"outputs": SCENARIO.outputs})
    assert hazy.sky.extinction_per_m > 0, "clear air would pass this vacuously"
    scene.build(hazy, band)
    sc = bpy.context.scene
    far_m = max(ranges_m)
    assert far_m < sc.camera.data.clip_end, "past the far plane there is no haze"
    lens = bpy.data.cameras.new("probe")
    lens.angle, lens.clip_end = math.radians(0.2), 2 * far_m
    camera = bpy.data.objects.new("probe", lens)
    sc.collection.objects.link(camera)
    elevation = math.radians(0.2)  # under the haze's top at every range
    camera.location = (0.0, 0.0, hazy.rigs["bow"].height_m)
    camera.rotation_euler = (math.pi / 2 + elevation, 0.0, 0.0)  # towards +Y
    sc.camera = camera
    sc.cycles.samples = 16
    sky = shoot((16, 16), "haze_sky")

    black = bpy.data.materials.new("black")
    tree = black.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 0.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    haze(black)
    for range_m in ranges_m:
        bpy.ops.mesh.primitive_plane_add(size=0.01 * range_m)
        card = bpy.context.object
        card.rotation_euler = (math.pi / 2, 0.0, 0.0)
        card.location = (
            0.0,
            range_m,
            hazy.rigs["bow"].height_m + range_m * math.tan(elevation),
        )
        card.data.materials.append(black)
        seen = shoot((16, 16), "haze_card")
        mesh = card.data
        bpy.data.objects.remove(card)
        bpy.data.meshes.remove(mesh)

        distance_m = range_m / math.cos(elevation)
        if band == "eo":
            depth = hazy.sky.extinction_per_m * distance_m
        else:
            air = hazy.sky
            depth = float(
                lwir.path_optical_depth(distance_m, air.visibility_km, air.atmosphere)
            )
        share = float(np.median(seen / sky))
        assert share == pytest.approx(1 - math.exp(-depth), rel=0.01), range_m


@pytest.mark.render
def test_ir_takes_the_same_light_however_cycles_samples_it() -> None:
    """Light sampling and BSDF sampling estimate the same frame, unless a path one of
    them takes skips the haze."""
    # The hull stays: its emission is light the two samplings must agree on.
    hazy = load(BASELINE)
    hazy = hazy.model_copy(update={"outputs": SCENARIO.outputs})
    scene.build(hazy, "ir")
    sc = bpy.context.scene
    sc.cycles.samples = 32
    # Kept, the second render would sample as the first did.
    sc.render.use_persistent_data = False
    means = []
    for sampled in (True, False):
        sc.world.cycles.sampling_method = "AUTOMATIC" if sampled else "NONE"
        for material in bpy.data.materials:
            material.cycles.emission_sampling = "AUTO" if sampled else "NONE"
        frame = shoot((160, 120), f"sampled_{sampled}")
        means.append(lwir.brightness_temperature(frame.ravel()).mean())
    assert means[0] == pytest.approx(means[1], abs=0.01)


def srgb_counts(linear: np.ndarray) -> np.ndarray:
    """What the Standard view writes to 8 bits: clipped, IEC 61966-2-1 encoded."""
    x = np.clip(linear, 0.0, 1.0)
    encoded = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.round(255 * encoded)


class TestCamera:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        png = SCENARIO.outputs.model_copy(update={"format": "png"})
        scene.build(SCENARIO.model_copy(update={"outputs": png}), "eo")

    def through_the_camera(
        self,
        left: float,
        right: float,
        sun: float = 0.0,
        sun_radius_deg: float = 1.8,
        size: tuple[int, int] = (96, 54),
    ) -> np.ndarray:
        """A camera looking straight up at a sky of `left` and `right` halves, with a
        disc of `sun` overhead; what the camera makes of it, before the display."""
        sc = bpy.context.scene
        tree = sc.world.node_tree
        generated = tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
        direction = tree.nodes.new("ShaderNodeSeparateXYZ")
        tree.links.new(generated, direction.inputs[0])
        east = _math(tree, "GREATER_THAN", direction.outputs["X"], 0.0)
        halves = _math(tree, "MULTIPLY_ADD", east, right - left, left)
        overhead = _math(
            tree,
            "GREATER_THAN",
            direction.outputs["Z"],
            math.cos(math.radians(sun_radius_deg)),
        )
        sky = _math(tree, "MULTIPLY_ADD", overhead, sun, halves)
        background = tree.nodes["Background"]
        tree.links.new(sky, background.inputs["Color"])
        background.inputs["Strength"].default_value = 1.0
        camera = bpy.data.objects.new("probe", bpy.data.cameras.new("probe"))
        sc.collection.objects.link(camera)
        camera.location = (0.0, 0.0, 10.0)
        # Straight up; +X is the frame's right.
        camera.rotation_euler = (math.pi, 0.0, 0.0)
        sc.camera = camera
        sc.cycles.samples = 16
        # The build put the camera in the compositor; an exr takes its output
        # undisplayed.
        return shoot(size, "camera")

    def test_auto_exposure_takes_a_brighter_sky_to_the_same_picture(self) -> None:
        assert self.through_the_camera(4.0, 4.0) == pytest.approx(
            self.through_the_camera(1.0, 1.0), rel=1e-3
        )

    def test_auto_exposure_meters_the_log_average(self) -> None:
        """Halves of 1 and 4 average 2 in log and 2.5 in linear, so a linear meter reads
        each half 20% darker."""
        frame = self.through_the_camera(1.0, 4.0)
        width = frame.shape[1]
        left, right = (
            np.median(frame[:, : width // 4]),
            np.median(frame[:, -width // 4 :]),
        )
        assert (left, right) == pytest.approx((MID_GREY / 2, 2 * MID_GREY), rel=0.03)

    def test_the_camera_takes_the_same_picture_at_any_resolution(self) -> None:
        """The frame at twice the width, averaged back down, within a few counts almost
        everywhere. Below a few hundred pixels the glare's core falls inside one."""
        small = self.through_the_camera(1.0, 1.0, sun=1e4, size=(384, 216))
        large = self.through_the_camera(1.0, 1.0, sun=1e4, size=(768, 432))
        shrunk = large.reshape(216, 2, 384, 2).mean(axis=(1, 3))
        off = np.abs(srgb_counts(shrunk) - srgb_counts(small))
        assert np.percentile(off, 99) <= 4

    def test_the_glare_scatters_its_share_in_harvey_s_form(self) -> None:
        """A sun of about a pixel on black sky: the light beyond a few pixels of it is
        the glare's share of the kernel's weight there."""
        # Half-resolution glare blurs the first few pixels round the sun.
        width, height, beyond_px = 384, 216, 12
        frame = self.through_the_camera(
            0.0, 0.0, sun=1e6, sun_radius_deg=0.05, size=(width, height)
        )
        x = ((np.arange(width) + 0.5) / width * 2 - 1)[None, :]
        y = ((np.arange(height) + 0.5) / height * 2 - 1)[:, None] * height / width
        r = np.hypot(x, y)
        kernel = (1 + (r / GLARE_SHOULDER) ** 2) ** (-GLARE_SLOPE / 2)
        far = r * width / 2 > beyond_px
        expected = GLARE_SHARE * kernel[far].sum() / kernel.sum()
        assert frame[far].sum() / frame.sum() == pytest.approx(expected, rel=0.1)


class TestSensor:
    """The sensor and ISP on a flat sky, through a camera looking straight up."""

    SIZE = (96, 54)

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        png = SCENARIO.outputs.model_copy(update={"format": "png"})
        graded = load(OPEN_SEA, ["rigs.bow.cameras.eo.sensor.full_well_e = 1e4"])
        scene.build(
            graded.model_copy(update={"outputs": png, "sky": SCENARIO.sky}), "eo"
        )
        sc = bpy.context.scene
        camera = bpy.data.objects.new("probe", bpy.data.cameras.new("probe"))
        sc.collection.objects.link(camera)
        camera.location = (0.0, 0.0, 10.0)
        camera.rotation_euler = (math.pi, 0.0, 0.0)
        sc.camera = camera
        sc.cycles.samples = 4

    def through(
        self,
        colour: tuple[float, float, float],
        name: str = "bow_eo",
        seed: int = 0,
        frame: int = 0,
        sun: float = 0.0,
        **camera: dict[str, float | bool],
    ) -> np.ndarray:
        """The frame, RGB top row first, of a sky of `colour`, with `sun` added in a
        disc overhead if given, through a camera the size of the probe's with
        `camera`'s settings."""
        sc = bpy.context.scene
        sc.frame_set(frame)
        background = sc.world.node_tree.nodes["Background"]
        for link in list(background.inputs["Color"].links):
            sc.world.node_tree.links.remove(link)
        background.inputs["Color"].default_value = (*colour, 1.0)
        background.inputs["Strength"].default_value = 1.0
        if sun:
            tree = sc.world.node_tree
            generated = tree.nodes.new("ShaderNodeTexCoord").outputs["Generated"]
            direction = tree.nodes.new("ShaderNodeSeparateXYZ")
            tree.links.new(generated, direction.inputs[0])
            overhead = _math(
                tree,
                "GREATER_THAN",
                direction.outputs["Z"],
                math.cos(math.radians(1.8)),
            )
            tree.links.new(
                _math(tree, "MULTIPLY_ADD", overhead, sun, colour[0]),
                background.inputs["Color"],
            )
        width, height = self.SIZE
        settings = Camera.model_validate(
            {
                "band": "eo",
                "hfov_deg": math.degrees(sc.camera.data.angle_x),
                "width_px": width,
                "height_px": height,
                **camera,
            }
        )
        rig = SCENARIO.rigs["bow"]
        mount = Mount("bow", rig, name.removeprefix("bow_"), settings)
        respond(mount, substream(seed, f"sensor/{name}"))
        sc.render.image_settings.file_format = "OPEN_EXR"
        sc.render.resolution_x, sc.render.resolution_y = self.SIZE
        sc.render.filepath = str(Path(bpy.app.tempdir) / "sensor")
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(sc.render.filepath + ".exr")
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        bpy.data.images.remove(image)
        return pixels.reshape(height, width, 4)[::-1, :, :3]

    def test_an_ideal_camera_in_a_sensor_s_tree_takes_the_ideal_frame(self) -> None:
        """No noise, no mosaic: the glare's border falloff is all that varies."""
        frame = self.through((1.0, 1.0, 1.0))
        assert frame.mean() == pytest.approx(MID_GREY, rel=0.01)
        assert np.ptp(frame) < 0.002
        assert np.array_equal(frame[..., 0], frame[..., 2])

    def test_the_noise_is_shot_and_dark_noise_at_its_level(self) -> None:
        full_well_e, dark_e, gain_db = 400.0, 4.0, 6.0
        frame = self.through(
            (1.0, 1.0, 1.0),
            sensor={
                "full_well_e": full_well_e,
                "dark_noise_e": dark_e,
                "gain_db": gain_db,
            },
        )
        gain = 10 ** (gain_db / 20)
        x = MID_GREY
        expected = math.sqrt(
            x * gain / full_well_e + (gain * dark_e / full_well_e) ** 2
        )
        red = frame[..., 0]
        # The sites are one parity class of four; the interpolated ones vary less.
        sigmas = [red[i::2, j::2].std() for i in (0, 1) for j in (0, 1)]
        assert max(sigmas) == pytest.approx(expected, rel=0.1)
        assert red.mean() == pytest.approx(x, rel=0.02)
        assert not np.array_equal(red[:, 0], red[:, 1])
        assert not np.array_equal(red[-1], red[-2])

    def test_a_sensor_twice_as_fine_reads_each_colour_over_its_sites(self) -> None:
        """Scaled down 2x, a pixel averages one red, two green and one blue site:
        green's noise falls by sqrt(2), and no demosaic spreads it to neighbours."""
        full_well_e, dark_e = 400.0, 4.0
        width, _ = self.SIZE
        frame = self.through(
            (1.0, 1.0, 1.0),
            sensor={
                "full_well_e": full_well_e,
                "dark_noise_e": dark_e,
                "width_px": 2 * width,
            },
        )
        x = MID_GREY
        site = math.sqrt(x / full_well_e + (dark_e / full_well_e) ** 2)
        red, green = frame[..., 0], frame[..., 1]
        assert red.std() == pytest.approx(site, rel=0.1)
        assert green.std() == pytest.approx(site / math.sqrt(2), rel=0.1)
        across = np.corrcoef(green[:, :-1].ravel(), green[:, 1:].ravel())[0, 1]
        assert abs(across) < 0.1

    def test_a_flat_colour_survives_the_mosaic(self) -> None:
        colour = (0.8, 0.4, 0.2)
        frame = self.through(colour, sensor={"full_well_e": 1e12})
        inner = frame[2:-2, 2:-2].reshape(-1, 3)
        ratios = inner / inner[:, 1:2]
        assert ratios.mean(axis=0) == pytest.approx((2.0, 1.0, 0.5), rel=0.01)
        assert ratios.std(axis=0).max() < 0.01

    def test_vignetting_falls_as_the_fourth_power_of_the_cosine(self) -> None:
        sc = bpy.context.scene
        frame = self.through((1.0, 1.0, 1.0), lens={"vignetting": True})[..., 1]
        width, height = self.SIZE
        x = ((np.arange(width) + 0.5) / width * 2 - 1)[None, :]
        y = ((np.arange(height) + 0.5) / height * 2 - 1)[:, None] * height / width
        tangent = np.hypot(x, y) * math.tan(sc.camera.data.angle_x / 2)
        falloff = (1 + tangent**2) ** -2
        corner = (frame[0, 0] + frame[0, -1] + frame[-1, 0] + frame[-1, -1]) / 4
        centre = frame[
            height // 2 - 1 : height // 2 + 1, width // 2 - 1 : width // 2 + 1
        ]
        assert corner / centre.mean() == pytest.approx(
            falloff[0, 0] / falloff.max(), rel=0.02
        )

    def test_no_exposure_turns_a_frame_into_nans(self) -> None:
        """At the largest gain the ideal frame stays a number, and a sensor reads
        white."""
        brightest = {"exposure_ev": 126.0}
        ideal = self.through((1.0, 1.0, 1.0), processing=brightest)
        read = self.through(
            (1.0, 1.0, 1.0), sensor={"full_well_e": 1e4}, processing=brightest
        )
        assert not np.isnan(ideal).any()
        assert read == pytest.approx(1.0)

    def test_hdr_keeps_what_one_exposure_clips(self) -> None:
        clean = {"full_well_e": 1e12}
        once = self.through((1.0, 1.0, 1.0), sun=9.0, sensor=clean)[..., 1]
        twice = self.through(
            (1.0, 1.0, 1.0), sun=9.0, sensor={**clean, "hdr_ratio": 16.0}
        )[..., 1]
        height, width = once.shape
        y, x = np.indices(once.shape)
        r = np.hypot(x - (width - 1) / 2, y - (height - 1) / 2)
        disc, sky = r < 3, r > 20
        assert np.median(once[disc]) == 1.0
        assert np.median(twice[sky]) == pytest.approx(MID_GREY, rel=0.1)
        # Reinhard's L / (1 + L) undone: the disc is its sky plus nine skies.
        linear = twice / (1 - twice)
        assert np.median(linear[disc]) / np.median(linear[sky]) == pytest.approx(
            10.0, rel=0.15
        )

    def test_hdr_keeps_a_bright_colour_s_hue(self) -> None:
        frame = self.through(
            (8.0, 4.0, 2.0),
            sensor={"full_well_e": 1e12, "hdr_ratio": 16.0},
        )
        inner = frame[2:-2, 2:-2].reshape(-1, 3).mean(axis=0)
        assert inner / inner[1] == pytest.approx((2.0, 1.0, 0.5), rel=0.02)

    def test_hdr_follows_the_exposure(self) -> None:
        hdr = {"full_well_e": 1e12, "hdr_ratio": 16.0}
        frame = self.through(
            (1.0, 1.0, 1.0), sensor=hdr, processing={"exposure_ev": 1.0}
        )
        assert np.median(frame) == pytest.approx(2 * MID_GREY, rel=0.1)

    def test_noise_is_drawn_apart_per_camera_scenario_and_frame(self) -> None:
        noisy = {"full_well_e": 400.0, "dark_noise_e": 4.0}
        grey = (1.0, 1.0, 1.0)
        first = self.through(grey, "bow_a", sensor=noisy)[..., 1]
        assert np.array_equal(first, self.through(grey, "bow_a", sensor=noisy)[..., 1])
        for other in (
            self.through(grey, "bow_b", sensor=noisy),
            self.through(grey, "bow_a", seed=1, sensor=noisy),
            self.through(grey, "bow_a", frame=1, sensor=noisy),
        ):
            assert abs(np.corrcoef(first.ravel(), other[..., 1].ravel())[0, 1]) < 0.1

    def test_white_balance_tints_by_its_gains(self) -> None:
        clean = {"full_well_e": 1e12}
        frame = self.through(
            (1.0, 1.0, 1.0), sensor=clean, processing={"white_balance_k": -1500.0}
        )
        r, g, b = frame[2:-2, 2:-2].reshape(-1, 3).mean(axis=0)
        gain_r, gain_g, gain_b = colour.white_balance_gains(-1500.0)
        assert (r / g, b / g) == pytest.approx(
            (gain_r / gain_g, gain_b / gain_g), rel=0.01
        )

    def test_what_clipped_white_stays_white_under_any_balance(self) -> None:
        frame = self.through(
            (1.0, 1.0, 1.0),
            sensor={"full_well_e": 1e12},
            processing={"white_balance_k": 3000.0, "exposure_ev": 4.0},
        )
        assert frame[2:-2, 2:-2] == pytest.approx(1.0)


def test_a_hull_s_foam_trails_behind_it() -> None:
    """From above, the sea astern of a hull heading east is brighter than ahead."""
    scenario = variant(
        objects=[target("yacht", 300.0, heading_deg=90.0, speed_mps=10.0)],
        outputs={"format": "exr"},
    )
    scene.build(scenario, "eo")
    sc = bpy.context.scene
    top = bpy.data.objects.new("top", bpy.data.cameras.new("top"))
    sc.collection.objects.link(top)
    top.data.type = "ORTHO"
    top.data.ortho_scale = 128.0
    top.location = (0.0, 300.0, 200.0)
    sc.camera = top
    sc.cycles.samples = 16
    frame = shoot((128, 128), "wake")
    # A metre a pixel, east to the right; both slices sit clear of the hull.
    astern, ahead = frame[59:69, 24:44], frame[59:69, 84:104]
    assert astern.mean() > 1.5 * ahead.mean()


def _rig(*widths_px: int) -> Scenario:
    names = ["eo", *(f"eo_{i}" for i in range(1, len(widths_px)))]
    cameras = {
        name: Camera(band="eo", hfov_deg=45.0, width_px=w, height_px=w * 9 // 16)
        for name, w in zip(names, widths_px, strict=True)
    }
    return variant(rig={"pitch_deg": -3.0}, cameras=cameras, outputs={"format": "exr"})


def _same(rendered: np.ndarray, expected: np.ndarray) -> bool:
    """One frame, as far as Cycles on the GPU repeats itself: a few pixels move by
    float noise between identical renders."""
    # A judgement: far above that noise, far below a sea drawn for another pixel.
    return float(np.abs(rendered - expected).mean() / expected.mean()) < 1e-4


def _rendered_at(width_px: int, percent: int) -> np.ndarray:
    scenario = _rig(width_px)
    scene.build(scenario, "eo")
    sc = bpy.context.scene
    sc.render.resolution_percentage = percent
    sc.cycles.samples = 16
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.filepath = str(Path(bpy.app.tempdir) / f"half_{width_px}_{percent}")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(sc.render.filepath + ".exr")
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return pixels.reshape(image.size[1], image.size[0], 4)[..., :3]


def test_a_render_at_half_resolution_is_a_build_at_half_width() -> None:
    assert _same(_rendered_at(640, 50), _rendered_at(320, 100))


def _set_after(setting: str, built: bool) -> np.ndarray:
    """With a ship far enough off to be hazed, so the haze's sun shows."""
    outputs: dict[str, object] = {"format": "exr"}
    sky: dict[str, object] = {}
    if built and setting == "samples":
        outputs["samples"] = {"eo": 32}
    elif built:
        sky["sun_elevation_deg"] = 12.0
    scenario = variant(
        rig={"pitch_deg": -3.0},
        cameras={"eo": {"width_px": 160, "height_px": 90}},
        objects=[preset_target("multipurpose_freighter", 3e3)],
        outputs=outputs,
        sky=sky,
    )
    scene.build(scenario, "eo")
    sc = bpy.context.scene
    if not built:
        shoot((160, 90), f"before_{setting}")
        if setting == "samples":
            sc.cycles.samples = 32
        else:
            sc.world["sun_elevation"] = math.radians(12.0)
            # As the UI does; a Python assignment to an ID property tags nothing.
            sc.world.update_tag()
    return shoot((160, 90), f"after_{setting}_{built}")


@pytest.mark.parametrize("setting", ["samples", "sun"])
def test_a_setting_changed_after_the_build_renders_as_if_built(setting: str) -> None:
    assert _same(_set_after(setting, built=False), _set_after(setting, built=True))


def test_each_camera_draws_for_its_own_pixel() -> None:
    """As `render` takes them, one after another in one build."""
    pair = _rig(640, 320)
    built = scene.build(pair, "eo")
    sc = bpy.context.scene
    sc.cycles.samples = 16
    fine, coarse = (built.cameras[m.name] for m in pair.mounts if m.camera.band == "eo")
    sc.camera = fine
    shoot((640, 360), "fine")
    sc.camera = coarse
    after = shoot((320, 180), "coarse")
    alone = _rig(320)
    built = scene.build(alone, "eo")
    # The build starts from factory settings, a new scene.
    sc = bpy.context.scene
    sc.cycles.samples = 16
    sc.camera = built.cameras[alone.mounts[0].name]
    assert _same(after, shoot((320, 180), "alone"))


def test_a_photographed_sun_sits_where_the_scenario_puts_it() -> None:
    scenario = variant(
        sky={"hdri": "kloofendal_48d_partly_cloudy", "sun_bearing_deg": 70.0},
        outputs={"format": "exr"},
    )
    scene.build(scenario, "eo")
    sc = bpy.context.scene
    camera = sc.camera
    camera.parent = None
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_mode = "QUATERNION"
    sun = Vector(_sun_vector(scenario.sky))
    camera.rotation_quaternion = sun.to_track_quat("-Z", "Y")
    camera.data.angle_x = math.radians(10.0)
    sc.cycles.samples = 4
    frame = shoot((64, 64), "photo_sun")
    row, col = np.unravel_index(np.argmax(frame), frame.shape)
    assert abs(row - 31.5) <= 2 and abs(col - 31.5) <= 2


@pytest.mark.parametrize("name", skies.library())
def test_the_committed_sun_is_what_the_photo_holds(name: str) -> None:
    photo = skies.library()[name]
    image = bpy.data.images.load(str(download(name, photo.url, photo.sha256)))
    w, h = image.size
    pixels = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    sun = skies.sun(pixels.reshape(h, w, 4)[::-1, :, :3])
    assert sun.bearing_deg == pytest.approx(photo.sun_bearing_deg, abs=0.06)
    if photo.sun_elevation_deg is None:
        assert sun.elevation_deg is None
    else:
        assert sun.elevation_deg == pytest.approx(photo.sun_elevation_deg, abs=0.06)


def test_a_cached_import_measures_as_the_first_one_did(tmp_path, monkeypatch) -> None:
    glbs = [n for n in manifest() if fetch(n).suffix == ".glb"]
    source = fetch(min(glbs, key=lambda n: manifest()[n].triangles))
    path = tmp_path / source.name
    shutil.copy(source, path)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    first = measure(path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    monkeypatch.setattr(
        "seascape.blender.meshes._parse", lambda _: pytest.fail("parsed again")
    )

    assert measure(path) == first


@pytest.mark.parametrize(
    "name", [n for n, a in manifest().items() if isinstance(a, Buoy)]
)
def test_a_buoy_floats_on_its_float(name: str) -> None:
    """The water meets the float: the mesh is near its widest at the waterline."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    parts = _import(name, "eo")
    bpy.context.view_layer.update()
    points = [
        mesh.matrix_world @ vertex.co
        for mesh in _meshes(parts)
        for vertex in mesh.data.vertices
    ]
    step = (max(p.z for p in points) - min(p.z for p in points)) / 100
    rings = defaultdict(list)
    for point in points:
        rings[round(point.z / step)].append(point.x)
    widths = {k: max(xs) - min(xs) for k, xs in rings.items()}
    at_water = widths[min(widths, key=abs)]
    assert at_water > 0.8 * max(widths.values())


def _horizon(aerosol_density: float) -> np.ndarray:
    """The build's Sky Texture at the horizon, with the sun behind the camera."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01
    sc.world = bpy.data.worlds.new("sky")
    tree = sc.world.node_tree
    texture = _sky_texture(tree, Sky(sun_bearing_deg=180.0))
    texture.aerosol_density = aerosol_density
    tree.links.new(texture.outputs["Color"], tree.nodes["Background"].inputs["Color"])
    lens = bpy.data.cameras.new("probe")
    lens.angle = math.radians(0.05)
    camera = bpy.data.objects.new("probe", lens)
    sc.collection.objects.link(camera)
    camera.rotation_euler = (math.pi / 2, 0.0, 0.0)  # level, towards +Y
    sc.camera = camera
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.resolution_x = sc.render.resolution_y = 5
    sc.render.filepath = str(Path(bpy.app.tempdir) / "horizon")
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(sc.render.filepath + ".exr")
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(5, 5, 4)[2, 2, :3]


def test_blender_s_aerosol_darkens_its_horizon() -> None:
    """Why the sky keeps Blender's default aerosol and `visibility_km` hazes only
    what is in front of it."""
    clear, hazy = _horizon(1.0), _horizon(6.0)
    assert hazy.sum() < 0.8 * clear.sum()
