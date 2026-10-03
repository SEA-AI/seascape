"""Properties of a rendered frame that a refactor must not shift.

Skipped unless `--render` is given.
"""

import math
from pathlib import Path

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape import lwir, scene, sea, skies, waves
from seascape.assets import download, fetch, manifest
from seascape.config import Band, Scenario, load

pytestmark = pytest.mark.render


def _clear(scenario: Scenario) -> Scenario:
    """The probes read the sea's shader: haze would add its airlight to every one, and
    an 8-bit format the camera's glare, blur and exposure."""
    sky = scenario.sky.model_copy(update={"visibility_km": None})
    outputs = scenario.outputs.model_copy(update={"format": "exr"})
    return scenario.model_copy(update={"sky": sky, "outputs": outputs})


SCENARIOS = Path(__file__).parent.parent / "scenarios"
OPEN_SEA = SCENARIOS / "open-sea.toml"
SCENARIO = _clear(load(OPEN_SEA))
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


def eo_pixel_rad(scenario: Scenario) -> float:
    mount = next(m for m in scenario.rig.mounts if m.camera.kind == "eo")
    return math.radians(mount.camera.hfov_deg) / mount.camera.width_px


def overhead(span_m: float, pixel_rad: float, px: int) -> bpy.types.Camera:
    """An orthographic lens `span_m` across and `px` wide whose pixel the sea reads as
    `pixel_rad`: the sea takes its pixel from the camera's angle, which an orthographic
    projection otherwise ignores."""
    lens = bpy.data.cameras.new("probe")
    lens.type, lens.ortho_scale = "ORTHO", span_m
    lens.angle_x = pixel_rad * px
    return lens


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
    """Spread within each row, as median absolute deviation."""
    return float(np.median(np.abs(rows - np.median(rows, axis=1, keepdims=True))))


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    return radiance("ir", "ir", (640, 512))


def test_the_sky_runs_from_cold_overhead_to_ambient_at_the_horizon(frame) -> None:
    horizon = frame.shape[0] // 2
    ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
    just_above = float(np.median(frame[horizon - 6 : horizon - 1]))
    assert just_above == pytest.approx(ambient, rel=0.02)
    # The top five rows' centre, on a level camera whose horizon is the middle row.
    height, width = frame.shape
    ir = next(m for m in SCENARIO.rig.mounts if m.camera.kind == "ir")
    half = math.tan(math.radians(ir.camera.hfov_deg) / 2) * height / width
    top = math.atan(half * (1 - 5 / height))
    sky = SCENARIO.sky
    expected = float(lwir.sky_radiance(top, sky.t_air_k, sky.atmosphere))
    assert float(np.median(frame[:5])) == pytest.approx(expected, rel=0.02)


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


LOOP = _clear(
    load(
        OPEN_SEA, ["outputs.duration_s = 30", "outputs.fps = 1", "outputs.loop = true"]
    )
)


@pytest.mark.render
@pytest.mark.parametrize(
    ("scenario", "frame"),
    # Past half-way, a loop's atan2 clock reads t - span.
    [(SCENARIO, 0), (LOOP, 20)],
)
def test_the_shader_tilts_the_sea_by_the_field_s_slope(
    scenario: Scenario, frame: int
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
    link(shown.outputs["Value"], emission.inputs["Color"])
    link(emission.outputs["Emission"], output.inputs["Surface"])

    # A pixel fine enough to draw every wave, as the numpy field does.
    lens = overhead(span_m, 1e-5, px)
    camera = bpy.data.objects.new("probe", lens)
    sc = bpy.context.scene
    sc.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.cycles.filter_width = 0.01  # the pixel centre, as the numpy grid
    sc.view_settings.view_transform = "Standard"
    centres = (np.arange(px) + 0.5) * span_m / px - span_m / 2
    east, north = np.meshgrid(centres, centres[::-1])
    t_s = scenario.outputs.times_s[frame]
    slope = waves.slope(scene.wave_field(scenario), east, north, t_s)

    for axis in (0, 1):
        link(component.outputs["XY"[axis]], shown.inputs["Value"])
        rendered = 2 * shoot((px, px), "slope_probe") - 1
        expected = -slope[axis] / np.sqrt(1 + slope[0] ** 2 + slope[1] ** 2)
        assert np.abs(rendered - expected).max() < 2e-3, "XY"[axis]


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
    wind, swell = scene.wind_waves(SCENARIO), scene.swell_waves(SCENARIO)
    speed = SCENARIO.sea.wind_speed_mps
    footprints = distance.ravel() * pixel_rad
    variance = [
        waves.unresolved_slope_variance(speed, wind, swell, f) for f in footprints
    ]
    lobes = [
        lobe(v, v, f, f, wind, sc.cycles.samples)
        for v, f in zip(variance, footprints, strict=True)
    ]
    # Top down, both footprints agree.
    expected = np.array([min((a * c) ** 0.125, 1.0) for a, c in lobes]).reshape(
        distance.shape
    )
    assert np.abs(rendered - expected).max() < 5e-3


def lobe(
    along: float,
    across: float,
    along_m: float,
    across_m: float,
    wind: tuple[waves.Wave, ...],
    samples: int,
) -> tuple[float, float]:
    """The unresolved variance left to the lobe once the glitter takes its share."""
    cells = along_m * across_m / waves.specular_cell_m2(wind)
    widen = max(cells / samples - 1, 0.0) * sea.SUN_SLOPE_RADIUS**2
    carried = max(across - widen, 0.0)
    left = max(along - carried, 0.0)
    return max(left, 1e-12), max(across - carried, left / 100, 1e-12)


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
    pixel_rad = lens.angle_x / px[0]
    wind, swell = scene.wind_waves(SCENARIO), scene.swell_waves(SCENARIO)
    speed = SCENARIO.sea.wind_speed_mps
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
            v_along, v_across = lobe(
                waves.unresolved_slope_variance(speed, wind, swell, along),
                waves.unresolved_slope_variance(speed, wind, swell, across),
                along,
                across,
                wind,
                sc.cycles.samples,
            )
            expected = (1 - math.sqrt(v_across / v_along)) / 0.9
            assert rendered[row, col] == pytest.approx(expected, abs=0.02), (row, col)
            checked += 1
    assert checked > 100
    assert rendered.max() > 0.2, "a grazing view is anisotropic"


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
    tree.links.new(tree.nodes["whitecaps"].outputs["Color"], emission.inputs["Color"])
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
    camera.location = (0.0, 0.0, hazy.rig.height_m)
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
    scene.haze(black)
    for range_m in ranges_m:
        bpy.ops.mesh.primitive_plane_add(size=0.01 * range_m)
        card = bpy.context.object
        card.rotation_euler = (math.pi / 2, 0.0, 0.0)
        card.location = (
            0.0,
            range_m,
            hazy.rig.height_m + range_m * math.tan(elevation),
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
    hazy = load(SCENARIOS / "baseline.toml")
    hazy = hazy.model_copy(update={"outputs": SCENARIO.outputs})
    scene.build(hazy, "ir")
    sc = bpy.context.scene
    sc.cycles.samples = 64
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
        east = sea._math(tree, "GREATER_THAN", direction.outputs["X"], 0.0)
        halves = sea._math(tree, "MULTIPLY_ADD", east, right - left, left)
        overhead = sea._math(
            tree,
            "GREATER_THAN",
            direction.outputs["Z"],
            math.cos(math.radians(sun_radius_deg)),
        )
        sky = sea._math(tree, "MULTIPLY_ADD", overhead, sun, halves)
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
        assert (left, right) == pytest.approx(
            (scene.MID_GREY / 2, 2 * scene.MID_GREY), rel=0.03
        )

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
        width, height, beyond_px = 384, 216, 6
        frame = self.through_the_camera(
            0.0, 0.0, sun=1e6, sun_radius_deg=0.05, size=(width, height)
        )
        x = ((np.arange(width) + 0.5) / width * 2 - 1)[None, :]
        y = ((np.arange(height) + 0.5) / height * 2 - 1)[:, None] * height / width
        r = np.hypot(x, y)
        kernel = (1 + (r / scene.GLARE_SHOULDER) ** 2) ** (-scene.GLARE_SLOPE / 2)
        far = r * width / 2 > beyond_px
        expected = scene.GLARE_SHARE * kernel[far].sum() / kernel.sum()
        assert frame[far].sum() / frame.sum() == pytest.approx(expected, rel=0.1)


def test_a_hull_s_foam_trails_behind_it() -> None:
    """From above, the sea astern of a hull heading east is brighter than ahead."""
    scenario = load(
        OPEN_SEA,
        [
            'outputs.format = "exr"',
            'objects = [{ asset = "yacht", range_m = 300.0, bearing_deg = 0.0, '
            "heading_deg = 90.0, speed_mps = 10.0 }]",
        ],
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
    cameras = ", ".join(
        f'{{ kind = "eo", hfov_deg = 45.0, width_px = {w}, height_px = {w * 9 // 16} }}'
        for w in widths_px
    )
    return load(
        OPEN_SEA,
        [
            'outputs.format = "exr"',
            "rig.pitch_deg = -3.0",
            f'rig.pods = [{{ name = "bow", yaw_deg = 0.0, cameras = [{cameras}] }}]',
        ],
    )


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
    camera = '{ kind = "eo", hfov_deg = 45.0, width_px = 320, height_px = 180 }'
    sets = [
        'outputs.format = "exr"',
        "rig.pitch_deg = -3.0",
        f'rig.pods = [{{ name = "bow", yaw_deg = 0.0, cameras = [{camera}] }}]',
        "objects = [{ preset = 'container_ship', range_m = 3e3, bearing_deg = 0.0 }]",
    ]
    if built:
        sets.append(
            "outputs.samples.eo = 64"
            if setting == "samples"
            else "sky.sun_elevation_deg = 12.0"
        )
    scene.build(load(OPEN_SEA, sets), "eo")
    sc = bpy.context.scene
    if not built:
        shoot((320, 180), f"before_{setting}")
        if setting == "samples":
            sc.cycles.samples = 64
        else:
            sc.world["sun_elevation"] = math.radians(12.0)
            # As the UI does; a Python assignment to an ID property tags nothing.
            sc.world.update_tag()
    return shoot((320, 180), f"after_{setting}_{built}")


@pytest.mark.parametrize("setting", ["samples", "sun"])
def test_a_setting_changed_after_the_build_renders_as_if_built(setting: str) -> None:
    assert _same(_set_after(setting, built=False), _set_after(setting, built=True))


def test_each_camera_draws_for_its_own_pixel() -> None:
    """As `render` takes them, one after another in one build."""
    pair = _rig(640, 320)
    built = scene.build(pair, "eo")
    sc = bpy.context.scene
    sc.cycles.samples = 16
    fine, coarse = (built.cameras[m.name] for m in pair.rig.mounts)
    sc.camera = fine
    shoot((640, 360), "fine")
    sc.camera = coarse
    after = shoot((320, 180), "coarse")
    alone = _rig(320)
    built = scene.build(alone, "eo")
    # The build starts from factory settings, a new scene.
    sc = bpy.context.scene
    sc.cycles.samples = 16
    sc.camera = built.cameras[alone.rig.mounts[0].name]
    assert _same(after, shoot((320, 180), "alone"))


def test_a_photographed_sun_sits_where_the_scenario_puts_it() -> None:
    scenario = load(
        OPEN_SEA,
        [
            'sky.hdri = "kloofendal_48d_partly_cloudy"',
            "sky.sun_bearing_deg = 70.0",
            'outputs.format = "exr"',
        ],
    )
    scene.build(scenario, "eo")
    sc = bpy.context.scene
    camera = sc.camera
    camera.parent = None
    camera.location = (0.0, 0.0, 10.0)
    camera.rotation_mode = "QUATERNION"
    sun = Vector(scene._sun_vector(scenario.sky))
    camera.rotation_quaternion = sun.to_track_quat("-Z", "Y")
    camera.data.angle_x = math.radians(10.0)
    sc.cycles.samples = 4
    frame = shoot((64, 64), "photo_sun")
    row, col = np.unravel_index(np.argmax(frame), frame.shape)
    assert abs(row - 31.5) <= 2 and abs(col - 31.5) <= 2


def test_the_committed_suns_are_what_the_photos_hold() -> None:
    for name, photo in skies.library().items():
        image = bpy.data.images.load(str(download(name, photo.url, photo.sha256)))
        w, h = image.size
        pixels = np.empty(w * h * 4, np.float32)
        image.pixels.foreach_get(pixels)
        bpy.data.images.remove(image)
        sun = skies.sun(pixels.reshape(h, w, 4)[::-1, :, :3])
        assert sun.bearing_deg == pytest.approx(photo.sun_bearing_deg, abs=0.06), name
        if photo.sun_elevation_deg is None:
            assert sun.elevation_deg is None, name
        else:
            assert sun.elevation_deg == pytest.approx(
                photo.sun_elevation_deg, abs=0.06
            ), name


def test_the_committed_meshes_are_what_the_files_hold() -> None:
    for name, mesh in manifest().items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        assert scene.measure(fetch(name)) == (mesh.triangles, mesh.texture_px), name
