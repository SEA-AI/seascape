"""Properties of a rendered LWIR frame that a refactor must not shift.

Skipped unless `--render` is given.
"""

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
    bump = next(
        n
        for n in bpy.data.materials["sea"].node_tree.nodes
        if n.bl_idname == "ShaderNodeBump"
    )
    if not waves:
        bump.inputs["Distance"].default_value = 0.0
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


@pytest.mark.render
@pytest.mark.parametrize(
    ("dimensions", "slope_per_unit"),
    [("3D", waves.NOISE_SLOPE_PER_UNIT), ("4D", waves.NOISE_SLOPE_PER_UNIT_4D)],
)
def test_the_noise_delivers_the_slope_it_is_asked_for(
    dimensions: str, slope_per_unit: float
) -> None:
    """A Blender change to the noise shows up as a number, not as a sea that looks
    slightly wrong."""
    span, px = 20.0, 1024  # 2 cm sampling
    bpy.ops.wm.read_factory_settings(use_empty=True)
    frame = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=span)
    material = bpy.data.materials.new("probe")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = dimensions
    noise.inputs["Scale"].default_value = 1.0  # one noise unit is one metre
    noise.inputs["Detail"].default_value = waves.NOISE_DETAIL
    noise.inputs["Roughness"].default_value = waves.NOISE_ROUGHNESS
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    position = tree.nodes.new("ShaderNodeNewGeometry").outputs["Position"]
    tree.links.new(position, noise.inputs["Vector"])
    tree.links.new(noise.outputs["Fac"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    bpy.context.object.data.materials.append(material)

    lens = bpy.data.cameras.new("probe")
    lens.type, lens.ortho_scale = "ORTHO", span
    camera = bpy.data.objects.new("probe", lens)
    frame.collection.objects.link(camera)
    camera.location = (0.0, 0.0, 10.0)
    frame.camera = camera
    frame.render.engine = "CYCLES"
    frame.cycles.samples = 1
    frame.cycles.use_denoising = False
    frame.render.resolution_x = frame.render.resolution_y = px
    frame.view_settings.view_transform = "Standard"

    fac = shoot((px, px), "noise_probe")
    gradient_y, gradient_x = np.gradient(fac.astype(np.float64), span / px)
    measured = float(np.sqrt(np.mean(gradient_x**2 + gradient_y**2)))
    assert measured == pytest.approx(slope_per_unit, abs=0.03)


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
