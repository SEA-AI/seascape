"""Render settings, the thermal image, and the object-index pass."""

import math
import tomllib
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import bpy
import cv2
import numpy as np
import pytest
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from seascape import render
from seascape.blender import blend, scene, sea
from seascape.blender.compositor import (
    FORMATS,
    JPEG_QUALITY,
    MID_GREY,
    _compositor,
    respond,
    set_output,
)
from seascape.blender.heat import Heat, zone_skin
from seascape.blender.sea import memory
from seascape.blender.sky import add_haze, add_world, sea_light
from seascape.config import (
    Agc,
    Band,
    Camera,
    ImageFormat,
    Microbolometer,
    Scenario,
    Sensor,
    load,
    substream,
)
from seascape.outputs import agc, labels
from seascape.outputs.calibration import Calibration
from seascape.physics import lwir, thermal, waves
from seascape.physics.wakes import Wake
from seascape.server import jobs
from tests.probes import probe
from tests.scenarios import (
    BASELINE,
    OPEN_SEA,
    SCENARIOS,
    UNDERWAY,
    target,
    variant,
    white_photo,
)

LINEAR = Agc(information_based=False, linear_fraction=1.0)


def _raise(*_: object) -> np.ndarray:
    raise RuntimeError("injected")


def exr_of(
    tmp_path: Path, radiance: list[float], width: int = 4, name: str = "probe"
) -> Path:
    """One row per value, bottom row first, as Blender orders pixels."""
    image = bpy.data.images.new("probe", width, len(radiance), float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    rows = np.repeat(np.array(radiance, dtype=np.float32), width)
    image.pixels.foreach_set(
        np.column_stack([rows, rows, rows, np.ones_like(rows)]).ravel()
    )
    image.file_format = "OPEN_EXR"
    path = tmp_path / f"{name}.exr"
    image.filepath_raw = str(path)
    image.save()
    bpy.data.images.remove(image)
    return path


IDEAL = (
    Camera(band="ir", hfov_deg=24.0, width_px=4, height_px=4),
    substream(0, "microbolometer"),
)


class TestThermalImage:
    """An inverted, flipped or sRGB-encoded frame is still a plausible-looking picture,
    so only the numbers catch it.
    """

    def test_a_png_is_centikelvin_with_the_top_row_first(self, tmp_path: Path) -> None:
        exr = exr_of(tmp_path, [lwir.band_radiance(t) for t in (272.0, 295.0)])
        render._write_thermal(exr, "png", agc.Agc(LINEAR), *IDEAL)
        t_k = agc.kelvin(exr.with_suffix(".png"))
        assert t_k is not None
        assert t_k[:, 0] == pytest.approx([295.0, 272.0], abs=0.01)

    def test_a_jpg_is_grey_through_the_agc(self, tmp_path: Path) -> None:
        exr = exr_of(tmp_path, [lwir.band_radiance(t) for t in (270.0, 285.0, 300.0)])
        render._write_thermal(exr, "jpg", agc.Agc(LINEAR), *IDEAL)
        jpg = cv2.imread(str(exr.with_suffix(".jpg")), cv2.IMREAD_GRAYSCALE)
        assert jpg is not None
        assert jpg[:, 0] == pytest.approx([255, 128, 0], abs=3)

    @pytest.mark.parametrize(
        ("fmt", "params"),
        [("png", []), ("jpg", [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])],
    )
    def test_only_a_jpg_takes_the_jpeg_quality(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fmt: ImageFormat,
        params: list[int],
    ) -> None:
        """OpenCV warns once per frame on a png given a jpeg flag."""
        seen: list[list[int]] = []

        def imwrite(_path: str, _image: np.ndarray, flags: list[int]) -> bool:
            seen.append(flags)
            return True

        monkeypatch.setattr(cv2, "imwrite", imwrite)
        render._write_thermal(
            exr_of(tmp_path, [lwir.band_radiance(285.0)]), fmt, agc.Agc(LINEAR), *IDEAL
        )
        assert seen == [params]

    def test_the_float_render_is_removed_on_success(self, tmp_path: Path) -> None:
        exr = exr_of(tmp_path, [lwir.band_radiance(285.0), lwir.band_radiance(295.0)])
        render._write_thermal(exr, "png", agc.Agc(LINEAR), *IDEAL)
        assert not exr.exists()

    def test_a_failure_keeps_the_float_render_and_leaks_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed conversion must not throw the render away."""
        exr = exr_of(tmp_path, [lwir.band_radiance(285.0)])
        before = len(bpy.data.images)
        monkeypatch.setattr(lwir, "brightness_temperature", _raise)
        with pytest.raises(RuntimeError):
            render._write_thermal(exr, "png", agc.Agc(LINEAR), *IDEAL)
        assert exr.exists()
        assert len(bpy.data.images) == before


@pytest.mark.parametrize(
    ("bolometer", "kind"),
    [({}, "BLACKMAN_HARRIS"), ({"pitch_um": 12.0, "f_number": 1.0}, "BOX")],
)
def test_a_microbolometer_integrates_over_its_pixel(
    bolometer: dict[str, float], kind: str
) -> None:
    scenario = load(OPEN_SEA)
    mount = scenario.band_mounts("ir")[0]
    camera = mount.camera.model_copy(
        update={"microbolometer": Microbolometer(**bolometer)}
    )
    respond(mount._replace(camera=camera), substream(0, "sensor"))
    assert bpy.context.scene.cycles.pixel_filter_type == kind


@pytest.mark.usefixtures("empty_blend")
def test_a_switch_set_off_is_muted() -> None:
    _compositor(0.0, True)
    mount = load(OPEN_SEA).band_mounts("eo")[0]
    respond(mount, substream(0, "sensor"))
    nodes = bpy.context.scene.compositing_node_group.nodes
    assert nodes["sensor.on"].mute
    with_sensor = mount.camera.model_copy(update={"sensor": Sensor(full_well_e=1e4)})
    respond(mount._replace(camera=with_sensor), substream(0, "sensor"))
    assert not nodes["sensor.on"].mute
    assert nodes["sensor.hdr"].mute


@pytest.mark.usefixtures("empty_blend")
def test_a_sensor_in_a_build_without_one_raises() -> None:
    _compositor(0.0)
    mount = load(OPEN_SEA).band_mounts("eo")[0]
    with_sensor = mount.camera.model_copy(update={"sensor": Sensor(full_well_e=1e4)})
    with pytest.raises(ValueError, match="no sensor"):
        respond(mount._replace(camera=with_sensor), substream(0, "sensor"))


class TestFrame:
    def test_a_png_keeps_its_one_16bit_channel(self, tmp_path: Path) -> None:
        counts = np.array([[29500], [27200]], np.uint16).repeat(4, axis=1)
        cv2.imwrite(str(tmp_path / "f.png"), counts)
        assert (render._frame(tmp_path / "f.png") == counts).all()

    def test_a_jpg_reads_as_rgb(self, tmp_path: Path) -> None:
        bgr = np.zeros((16, 16, 3), np.uint8)
        bgr[..., 2] = 255  # red
        cv2.imwrite(str(tmp_path / "f.jpg"), bgr)
        assert render._frame(tmp_path / "f.jpg")[8, 8] == pytest.approx(
            [255, 0, 0], abs=3
        )

    def test_an_exr_reads_top_row_first(self, tmp_path: Path) -> None:
        exr = exr_of(tmp_path, [1.0, 2.0])
        assert render._frame(exr)[:, 0] == pytest.approx(
            np.array([[2.0] * 3, [1.0] * 3])
        )


def built(band: Band, **outputs: object) -> bpy.types.Scene:
    # No ship: these are render settings.
    scenario = load(OPEN_SEA)
    scenario = scenario.model_copy(
        update={"outputs": scenario.outputs.model_copy(update=outputs)}
    )
    scene.build(scenario, band)
    return bpy.context.scene


class TestSettings:
    def test_every_format_maps_to_one_blender_identifier(self) -> None:
        """A format added to the Literal alone renders as whatever was set last."""
        assert set(FORMATS) == set(get_args(ImageFormat.__value__))

    @pytest.mark.usefixtures("empty_blend")
    @pytest.mark.parametrize("fmt", get_args(ImageFormat.__value__))
    def test_the_extension_matches_the_name_render_files_under(
        self, fmt: ImageFormat
    ) -> None:
        """`render` composes its return paths from the format, not from Blender."""
        settings = bpy.context.scene.render
        settings.image_settings.file_format = FORMATS[fmt][0]

        assert settings.file_extension == f".{fmt}"

    def test_the_compensation_is_stops_over_auto_exposure(self) -> None:
        sc = built("eo", format="png", exposure_compensation_ev=1.0)

        gain = sc.compositing_node_group.nodes["exposure"]
        assert gain.inputs[0].default_value == pytest.approx(2 * MID_GREY)

    def test_radiance_skips_the_camera(self) -> None:
        sc = built("eo", format="exr")

        assert "exposure" not in sc.compositing_node_group.nodes


class TestIrAsPng:
    @pytest.fixture(scope="class")
    @classmethod
    def sc(cls) -> bpy.types.Scene:
        return built("ir", format="png")

    def test_ir_renders_float(self, sc: bpy.types.Scene) -> None:
        assert sc.render.image_settings.color_depth == "32"

    def test_ir_radiance_skips_the_camera(self, sc: bpy.types.Scene) -> None:
        assert "exposure" not in sc.compositing_node_group.nodes

    def test_the_active_camera_sets_the_resolution(self, sc: bpy.types.Scene) -> None:
        """Factory 1920x1080 otherwise; a camera of that size would pass regardless."""
        ir = load(OPEN_SEA).band_mounts("ir")[0].camera
        assert (ir.width_px, ir.height_px) != (1920, 1080)

        assert (sc.render.resolution_x, sc.render.resolution_y) == (
            ir.width_px,
            ir.height_px,
        )


class TestEachBand:
    """What a default build sets, whichever band."""

    @pytest.fixture(scope="class", params=get_args(Band.__value__))
    @classmethod
    def band(cls, request: pytest.FixtureRequest) -> Band:
        return request.param

    @pytest.fixture(scope="class")
    @classmethod
    def sc(cls, band: Band) -> bpy.types.Scene:
        return built(band)

    def test_no_band_is_denoised(self, sc: bpy.types.Scene) -> None:
        assert sc.cycles.use_denoising is False

    def test_ir_takes_every_sample(self, sc: bpy.types.Scene, band: Band) -> None:
        assert sc.cycles.use_adaptive_sampling is (band != "ir")

    def test_the_compositor_runs_where_cycles_does(self, sc: bpy.types.Scene) -> None:
        assert sc.render.compositor_device == sc.cycles.device

    def test_both_bands_render_in_cycles(self, sc: bpy.types.Scene) -> None:
        assert sc.render.engine == "CYCLES"

    def test_eevee_does_not_cap_reflections(self, sc: bpy.types.Scene) -> None:
        assert (sc.world.sun_threshold, sc.eevee.clamp_surface_indirect) == (0.0, 0.0)

    def test_the_active_camera_belongs_to_the_band_built(
        self, sc: bpy.types.Scene, band: Band
    ) -> None:
        """Opened on the rig's first camera, an IR build could render through EO optics
        against IR materials, with nothing to say so.
        """
        assert sc.camera.name == f"bow_{band}"


@pytest.mark.parametrize("width_px", [640, 3840])
def test_a_frame_renders_in_one_tile(width_px: int) -> None:
    """Past one tile, Cycles caches each to disk and reads the frame back."""
    scene.build(variant(cameras={"eo": {"width_px": width_px}}), "eo")
    assert bpy.context.scene.cycles.tile_size >= width_px


def test_a_relative_output_reaches_blender_absolute_and_must_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handed: list[str] = []

    def writes_nothing(**_: object) -> set[str]:
        handed.append(bpy.context.scene.render.filepath)
        return {"FINISHED"}

    # bpy.ops.render is rebuilt on every access, so patch the attribute that holds it.
    monkeypatch.setattr(bpy.ops, "render", SimpleNamespace(render=writes_nothing))
    monkeypatch.chdir(tmp_path)
    scenario = variant(outputs={"bands": ["eo"]})
    mount = scenario.band_mounts("eo")[0]

    with pytest.raises(RuntimeError, match=f"camera {mount.name}'s frame"):
        render.render(scenario, Path("out"))

    assert Path(handed[0]).is_absolute()
    assert Path(handed[0]).parent == tmp_path / "out"


@pytest.mark.parametrize(
    ("returns", "writes"),
    [({"FINISHED"}, None), ({"FINISHED"}, b""), ({"CANCELLED"}, b"partial")],
)
def test_an_unwritten_frame_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returns: set[str],
    writes: bytes | None,
) -> None:
    path = tmp_path / "bow.png"

    def blender(**_: object) -> set[str]:
        if writes is not None:
            path.write_bytes(writes)
        return returns

    monkeypatch.setattr(bpy.ops, "render", SimpleNamespace(render=blender))

    with pytest.raises(RuntimeError, match="frame 0 was not written"):
        render._render_still(path, "frame 0")


def _plane_seen_from_above(
    corners: Sequence[tuple[float, float, float]],
    resolution: tuple[int, int],
    camera_location: tuple[float, float, float],
    ortho_scale: float,
) -> bpy.types.Object:
    """A factory scene in Cycles: a plane on z = 0, looked straight down on by an
    orthographic camera.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = resolution
    # As a build leaves it. Factory settings put the compositor on the GPU, which a
    # machine without one aborts on.
    sc.render.compositor_device = "CPU"
    mesh = bpy.data.meshes.new("plane")
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new("plane", mesh)
    lens = bpy.data.cameras.new("lens")
    lens.type, lens.sensor_fit, lens.ortho_scale = "ORTHO", "HORIZONTAL", ortho_scale
    camera = bpy.data.objects.new("camera", lens)
    camera.location = camera_location
    for obj in (plane, camera):
        sc.collection.objects.link(obj)
    sc.camera = camera
    return plane


@pytest.mark.render
@pytest.mark.parametrize(("edge_m", "columns"), [(3.45, 3), (3.55, 4)])
def test_the_object_index_pass_samples_the_pixel_centre(
    tmp_path: Path, edge_m: float, columns: int
) -> None:
    """A plane's edge just either side of a column of centres: a sample anywhere
    else in the pixel filter would put it in a different column from row to row.
    """
    corners = [(-10, -10, 0), (edge_m, -10, 0), (edge_m, 74, 0), (-10, 74, 0)]
    plane = _plane_seen_from_above(corners, (8, 64), (4.0, 32.0, 10.0), 8.0)
    plane.pass_index = 1
    _compositor(None)
    render._index_output(tmp_path).file_name = "probe."
    bpy.context.scene.render.filepath = str(tmp_path / "frame")

    bpy.ops.render.render(write_still=True)

    index = np.rint(blend.read(tmp_path / "probe.index.exr")[..., 0])
    assert (index[:, :columns] == 1).all()
    assert (index[:, columns:] == 0).all()


@pytest.mark.render
@pytest.mark.parametrize(("sun", "world"), [(math.pi, 0.0), (0.0, 1.0)])
def test_a_sun_s_strength_is_irradiance_in_the_world_s_radiance(
    tmp_path: Path, sun: float, world: float
) -> None:
    corners = [(-1e3, -1e3, 0), (1e3, -1e3, 0), (1e3, 1e3, 0), (-1e3, 1e3, 0)]
    plane = _plane_seen_from_above(corners, (4, 4), (0.0, 0.0, 1.0), 1.0)
    material = bpy.data.materials.new("white")
    tree = material.node_tree
    diffuse = tree.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    tree.links.new(
        diffuse.outputs["BSDF"], tree.nodes["Material Output"].inputs["Surface"]
    )
    plane.data.materials.append(material)
    sc = bpy.context.scene
    sc.world = bpy.data.worlds.new("world")
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        *(world,) * 3,
        1.0,
    )
    light = bpy.data.lights.new("sun", "SUN")
    light.energy, light.angle = sun, 0.0
    sc.collection.objects.link(bpy.data.objects.new("sun", light))
    sc.cycles.samples, sc.cycles.use_denoising = 16, False
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.filepath = str(tmp_path / "frame.exr")

    bpy.ops.render.render(write_still=True)

    assert blend.read(tmp_path / "frame.exr")[..., :3] == pytest.approx(1.0, rel=1e-3)


UNSEEN = sea.Seen()


def _svm_errors(
    capfd: pytest.CaptureFixture[str],
    scenario: Scenario,
    trails: tuple[Wake, ...],
    band: Band,
    pad: int = 0,
    seen: sea.Seen = UNSEEN,
) -> list[str]:
    """The SVM lines in Cycles' log after rendering `sea.water` alone.

    `pad` wraps the surface in that many Mix Shaders, each holding a factor on the SVM
    stack.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    set_output(scenario.outputs, band, 16)
    sc = bpy.context.scene
    sc.world = add_world(scenario.sky, band)
    seed = scenario.seed
    # Unculled: culling drops waves, which would shrink the shader under test.
    material = sea.material(
        scenario.sea,
        scenario.wind_waves(),
        scenario.swell_waves(),
        scenario.outputs,
        (
            substream(seed, "sea/gust"),
            substream(seed, "sea/slick"),
            substream(seed, "sea/foam"),
        ),
        trails,
        sea_light(scenario.sky, band),
        seen,
    )
    sea.water(scenario.sea, 300.0, material)
    add_haze(scenario.sky, band, 300.0)
    tree = material.node_tree
    output = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    surface = output.inputs["Surface"].links[0].from_socket
    x = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(
        tree.nodes.new("ShaderNodeNewGeometry").outputs["Position"], x.inputs[0]
    )
    for i in range(pad):
        # Not a constant, or Cycles folds the Mix away.
        factor = tree.nodes.new("ShaderNodeMath")
        factor.operation = "MULTIPLY"
        tree.links.new(x.outputs["X"], factor.inputs[0])
        factor.inputs[1].default_value = 1e-12 * (i + 1)
        mix = tree.nodes.new("ShaderNodeMixShader")
        tree.links.new(factor.outputs[0], mix.inputs[0])
        tree.links.new(surface, mix.inputs[1])
        tree.links.new(
            tree.nodes.new("ShaderNodeBsdfDiffuse").outputs["BSDF"], mix.inputs[2]
        )
        surface = mix.outputs[0]
    tree.links.new(surface, output.inputs["Surface"])
    probe(at=(0.0, -50.0, 10.0), pitch_deg=85.0)
    sc.render.resolution_x = sc.render.resolution_y = 16
    return _render_svm_errors(capfd)


def _render_svm_errors(capfd: pytest.CaptureFixture[str]) -> list[str]:
    capfd.readouterr()
    bpy.ops.render.render()
    log = capfd.readouterr()
    return [line for line in (log.out + log.err).splitlines() if "SVM" in line]


def _heaviest_drawn_sea(*, strongest: bool = False, **sections: Any) -> Scenario:
    """The `randomized.toml` draw that builds the largest sea shader: at its lightest
    wind, which draws the most waves, or its strongest, whose foam gathers in rows.
    """
    with (SCENARIOS / "randomized.toml").open("rb") as file:
        drawn = tomllib.load(file)["sea"]
    lightest, strongest_mps = drawn["wind_speed_mps"]["uniform"]
    return variant(
        OPEN_SEA,
        sea={
            "wind_speed_mps": strongest_mps if strongest else lightest,
            "fetch_km": drawn["fetch_km"]["log_uniform"][0],
            "swell": {
                "height_m": drawn["swell"]["height_m"]["uniform"][1],
                "period_s": drawn["swell"]["period_s"]["uniform"][0],
            },
            "slick_cover": drawn["slick_cover"]["uniform"][1],
        },
        sky={"visibility_km": None, **sections.pop("sky", {})},
        **sections,
    )


def _wakes(hulls: int) -> tuple[Wake, ...]:
    """Fast enough that their Kelvin arms draw, each its own: Cycles merges identical
    nodes, so copies of one wake hold the stack of one.
    """
    return tuple(
        Wake(
            speed_mps=6.0 + 0.1 * i,
            length_m=60.0,
            beam_m=10.0,
            start_m=(50.0 * i, 2000.0),
        )
        for i in range(hulls)
    )


CROWD = 64


# The heaviest sky: a photo, whose clouds the IR sea reflects, in mist, whose haze takes
# off the photo's sun disc. The stack is the shader's, so the photo is white.
MISTED_PHOTO = {"hdri": "rocky_ridge", "visibility_km": 1.2}


def _heaviest_haze(band: Band) -> None:
    nodes = bpy.data.node_groups["haze"].nodes
    assert (band == "eo") == ("photo_disc" in nodes)


@pytest.mark.render
@pytest.mark.parametrize("band", ["eo", "ir"])
@pytest.mark.parametrize("hulls", [0, 1, CROWD])
def test_the_heaviest_drawn_sea_fits_cycles_svm_stack(
    capfd: pytest.CaptureFixture[str],
    tmp_path: Path,
    band: Band,
    hulls: int,
) -> None:
    """Cycles renders a shader past its SVM stack black and logs an error."""
    drawn = _heaviest_drawn_sea(sky=MISTED_PHOTO)
    with white_photo(tmp_path):
        errors = _svm_errors(capfd, drawn, _wakes(hulls), band)
    assert errors == []
    _heaviest_haze(band)


@pytest.mark.render
@pytest.mark.parametrize("band", ["eo", "ir"])
def test_the_heaviest_drawn_whitecaps_fit_cycles_svm_stack(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    band: Band,
) -> None:
    """Remembered on every grid level a camera can need and gathered into windrows.
    The stack is the shader's, whatever its frames hold, so none are simulated.
    """
    monkeypatch.setattr(memory, "_simulate", lambda *_: None)
    grids = tuple(
        waves.FoamGrid(-8 * cell_m, 0.0, cell_m, 16, 16) for cell_m in (0.25, 1.0, 4.0)
    )
    seen = sea.Seen(foam_grids=grids)
    drawn = _heaviest_drawn_sea(strongest=True, sky=MISTED_PHOTO)
    with white_photo(tmp_path):
        errors = _svm_errors(capfd, drawn, _wakes(1), band, seen=seen)
    assert errors == []
    _heaviest_haze(band)
    nodes = bpy.data.materials["sea"].node_tree.nodes
    assert {"whitecap_gathered", "sea_foam_0.25m", "sea_foam_4m"} <= set(nodes.keys())
    assert (band == "ir") == ("sea_cloud_ratio" in nodes)


@pytest.mark.render
@pytest.mark.parametrize(
    "hull",
    [{"speed_mps": 3.0}, {"orbit": {"period_s": 80.0, "count": 2}}],
    ids=["line", "orbit"],
)
def test_the_wakes_a_build_lays_fit_cycles_svm_stack(
    capfd: pytest.CaptureFixture[str], hull: dict[str, Any]
) -> None:
    """The wake a built hull leaves, not one handed to the material."""
    scenario = _heaviest_drawn_sea(objects=[target("coaster", 500.0, **hull)])
    built = scene.build(scenario, "eo")
    mount = scenario.band_mounts("eo")[0]
    sc = bpy.context.scene
    sc.camera = built.cameras[mount.name]
    sc.cycles.samples = 1
    sc.render.resolution_x = sc.render.resolution_y = 16
    assert _render_svm_errors(capfd) == []


@pytest.mark.render
def test_the_svm_check_sees_an_overflow(capfd: pytest.CaptureFixture[str]) -> None:
    """A reworded Cycles log line would pass the test above on any shader."""
    assert _svm_errors(capfd, _heaviest_drawn_sea(), _wakes(0), "eo", pad=64)


@pytest.mark.render
def test_a_render_culls_only_waves_no_pixel_of_it_draws(tmp_path: Path) -> None:
    """High enough that the culled waves are many."""
    scenario = variant(
        rig={"height_m": 100.0},
        cameras={"eo": {"width_px": 256, "height_px": 192}},
        sky={"visibility_km": None},
        outputs={"format": "exr", "bands": ["eo"], "samples": {"eo": 4}},
    )
    (culled,) = (p for p in render.render(scenario, tmp_path) if p.suffix == ".exr")

    scene.build(scenario, "eo")
    sc = bpy.context.scene
    sc.render.filepath = str(tmp_path / "whole")
    bpy.ops.render.render(write_still=True)

    whole = blend.read(tmp_path / "whole.exr")[..., :3]
    drawn = blend.read(culled)[..., :3]
    assert np.abs(drawn - whole).mean() / whole.mean() < 1e-4


@pytest.mark.render
def test_each_box_holds_its_hull_centre_through_the_calibration(
    tmp_path: Path,
) -> None:
    """Boxes are sampled at pixel centres, so a hull reaches half a pixel past one."""
    # Boxes come from the index pass, which takes the pixel centre at any sample count.
    # A rolled ship and a pitched rig, so no axis of the calibration is identity.
    scenario = variant(
        ownship={"roll_deg": 2.0, "pitch_deg": 0.5},
        rig={"pitch_deg": -3.0},
        objects=[target("yacht", 300.0, -9.0), target("yacht", 600.0, 7.0)],
        outputs={"bands": ["ir"], "samples": {"ir": 1}},
    )

    render.render(scenario, tmp_path)

    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    cameras = {c.name: c for c in Calibration.read(tmp_path).cameras}
    frames = {image.id: cameras[image.camera] for image in truth.images}
    assert truth.annotations, "no target in any frame: nothing below ran"
    for found in truth.annotations:
        camera = frames[found.image_id]
        centre = bpy.data.objects[found.name].matrix_world.translation
        x, y, z, _ = np.linalg.inv(camera.extrinsics["world"]) @ (*centre, 1.0)
        u, v, _ = np.array(camera.K) @ (x, y, z) / z + 0.5  # COCO pixels
        left, top, width, height = found.bbox
        assert left - 0.5 <= u <= left + width + 0.5, found.name
        assert top - 0.5 <= v <= top + height + 0.5, found.name
        assert found.waterline_range_m < found.range_m


@pytest.mark.render
def test_a_fisheye_draws_each_hull_where_its_lens_puts_it(tmp_path: Path) -> None:
    """Both axes: Cycles' polynomial lens draws the scene half a turn about the optical
    axis, and the build turns the image and its index pass back.
    """
    k = (-0.0793, -0.0461, 0.0331, -0.0065)
    scenario = variant(
        cameras={
            "eo": {
                "hfov_deg": 132.93,
                "width_px": 640,
                "height_px": 360,
                "lens": {"fisheye": list(k)},
            }
        },
        objects=[target("yacht", 300.0, -40.0), target("yacht", 300.0, 35.0)],
        outputs={"bands": ["eo"], "samples": {"eo": 2}},
    )

    render.render(scenario, tmp_path)

    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    (camera,) = Calibration.read(tmp_path).cameras
    assert len(truth.annotations) == 2
    for found in truth.annotations:
        centre = bpy.data.objects[found.name].matrix_world.translation
        seen = np.linalg.inv(camera.extrinsics["world"]) @ (*centre, 1.0)
        zero = np.zeros(3)
        image, _ = cv2.fisheye.projectPoints(
            seen[None, None, :3], zero, zero, np.array(camera.K), np.array(k)
        )
        u, v = image.ravel() + 0.5  # COCO pixels
        left, top, width, height = found.bbox
        assert left - 0.5 <= u <= left + width + 0.5, found.name
        assert top - 0.5 <= v <= top + height + 0.5, found.name
    frame = cv2.imread(str(tmp_path / truth.images[0].file_name))
    assert frame is not None
    assert frame[:60].mean() > frame[-60:].mean(), "the sky is above the sea"


@pytest.mark.render
def test_a_crop_records_the_cameras_frame_less_its_edges(tmp_path: Path) -> None:
    """Uneven edges, so a crop measured from the wrong side or flipped shows."""
    crop = {"left_px": 24, "right_px": 8, "top_px": 6, "bottom_px": 10}
    scenario = variant(
        rig={
            "videos": {
                "cut": {"kind": "crop", "camera": "eo"} | crop,
                "thermal": {"kind": "identity", "camera": "ir"},
            }
        },
        cameras={"eo": {"width_px": 160, "height_px": 90}},
        objects=[target("yacht", 300.0, -9.0)],
        outputs={"bands": ["eo"], "format": "png", "samples": {"eo": 2}},
    )

    render.render(scenario, tmp_path)

    whole = cv2.imread(str(tmp_path / "bow_eo.png"))
    cut = cv2.imread(str(tmp_path / "bow_cut.png"))
    assert whole is not None and cut is not None
    # Blender dithers each 8-bit save on its own.
    assert np.abs(cut - whole[6:-10, 24:-8].astype(int)).max() <= 2
    (camera,) = Calibration.read(tmp_path).cameras
    (source,) = Calibration.read(tmp_path).raw
    assert (camera.name, camera.width_px, camera.height_px) == ("bow_cut", 128, 74)
    assert np.subtract(source.K, camera.K)[:2, 2] == pytest.approx((24.0, 6.0))
    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    (found,) = truth.annotations
    centre = bpy.data.objects[found.name].matrix_world.translation
    x, y, z, _ = np.linalg.inv(camera.extrinsics["world"]) @ (*centre, 1.0)
    u, v, _ = np.array(camera.K) @ (x, y, z) / z + 0.5  # COCO pixels
    left, top, width, height = found.bbox
    assert left - 0.5 <= u <= left + width + 0.5
    assert top - 0.5 <= v <= top + height + 0.5


@pytest.mark.render
def test_an_undistortion_renders_its_pinhole_and_records_the_fisheye(
    tmp_path: Path,
) -> None:
    undistortion = {"hfov_deg": 130.0, "vfov_deg": 65.5, "downscale": 2}
    scenario = variant(
        rig={
            "videos": {
                "RGB1": {"kind": "undistortion", "camera": "eo"} | undistortion,
                "T1": {"kind": "identity", "camera": "ir"},
            }
        },
        cameras={
            "eo": {
                "hfov_deg": 136.0,
                "width_px": 640,
                "height_px": 360,
                "lens": {"fisheye": [0.0, 0.0, 0.0, 0.0]},
            }
        },
        objects=[target("yacht", 300.0, -9.0)],
        outputs={"bands": ["eo"], "samples": {"eo": 2}},
    )

    render.render(scenario, tmp_path)

    (camera,) = Calibration.read(tmp_path).cameras
    (source,) = Calibration.read(tmp_path).raw
    frame = cv2.imread(str(tmp_path / camera.image))
    assert frame is not None and frame.shape[:2] == (180, 600)
    assert (camera.name, camera.distortion) == ("bow_RGB1", None)
    assert camera.hfov_deg == pytest.approx(130.0)
    assert (source.width_px, source.height_px) == (640, 360)
    assert source.distortion == (0.0, 0.0, 0.0, 0.0)
    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    (found,) = truth.annotations
    centre = bpy.data.objects[found.name].matrix_world.translation
    x, y, z, _ = np.linalg.inv(camera.extrinsics["world"]) @ (*centre, 1.0)
    u, v, _ = np.array(camera.K) @ (x, y, z) / z + 0.5  # COCO pixels
    left, top, width, height = found.bbox
    assert left - 0.5 <= u <= left + width + 0.5
    assert top - 0.5 <= v <= top + height + 0.5


@pytest.mark.render
def test_a_stitch_labels_one_frame_of_both_cameras(tmp_path: Path) -> None:
    ir = load(OPEN_SEA).rigs["bow"].cameras["ir"]
    small = {"width_px": 80, "height_px": 64}
    scenario = variant(
        rig={
            "videos": {
                "E": {"kind": "identity", "camera": "eo"},
                "T1": {"kind": "horizontal_stitching", "cameras": ["ir", "ir_r"]},
            }
        },
        cameras={
            "eo": {"width_px": 96, "height_px": 54},
            "ir": small | {"yaw_deg": -11.4},
            "ir_r": ir.model_copy(update=small | {"yaw_deg": 11.4}),
        },
        objects=[target("yacht", 300.0, 0.0)],
        # eo too: its frames have no stitch to make.
        outputs={"bands": ["eo", "ir"], "samples": {"eo": 1, "ir": 4}},
    )

    render.render(scenario, tmp_path)

    calibration = Calibration.read(tmp_path)
    (camera,) = (c for c in calibration.cameras if c.band == "ir")
    assert camera.name == "bow_T1"
    assert {c.name for c in calibration.raw} == {"bow_eo", "bow_ir", "bow_ir_r"}
    frame = cv2.imread(str(tmp_path / camera.image), cv2.IMREAD_UNCHANGED)
    assert frame is not None and frame.shape == (camera.height_px, camera.width_px)
    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    (image,) = (i for i in truth.images if i.camera == "bow_T1")
    (found,) = (a for a in truth.annotations if a.image_id == image.id)
    centre = bpy.data.objects[found.name].matrix_world.translation
    x, y, z, _ = np.linalg.inv(camera.extrinsics["world"]) @ (*centre, 1.0)
    u, v, _ = np.array(camera.K) @ (x, y, z) / z + 0.5  # COCO pixels
    left, top, width, height = found.bbox
    assert left - 0.5 <= u <= left + width + 0.5
    assert top - 0.5 <= v <= top + height + 0.5


@pytest.mark.render
def test_a_sequence_writes_each_camera_a_folder_of_frames(tmp_path: Path) -> None:
    """Fast enough that the ship crosses pixels between frames."""
    scenario = variant(
        UNDERWAY,
        cameras={
            "eo": {"width_px": 96, "height_px": 54},
            "ir": {"width_px": 80, "height_px": 64},
        },
        objects=[target("yacht", 300.0, 8.0, heading_deg=270.0, speed_mps=10.0)],
        outputs={"duration_s": 3.0, "fps": 1, "samples": {"eo": 2, "ir": 4}},
    )

    render.render(scenario, tmp_path)

    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    for mount in scenario.mounts:
        frames = [image for image in truth.images if image.camera == mount.name]
        assert [image.file_name for image in frames] == [
            f"{mount.name}/{f:04d}.jpg" for f in range(3)
        ]
        assert all((tmp_path / image.file_name).exists() for image in frames)
        assert [image.time_s for image in frames] == [0.0, 1.0, 2.0]
        left = [
            next(a.bbox[0] for a in truth.annotations if a.image_id == image.id)
            for image in frames
        ]
        # Heading west, across a camera facing north.
        assert left == sorted(left, reverse=True), mount.name
        assert left[0] > left[-1], mount.name
    assert not list(tmp_path.rglob("*.exr"))


@pytest.mark.render
def test_a_render_that_dies_keeps_the_truth_of_every_frame_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = variant(
        cameras={"ir": {"width_px": 80, "height_px": 64}},
        outputs={"duration_s": 3.0, "fps": 1, "bands": ["ir"], "samples": {"ir": 2}},
    )
    real, calls = bpy.ops.render.render, []

    def dies_on_the_third(**kwargs: object) -> set[str]:
        calls.append(None)
        if len(calls) == 3:
            raise RuntimeError("killed")
        return real(**kwargs)

    monkeypatch.setattr(bpy.ops, "render", SimpleNamespace(render=dies_on_the_third))
    with pytest.raises(RuntimeError, match="killed"):
        render.render(scenario, tmp_path)

    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    assert [image.time_s for image in truth.images] == [0.0, 1.0]
    assert truth.info["scenario"] == scenario.model_dump(mode="json")
    named = [image.file_name for image in truth.images]
    assert [Path(name).suffix for name in named] == [".jpg", ".jpg"]
    assert all((tmp_path / name).exists() for name in named)
    assert [c.image for c in Calibration.read(tmp_path).cameras] == named


@pytest.mark.render
@pytest.mark.parametrize(("fmt", "boxes"), [("jpg", 1), ("png", 2)])
def test_only_an_lwir_jpg_drops_a_hull_lost_in_the_sea(
    tmp_path: Path, fmt: str, boxes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 16-bit png keeps every box, as an exr does."""
    # Between the two hulls' excess: the test is which format is judged, not the cut.
    monkeypatch.setattr(labels, "MIN_EXCESS_DE", 10.0)
    scenario = variant(
        cameras={"ir": {"width_px": 160, "height_px": 128}},
        # The far hull reads nearly as warm as the sea behind it.
        objects=[target("yacht", 300.0), target("yacht", 2000.0, 4.0, t_k=274.0)],
        sky={"solar_gain_k": 0.0},
        outputs={"bands": ["ir"], "samples": {"ir": 4}, "format": fmt},
    )

    render.render(scenario, tmp_path)

    truth = labels.Labels.model_validate_json((tmp_path / labels.FILENAME).read_text())
    assert len(truth.annotations) == boxes


def test_an_install_without_git_names_the_image_s_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(render, "__file__", str(tmp_path / "__init__.py"))
    monkeypatch.setenv("SEASCAPE_COMMIT", "abc123")
    assert render._commit() == "abc123"
    monkeypatch.setenv("SEASCAPE_COMMIT", "")
    assert render._commit() is None


@pytest.mark.render
@pytest.mark.parametrize("fmt", get_args(ImageFormat.__value__))
def test_no_image_outgrows_what_the_server_reserves_for_it(
    tmp_path: Path, fmt: ImageFormat
) -> None:
    """At 1 sample, the noisiest, which a jpg compresses worst. Smaller than the
    baseline's frames: a pixel holds more of the scene and compresses no better.
    """
    scenario = variant(
        BASELINE,
        cameras={
            "eo": {"width_px": 960, "height_px": 540},
            "ir": {"width_px": 320, "height_px": 256},
        },
        outputs={"format": fmt, "samples": {"eo": 1, "ir": 1}},
    )

    render.render(scenario, tmp_path)

    for mount in scenario.mounts:
        camera = mount.camera
        reserved = camera.width_px * camera.height_px
        reserved *= jobs.BYTES_PER_PX[camera.band, fmt]
        assert (tmp_path / f"{mount.name}.{fmt}").stat().st_size <= reserved


def _plate(
    centre: tuple[float, float, float], normal: tuple[float, float, float], size: float
) -> list[tuple[float, float, float]]:
    """A square's corners, counter-clockwise seen from `normal`."""
    n = np.array(normal) / np.linalg.norm(normal)
    u = np.cross(n, (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (1.0, 0.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c = np.array(centre)
    h = size / 2
    return [
        tuple(c + h * (a * u + b * v)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def _radiance_scene(sky_k: float, width: int, height: int) -> bpy.types.Scene:
    """An empty Cycles scene that writes radiance to a float EXR, under a sky of one
    brightness temperature.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 16
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = width, height
    sc.render.compositor_device = "CPU"
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_depth = "32"
    world = bpy.data.worlds.new("sky")
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        *(lwir.band_radiance(sky_k),) * 3,
        1.0,
    )
    sc.world = world
    return sc


@pytest.mark.render
@pytest.mark.parametrize("rise_m", [0.0, 2.0])
def test_a_zone_renders_at_its_heat_balance(tmp_path: Path, rise_m: float) -> None:
    """Under a uniform sky, each plate reads eps B(T) + (1 - eps) B(sky), T as
    `thermal` gives it for that normal and that sunlight; below the waterline, the
    water's. A hull raised `rise_m` meets the water that far down its own frame.
    """
    sky_k = 270.0
    sc = _radiance_scene(sky_k, 96, 64)

    elevation, bearing = math.radians(30.0), math.radians(120.0)
    sun = (
        math.cos(elevation) * math.sin(bearing),
        math.cos(elevation) * math.cos(bearing),
        math.sin(elevation),
    )
    env = thermal.Environment(
        t_air_k=288.0, t_sea_k=287.5, wind_m_s=3.0, sun_elevation_rad=elevation
    )
    heat = Heat(env, 286.0, sun)
    tilted = (0.5 * sun[0], 0.5 * sun[1], 1.0)
    plates = {
        "lit": ((-6.0, 0.0, 2.0), (0.0, 0.0, 1.0)),
        "tilted": ((6.0, 0.0, 2.0), tilted),
        "low": ((0.0, 8.0, -1.0), (0.0, 0.0, 1.0)),
    }
    corners = [c for centre, n in plates.values() for c in _plate(centre, n, 4.0)]
    mesh = bpy.data.meshes.new("hull")
    quads = [tuple(range(4 * i, 4 * i + 4)) for i in range(len(plates))]
    mesh.from_pydata(corners, [], quads)
    hull = bpy.data.objects.new("hull", mesh)
    hull["rise_m"] = rise_m
    sc.collection.objects.link(hull)
    hull.data.materials.append(zone_skin("paint_hull", heat))

    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.angle = math.radians(40.0)
    sc.collection.objects.link(camera)
    camera.location = (-4.0, 12.0, 40.0)
    direction = Vector((0.0, 3.0, 0.0)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    sc.camera = camera
    sc.render.filepath = str(tmp_path / "frame.exr")
    bpy.ops.render.render(write_still=True)
    radiance = blend.read(tmp_path / "frame.exr")[..., 0]

    zone = thermal.ZONES["paint_hull"]
    s = np.array(sun)
    for name, (centre, normal) in plates.items():
        u, v, _ = world_to_camera_view(sc, camera, Vector(centre))
        assert 0.1 < u < 0.9 and 0.1 < v < 0.9, name
        seen = radiance[int(v * 64), int(u * 96)]
        n = np.array(normal) / np.linalg.norm(normal)
        if centre[2] + rise_m < 0.0:
            emissivity, expected = thermal.SEA_EMISSIVITY, heat.t_water_k
        else:
            emissivity = zone.emissivity
            lit = max(0.0, float(n @ s))
            expected = float(thermal.lit_temperature_k(zone, env, lit, n[2]))
        emitted = (seen - (1.0 - emissivity) * lwir.band_radiance(sky_k)) / emissivity
        assert float(lwir.brightness_temperature(emitted)) == pytest.approx(
            expected, abs=0.3
        ), name


@pytest.mark.render
def test_a_hull_warms_the_calm_sea_below_it(tmp_path: Path) -> None:
    """The sea's lobe meets the hull, warmer than the low sky it hides. Hidden from
    glossy rays, the hull leaves the sea as its tables have it.
    """
    scenario = variant(
        rig={"height_m": 15.0, "pitch_deg": -2.0},
        cameras={"ir": {"hfov_deg": 4.0, "width_px": 96, "height_px": 64}},
        objects=[
            {
                "asset": "cruising_sloop",
                "range_m": 450.0,
                "bearing_deg": 0.0,
                "heading_deg": 90.0,
            }
        ],
        sea={"wind_speed_mps": 3.0},
    )
    built = scene.build(scenario, "ir")
    sc = bpy.context.scene
    sc.camera = built.cameras["bow_ir"]
    sc.render.resolution_x, sc.render.resolution_y = 96, 64
    sc.cycles.samples = 64
    sc.cycles.use_adaptive_sampling = False

    def kelvin(name: str) -> np.ndarray:
        sc.render.filepath = str(tmp_path / f"{name}.exr")
        bpy.ops.render.render(write_still=True)
        return lwir.brightness_temperature(blend.read(tmp_path / f"{name}.exr")[..., 0])

    seen = kelvin("seen")
    for hull in chain.from_iterable(built.targets.values()):
        for part in (hull, *hull.children_recursive):
            part.visible_glossy = False
    hidden = kelvin("hidden")
    warmer = seen - hidden
    assert warmer.max() > 1.0
    assert warmer.mean() > 0.0
