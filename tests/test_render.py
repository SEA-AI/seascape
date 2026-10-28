"""Render settings, the thermal image, and the object-index pass."""

import math
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import bpy
import cv2
import numpy as np
import pytest
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from seascape import render, server
from seascape.blender import blend, scene, sea
from seascape.blender.compositor import (
    FORMATS,
    MID_GREY,
    _compositor,
    respond,
    set_output,
)
from seascape.blender.heat import Heat, zone_skin
from seascape.blender.sea import memory
from seascape.blender.sky import add_world, sea_light
from seascape.config import (
    Agc,
    Band,
    Camera,
    ImageFormat,
    Microbolometer,
    Scenario,
    load,
    substream,
)
from seascape.outputs import agc, labels
from seascape.outputs.calibration import Calibration
from seascape.physics import lwir, thermal, waves
from seascape.physics.wakes import Wake
from tests.scenarios import BASELINE, OPEN_SEA, SCENARIOS, UNDERWAY, target, variant

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
    mount = next(m for m in scenario.mounts if m.camera.band == "ir")
    camera = mount.camera.model_copy(
        update={"microbolometer": Microbolometer(**bolometer)}
    )
    respond(mount._replace(camera=camera), substream(0, "sensor"))
    assert bpy.context.scene.cycles.pixel_filter_type == kind


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

    @pytest.mark.parametrize("fmt", get_args(ImageFormat.__value__))
    def test_the_extension_matches_the_name_render_files_under(
        self, fmt: ImageFormat
    ) -> None:
        """`render` composes its return paths from the format, not from Blender."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
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
        ir = next(m.camera for m in load(OPEN_SEA).mounts if m.camera.band == "ir")
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

    def test_the_compositor_runs_on_the_cpu(self, sc: bpy.types.Scene) -> None:
        assert sc.render.compositor_device == "CPU"

    def test_both_bands_render_in_cycles(self, sc: bpy.types.Scene) -> None:
        assert sc.render.engine == "CYCLES"

    def test_eevee_does_not_cap_reflections(self, sc: bpy.types.Scene) -> None:
        assert (sc.world.sun_threshold, sc.eevee.clamp_surface_indirect) == (0.0, 0.0)

    def test_the_active_camera_belongs_to_the_band_built(
        self, sc: bpy.types.Scene, band: Band
    ) -> None:
        """Opened on the rig's first camera, an IR build could render through EO optics
        against IR materials, with nothing to say so."""
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
    mount = next(m for m in scenario.mounts if m.camera.band == "eo")

    with pytest.raises(RuntimeError, match=f"frame 0 of camera {mount.name}"):
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


@pytest.mark.render
@pytest.mark.parametrize(("edge_m", "columns"), [(3.45, 3), (3.55, 4)])
def test_the_object_index_pass_samples_the_pixel_centre(
    tmp_path: Path, edge_m: float, columns: int
) -> None:
    """A plane's edge just either side of a column of centres: a sample anywhere
    else in the pixel filter would put it in a different column from row to row."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = 8, 64
    mesh = bpy.data.meshes.new("plane")
    corners = [(-10, -10, 0), (edge_m, -10, 0), (edge_m, 74, 0), (-10, 74, 0)]
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    plane = bpy.data.objects.new("plane", mesh)
    plane.pass_index = 1
    lens = bpy.data.cameras.new("lens")
    lens.type, lens.sensor_fit, lens.ortho_scale = "ORTHO", "HORIZONTAL", 8.0
    camera = bpy.data.objects.new("camera", lens)
    camera.location = (4.0, 32.0, 10.0)
    for obj in (plane, camera):
        sc.collection.objects.link(obj)
    sc.camera = camera
    # As a build leaves it. Factory settings put the compositor on the GPU, which a
    # machine without one aborts on.
    sc.render.compositor_device = "CPU"
    _compositor(None)
    render._index_output(tmp_path).file_name = "probe."
    sc.render.filepath = str(tmp_path / "frame")

    bpy.ops.render.render(write_still=True)

    index = np.rint(blend.read(tmp_path / "probe.index.exr")[..., 0])
    assert (index[:, :columns] == 1).all()
    assert (index[:, columns:] == 0).all()


def _svm_errors(
    capfd: pytest.CaptureFixture[str],
    scenario: Scenario,
    trails: tuple[Wake, ...],
    band: Band,
    pad: int = 0,
    seen: sea.Seen = sea.UNSEEN,
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
    camera = bpy.data.objects.new("probe", bpy.data.cameras.new("probe"))
    sc.collection.objects.link(camera)
    camera.location = (0.0, -50.0, 10.0)
    camera.rotation_euler = (math.radians(85.0), 0.0, 0.0)
    sc.camera = camera
    sc.cycles.samples = 1
    sc.render.resolution_x = sc.render.resolution_y = 16
    return _render_svm_errors(capfd)


def _render_svm_errors(capfd: pytest.CaptureFixture[str]) -> list[str]:
    capfd.readouterr()
    bpy.ops.render.render()
    log = capfd.readouterr()
    return [line for line in (log.out + log.err).splitlines() if "SVM" in line]


def _heaviest_drawn_sea(*, strongest: bool = False, **sections: Any) -> Scenario:
    """The `randomized.toml` draw that builds the largest sea shader: at its lightest
    wind, which draws the most waves, or its strongest, whose foam gathers in rows."""
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
    return tuple(
        Wake(speed_mps=3.0, length_m=60.0, beam_m=10.0, start_m=(0.0, 2000.0))
        for _ in range(hulls)
    )


@pytest.mark.render
@pytest.mark.parametrize("band", ["eo", "ir"])
@pytest.mark.parametrize("hulls", [0, 1])
def test_the_heaviest_drawn_sea_fits_cycles_svm_stack(
    capfd: pytest.CaptureFixture[str], band: Band, hulls: int
) -> None:
    """Cycles renders a shader past its SVM stack black and logs an error."""
    errors = _svm_errors(capfd, _heaviest_drawn_sea(), _wakes(hulls), band)
    assert errors == []


@pytest.mark.render
@pytest.mark.parametrize("band", ["eo", "ir"])
def test_the_heaviest_drawn_whitecaps_fit_cycles_svm_stack(
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    band: Band,
) -> None:
    """Remembered on every grid level a camera can need, gathered into windrows, and
    in IR under a photographed sky, whose clouds the sea reflects. The stack is the
    shader's, whatever its frames hold, so none are simulated and the photo is
    white."""
    monkeypatch.setattr(memory, "_simulate", lambda *_: None)
    grids = tuple(
        waves.FoamGrid(-8 * cell_m, 0.0, cell_m, 16, 16) for cell_m in (0.25, 1.0, 4.0)
    )
    seen = sea.Seen(foam_grids=grids)
    sky: dict[str, str] = {}
    if band == "ir":
        photo = tmp_path / "white.hdr"
        cv2.imwrite(str(photo), np.ones((32, 64, 3), np.float32))
        monkeypatch.setattr("seascape.blender.sky.download", lambda *_: photo)
        sky = {"hdri": "overcast_soil"}
    drawn = _heaviest_drawn_sea(strongest=True, sky=sky)
    errors = _svm_errors(capfd, drawn, _wakes(1), band, seen=seen)
    assert errors == []
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
    mount = next(m for m in scenario.mounts if m.camera.band == "eo")
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
    baseline's frames: a pixel holds more of the scene and compresses no better."""
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
        reserved *= server.BYTES_PER_PX[camera.band, fmt]
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
    brightness temperature."""
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
def test_a_zone_renders_at_its_heat_balance(tmp_path: Path) -> None:
    """Under a uniform sky, each plate reads eps B(T) + (1 - eps) B(sky), T as
    `thermal` gives it for that normal and that sunlight; below the waterline, the
    water's."""
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
        "wetted": ((0.0, 8.0, -1.0), (0.0, 0.0, 1.0)),
    }
    corners = [c for centre, n in plates.values() for c in _plate(centre, n, 4.0)]
    mesh = bpy.data.meshes.new("hull")
    quads = [tuple(range(4 * i, 4 * i + 4)) for i in range(len(plates))]
    mesh.from_pydata(corners, [], quads)
    hull = bpy.data.objects.new("hull", mesh)
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
        if name == "wetted":
            emissivity, expected = thermal.SEA_EMISSIVITY, heat.t_water_k
        else:
            emissivity = zone.emissivity
            lit = max(0.0, float(n @ s))
            expected = float(thermal.lit_temperature_k(zone, env, lit, n[2]))
        emitted = (seen - (1.0 - emissivity) * lwir.band_radiance(sky_k)) / emissivity
        assert float(lwir.brightness_temperature(emitted)) == pytest.approx(
            expected, abs=0.3
        ), name
