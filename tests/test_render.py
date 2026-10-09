"""Render settings, the thermal image, and the object-index pass."""

from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import bpy
import cv2
import numpy as np
import pytest

from seascape import agc, labels, lwir, render, scene
from seascape.calibration import Calibration
from seascape.config import Band, ImageFormat, load
from tests.scenarios import OPEN_SEA, TWIN_POD, UNDERWAY, target, variant


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


class TestThermalImage:
    """An inverted, flipped or sRGB-encoded frame is still a plausible-looking picture,
    so only the numbers catch it.
    """

    def test_a_png_is_centikelvin_with_the_top_row_first(self, tmp_path: Path) -> None:
        exr = exr_of(tmp_path, [lwir.band_radiance(t) for t in (272.0, 295.0)])
        render._write_thermal(exr, "png", agc.Agc())
        t_k = agc.kelvin(exr.with_suffix(".png"))
        assert t_k is not None
        assert t_k[:, 0] == pytest.approx([295.0, 272.0], abs=0.01)

    def test_a_jpg_is_grey_through_the_agc(self, tmp_path: Path) -> None:
        exr = exr_of(tmp_path, [lwir.band_radiance(t) for t in (270.0, 285.0, 300.0)])
        render._write_thermal(exr, "jpg", agc.Agc())
        jpg = cv2.imread(str(exr.with_suffix(".jpg")), cv2.IMREAD_GRAYSCALE)
        assert jpg is not None
        assert jpg[:, 0].astype(int) == pytest.approx([255, 128, 0], abs=3)

    def test_the_float_render_is_removed_on_success(self, tmp_path: Path) -> None:
        exr = exr_of(tmp_path, [lwir.band_radiance(285.0), lwir.band_radiance(295.0)])
        render._write_thermal(exr, "png", agc.Agc())
        assert not exr.exists()

    def test_a_failure_keeps_the_float_render_and_leaks_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed conversion must not throw the render away."""
        exr = exr_of(tmp_path, [lwir.band_radiance(285.0)])
        before = len(bpy.data.images)
        monkeypatch.setattr(render.lwir, "brightness_temperature", _raise)
        with pytest.raises(RuntimeError):
            render._write_thermal(exr, "png", agc.Agc())
        assert exr.exists()
        assert len(bpy.data.images) == before


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
        assert set(scene.FORMATS) == set(get_args(ImageFormat.__value__))

    @pytest.mark.parametrize("fmt", get_args(ImageFormat.__value__))
    def test_the_extension_matches_the_name_render_files_under(
        self, fmt: ImageFormat
    ) -> None:
        """`render` composes its return paths from the format, not from Blender."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        settings = bpy.context.scene.render
        settings.image_settings.file_format = scene.FORMATS[fmt][0]

        assert settings.file_extension == f".{fmt}"

    def test_the_compensation_is_stops_over_auto_exposure(self) -> None:
        sc = built("eo", format="png", exposure_compensation_ev=1.0)

        gain = sc.compositing_node_group.nodes["exposure"]
        assert gain.inputs[0].default_value == pytest.approx(2 * scene.MID_GREY)

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


def test_a_relative_output_reaches_blender_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handed: list[str] = []

    def capture(**_: object) -> None:
        handed.append(bpy.context.scene.render.filepath)
        raise RuntimeError("captured")

    # bpy.ops.render is rebuilt on every access, so patch the attribute that holds it.
    monkeypatch.setattr(bpy.ops, "render", SimpleNamespace(render=capture))
    monkeypatch.chdir(tmp_path)
    scenario = variant(outputs={"bands": ["eo"]})

    with pytest.raises(RuntimeError, match="captured"):
        render.render(scenario, Path("out"))

    assert Path(handed[0]).is_absolute()
    assert Path(handed[0]).parent == tmp_path / "out"


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
    scene._compositor(None)  # as a build leaves it
    render._index_output(tmp_path).file_name = "probe."
    sc.render.filepath = str(tmp_path / "frame")

    bpy.ops.render.render(write_still=True)

    index = np.rint(render._pixels(tmp_path / "probe.index.exr")[..., 0])
    assert (index[:, :columns] == 1).all()
    assert (index[:, columns:] == 0).all()


@pytest.mark.render
def test_each_box_holds_its_hull_centre_through_the_calibration(
    tmp_path: Path,
) -> None:
    """Boxes are sampled at pixel centres, so a hull reaches half a pixel past one."""
    # Boxes come from the index pass, which takes the pixel centre at any sample count.
    scenario = variant(TWIN_POD, outputs={"bands": ["ir"], "samples": {"ir": 1}})

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

    def dies_on_the_third(**kwargs: object) -> None:
        calls.append(None)
        if len(calls) == 3:
            raise RuntimeError("killed")
        real(**kwargs)

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
