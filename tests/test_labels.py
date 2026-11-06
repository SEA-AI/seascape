"""labels.json from an index map and a calibration, without a render."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from seascape.config import Agc, Camera, Lens, Sensor
from seascape.outputs import labels
from seascape.outputs.calibration import CameraCalibration
from seascape.physics import earth, fisheye

RADIUS_M = earth.earth_radius_m(0.13)
FLAT = np.ones((48, 64, 3))


def camera(
    width: int = 64,
    height: int = 48,
    hfov_deg: float = 49.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    at_m: tuple[float, float, float] = (0.0, 0.0, 50.0),
    distortion: fisheye.K | None = None,
) -> CameraCalibration:
    """Facing north, then pitched about its x and rolled about its z; a fisheye with
    `distortion`.
    """
    if distortion is None:
        f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
    else:
        f = fisheye.focal_px(math.radians(hfov_deg), width, distortion)
    level = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    pitch, _ = cv2.Rodrigues(np.array([math.radians(pitch_deg), 0.0, 0.0]))
    roll, _ = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(roll_deg)]))
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = level @ pitch @ roll, at_m
    return CameraCalibration(
        name="C",
        band="eo",
        image="C.png",
        rig="bow",
        width_px=width,
        height_px=height,
        K=((f, 0, (width - 1) / 2), (0, f, (height - 1) / 2), (0, 0, 1)),
        distortion=distortion,
        extrinsics={"world": tuple(map(tuple, pose))},
    )


def target(
    pass_index: int, category: str = "ship", category_id: int = 1
) -> labels.Target:
    return labels.Target(
        pass_index=pass_index,
        name=f"t{pass_index}",
        category_id=category_id,
        category=category,
        supercategory="vessel",
        centre_m=(0.0, 1000.0),
        waterline_m=np.array([[0.0, 990.0], [5.0, 1010.0]]),
        heading_deg=90.0,
        dims_m=(20.0, 5.0, 4.0),
    )


def added(
    index: np.ndarray | None = None,
    targets: Sequence[labels.Target] = (),
    *,
    cam: CameraCalibration | None = None,
    time_s: float = 0.0,
    frame: np.ndarray = FLAT,
    settings: Camera | None = None,
    faint: Mapping[str, float] | None = None,
) -> labels.Labels:
    """One frame's labels; an empty index map unless given."""
    truth = labels.Labels()
    truth.add(
        cam or camera(),
        time_s,
        np.zeros((48, 64), dtype=int) if index is None else index,
        frame,
        targets,
        RADIUS_M,
        settings,
        faint=faint or {},
    )
    return truth


def test_a_box_covers_whole_pixels_from_its_top_left_corner() -> None:
    """COCO's pixel i spans [i, i + 1), so an inclusive box grows by one."""
    index = np.zeros((48, 64), dtype=int)
    index[10:13, 20:25] = 1

    (found,) = added(index, [target(1)]).annotations

    assert found.bbox == (20, 10, 5, 3)
    assert found.area == 15
    assert not found.truncated


@pytest.mark.parametrize("corner", [(10, 20), (0, 0)])
def test_a_segmentation_decodes_to_the_pixels_it_was_cut_from(
    corner: tuple[int, int],
) -> None:
    """Two parts and a hole, with the mask starting on the first pixel or not."""
    index = np.zeros((48, 64), dtype=int)
    y, x = corner
    index[y : y + 6, x : x + 8] = 1
    index[y + 2 : y + 4, x + 3 : x + 5] = 0
    index[y + 20 : y + 22, x + 30 : x + 31] = 1

    (found,) = added(index, [target(1)]).annotations

    height, width = found.segmentation.size
    decoded = np.repeat(
        np.arange(len(found.segmentation.counts)) % 2 == 1, found.segmentation.counts
    )
    assert (height, width) == (48, 64)
    assert ((index == 1) == decoded.reshape(width, height).T).all()
    assert found.area == decoded.sum()


def test_a_box_on_the_frame_edge_is_truncated() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[40:, 60:] = 1

    assert added(index, [target(1)]).annotations[0].truncated


def test_only_a_target_in_frame_is_labelled_or_categorised() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1

    truth = added(index, [target(1), target(2, "buoy", 6)])

    assert [a.name for a in truth.annotations] == ["t1"]
    assert [c.name for c in truth.categories] == ["ship"]


def test_targets_of_one_category_share_it() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2

    truth = added(index, [target(1), target(2)])

    assert {a.category_id for a in truth.annotations} == {1}
    assert [(c.name, c.supercategory) for c in truth.categories] == [("ship", "vessel")]


def test_a_category_keeps_its_own_id_whatever_came_first() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2

    truth = added(index, [target(1, "buoy", 6), target(2)])

    assert [(c.id, c.name) for c in truth.categories] == [(6, "buoy"), (1, "ship")]
    assert [a.category_id for a in truth.annotations] == [6, 1]


def test_an_image_carries_its_cameras_field_of_view_and_its_time() -> None:
    (image,) = added(cam=camera(hfov_deg=42.0), time_s=2.5).images

    assert image.hfov_deg == pytest.approx(42.0)
    assert image.time_s == 2.5


def test_an_annotation_carries_its_targets_heading_dimensions_and_engine_stop() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    stopped = target(1)._replace(engine_off_min=45.0)

    (annotation,) = added(index, [stopped]).annotations

    assert (annotation.heading_deg, annotation.dims_m, annotation.engine_off_min) == (
        90.0,
        (20.0, 5.0, 4.0),
        45.0,
    )


def test_ranges_run_from_the_camera_to_the_centre_and_the_nearest_waterline() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1

    found = added(index, [target(1)]).annotations[0]

    assert (found.range_m, found.waterline_range_m, found.bearing_deg) == (
        pytest.approx(1000.0),
        pytest.approx(990.0),
        pytest.approx(0.0),
    )


def square(left: int = 20) -> np.ndarray:
    index = np.zeros((48, 64), dtype=int)
    index[10:20, left : left + 10] = 1
    return index


def boxed(index: np.ndarray, frame: np.ndarray) -> bool:
    return not labels.faint(index, frame, [target(1)])


def eight_bit(index: np.ndarray, values: dict[int, int]) -> np.ndarray:
    frame = np.full(index.shape, 120, np.uint8)
    for pass_index, value in values.items():
        frame[index == pass_index] = value
    return frame


def noisy(index: np.ndarray, offset: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.normal(120.0, 3.0, index.shape) + offset * (index == 1)
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def excess(index: np.ndarray, frame: np.ndarray) -> float:
    value = labels.excess_de(index, frame, 1)
    assert value is not None
    return value


def patch(size: int, left: int | None = None) -> np.ndarray:
    index = np.zeros((size + 24, 4 * size + 48), dtype=int)
    left = 2 * size + 12 if left is None else left
    index[12 : 12 + size, left : left + size] = 1
    return index


@pytest.mark.parametrize("size", [4, 16, 64])
@pytest.mark.parametrize("edge", [False, True])
def test_a_patch_of_the_backgrounds_own_noise_is_dropped(size: int, edge: bool) -> None:
    """At the edge the patch has one flank."""
    index = patch(size, left=0 if edge else None)
    for seed in range(20):
        assert not boxed(index, noisy(index, seed=seed))


@pytest.mark.parametrize(("size", "offset"), [(1, 8), (4, 6), (16, 6)])
def test_a_patch_offset_from_its_noise_is_kept(size: int, offset: float) -> None:
    index = patch(size)
    assert boxed(index, noisy(index, offset))


def test_a_target_exactly_at_the_threshold_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = patch(4)
    frame = noisy(index, 3)
    monkeypatch.setattr(labels, "MIN_EXCESS_DE", excess(index, frame))

    assert boxed(index, frame)


def test_bright_and_dark_halves_that_cancel_in_the_mean_are_kept() -> None:
    index = patch(16)
    frame = noisy(index)
    frame[12:20][index[12:20] == 1] = 160
    frame[20:28][index[20:28] == 1] = 80

    assert boxed(index, frame)


@pytest.mark.parametrize(("size", "value", "seen"), [(24, 60, True), (2, 128, False)])
def test_glitter_hides_only_what_it_outshines(
    size: int, value: int, seen: bool
) -> None:
    """White one-pixel glints round a dark hull or a faint speck."""
    rng = np.random.default_rng(size)
    index = np.zeros((64, 128), dtype=int)
    index[20 : 20 + size, 52 : 52 + size] = 1
    frame = np.full(index.shape, 120, np.uint8)
    frame[(rng.random(index.shape) < 0.1) & (index == 0)] = 255
    frame[index == 1] = value

    assert boxed(index, frame) is seen


@pytest.mark.parametrize(("offset", "seen"), [(0, False), (6, True)])
def test_the_background_follows_a_gradient_along_the_row(
    offset: float, seen: bool
) -> None:
    """Level, the background would count the gradient as clutter."""
    index = patch(8)
    columns = np.arange(index.shape[1])
    frame = noisy(index, offset).astype(float) - 40 + columns
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    assert boxed(index, frame) is seen


def test_a_wide_targets_flanks_reach_half_its_width() -> None:
    index = np.zeros((48, 96), dtype=int)
    index[10:20, 30:60] = 1

    beside = labels.flanks(index, 1)

    assert labels.flank_px(30) == 15
    assert beside[15, 30 - labels.FLANK_GAP_PX - 15]
    assert not beside[15, 30 - labels.FLANK_GAP_PX - 16]


def test_no_pixel_within_the_gap_is_a_flank() -> None:
    """A hull with a mast: the mast's rows reach past the gap beside the mast, but not
    into the gap above the hull.
    """
    index = np.zeros((48, 64), dtype=int)
    index[15:20, 20:40] = 1
    index[8:15, 28:32] = 1
    gap = 2 * labels.FLANK_GAP_PX + 1
    near = cv2.dilate((index == 1).astype(np.uint8), np.ones((gap, gap), np.uint8))

    beside = labels.flanks(index, 1)

    assert not (beside & (near == 1)).any()
    assert beside[17, 20 - labels.FLANK_GAP_PX - 1]
    assert beside[17, 39 + labels.FLANK_GAP_PX + 1]
    assert beside[9, 28 - labels.FLANK_GAP_PX - 1]


def test_a_target_near_the_left_edge_keeps_the_flank_it_has() -> None:
    """Its left flank would reach past column 0: it is cut there, not lost or
    wrapped round to the right edge.
    """
    index = square(left=10)

    beside = labels.flanks(index, 1)

    left = 10 - labels.FLANK_GAP_PX
    assert beside[15, :left].all()
    assert not beside[15, left:10].any()
    assert not beside[:, 40:].any()


@pytest.mark.parametrize(("swapped", "seen"), [(False, False), (True, True)])
def test_each_row_is_judged_against_its_own_background(
    swapped: bool, seen: bool
) -> None:
    """Across the horizon a target the colour of its row's sky or sea is unseen, though
    it differs from its surroundings as a whole.
    """
    index = square()
    sky, sea = (60, 200) if swapped else (200, 60)
    frame = np.full(index.shape, 200, np.uint8)
    frame[15:] = 60
    frame[10:15][index[10:15] == 1] = sky
    frame[15:20][index[15:20] == 1] = sea

    assert boxed(index, frame) is seen


def test_a_glint_in_the_flank_does_not_move_the_background() -> None:
    index = square()
    frame = eight_bit(index, {})
    frame[12, 36] = 255
    lab = labels.cielab(frame)
    mask = index == 1

    behind = labels.background(lab, mask, labels.flanks(index, 1))

    assert behind is not None
    assert behind[mask] == pytest.approx(np.broadcast_to(lab[0, 0], (mask.sum(), 3)))


def test_another_object_in_the_flank_is_not_background() -> None:
    """Counted, the bright neighbour would be noise that buries the target."""
    index = patch(4)
    frame = noisy(index, 6)
    index[12:16, 34:44] = 2
    frame[index == 2] = 200

    assert boxed(index, frame)


def test_a_target_as_bright_as_its_background_is_seen_by_its_colour() -> None:
    index = square()
    frame = cv2.cvtColor(eight_bit(index, {}), cv2.COLOR_GRAY2RGB)
    frame[index == 1] = (182, 91, 120)  # BT.709 luminance within 0.01 % of grey 120

    assert boxed(index, frame)


def test_a_target_with_no_background_to_measure_keeps_its_box() -> None:
    index = np.ones((48, 64), dtype=int)

    assert boxed(index, eight_bit(index, {}))


@pytest.mark.parametrize(
    "frame", [np.ones((48, 64, 3), np.float32), np.ones((48, 64), np.uint16)]
)
def test_a_frame_deeper_than_8_bits_keeps_every_box(frame: np.ndarray) -> None:
    assert boxed(square(), frame)


def test_a_frame_is_read_as_rgb() -> None:
    """Pure red is L* 53.2; read as BGR it would be blue, L* 32.3 (CIE 15:2004)."""
    red = np.array([[[255, 0, 0]]], np.uint8)

    assert labels.cielab(red)[0, 0, 0] == pytest.approx(53.2, abs=0.1)


def test_a_grey_frame_has_no_chroma() -> None:
    assert not labels.cielab(np.arange(256, dtype=np.uint8)[None])[..., 1:].any()


@pytest.mark.parametrize(("value", "seen"), [(60, False), (200, True)])
def test_a_row_with_no_flank_takes_the_nearest_rows_background(
    value: int, seen: bool
) -> None:
    """Sky over sea, the target the colour of each row but its flankless sea row."""
    index = square()
    index[17, :20] = index[17, 30:] = 2
    frame = np.full(index.shape, 200, np.uint8)
    frame[15:] = 60
    frame[17][index[17] == 1] = value

    assert boxed(index, frame) is seen


def grazing_circle_px(cam: CameraCalibration, radius_m: float) -> np.ndarray:
    """The horizon another way: the circle where rays from C touch the paraboloid,
    (x - cx)^2 + (y - cy)^2 = 2R cz + cx^2 + cy^2, projected. COCO pixels, by u.
    """
    pose = np.array(cam.extrinsics["world"])
    rotation, centre = pose[:3, :3], pose[:3, 3]
    cx, cy, cz = centre
    rho = math.sqrt(2 * radius_m * cz + cx * cx + cy * cy)
    phi = np.radians(np.arange(-180.0, 180.0, 0.001))
    x, y = cx + rho * np.sin(phi), cy + rho * np.cos(phi)
    ahead = rotation.T @ (
        np.stack([x, y, -(x * x + y * y) / (2 * radius_m)]) - centre[:, None]
    )
    ahead = ahead[:, ahead[2] > 0]
    if cam.distortion is None:
        u, v, _ = np.array(cam.K) @ ahead / ahead[2]
    else:
        across = np.hypot(ahead[0], ahead[1])
        scale = fisheye.radius_px(
            np.arctan2(across, ahead[2]), cam.distortion, cam.K[0][0]
        )
        u = cam.K[0][2] + scale * ahead[0] / across
        v = cam.K[1][2] + scale * ahead[1] / across
    order = np.argsort(u)
    return np.stack([u[order], v[order]], axis=1) + 0.5


FISHEYE = (-0.0793, -0.0461, 0.0331, -0.0065)


@pytest.mark.parametrize(
    ("pitch_deg", "roll_deg", "at_m", "distortion", "tolerance_px"),
    [
        (0.0, 0.0, (0.0, 0.0, 50.0), None, 1e-3),
        (-6.0, 2.0, (21.5, -68.2, 51.8), None, 1e-3),
        (0.0, 0.0, (0.0, 0.0, 50.0), FISHEYE, 0.05),
        (25.0, 0.0, (0.0, 0.0, 8.0), FISHEYE, 0.05),
        (0.0, 20.0, (21.5, -68.2, 51.8), FISHEYE, 0.05),
    ],
)
def test_the_horizon_lies_where_rays_graze_the_sea(
    pitch_deg: float,
    roll_deg: float,
    at_m: tuple[float, float, float],
    distortion: fisheye.K | None,
    tolerance_px: float,
) -> None:
    hfov_deg = 49.0 if distortion is None else 132.93
    cam = camera(3840, 2160, hfov_deg, pitch_deg, roll_deg, at_m, distortion)
    reference = grazing_circle_px(cam, RADIUS_M)

    points = np.array(labels.horizon_px(cam, RADIUS_M))

    assert len(points) == labels.HORIZON_POINTS
    expected = np.interp(points[:, 0], reference[:, 0], reference[:, 1])
    assert points[:, 1] == pytest.approx(expected, abs=tolerance_px)


def test_a_fisheyes_field_of_view_is_the_angle_across_its_width() -> None:
    assert camera(3840, 2160, 132.93, distortion=FISHEYE).hfov_deg == pytest.approx(
        132.93
    )
    assert camera(3840, 2160, 49.0).hfov_deg == pytest.approx(49.0)


def test_no_horizon_in_a_frame_that_looks_at_the_sea_only() -> None:
    assert labels.horizon_px(camera(pitch_deg=-60.0), RADIUS_M) == []


def written(root: Path, names: dict[str, labels.Labels]) -> list[Path]:
    """Each labels in a folder of its own under `root`."""
    folders = [root / name for name in names]
    for folder, truth in zip(folders, names.values(), strict=True):
        folder.mkdir()
        truth.write(folder)
    return folders


def test_a_merge_renumbers_reaches_every_file_and_keeps_each_scenario(
    tmp_path,
) -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    parts = {
        str(seed): added(index, [target(1, *category)])
        for seed, category in ((7, ("ship", 1)), (8, ("buoy", 6)))
    }
    for seed, truth in parts.items():
        truth.info = {"scenario": {"seed": int(seed)}, "version": "x"}

    merged = labels.merge(tmp_path, written(tmp_path, parts))

    assert [(i.id, i.file_name) for i in merged.images] == [
        (1, "7/C.png"),
        (2, "8/C.png"),
    ]
    assert [(a.id, a.image_id, a.category_id) for a in merged.annotations] == [
        (1, 1, 1),
        (2, 2, 6),
    ]
    assert [c.id for c in merged.categories] == [1, 6]
    assert merged.info == {
        "version": "x",
        "scenarios": {"7": {"seed": 7}, "8": {"seed": 8}},
    }


def test_a_merge_refuses_two_names_for_one_category(tmp_path) -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    parts = {name: added(index, [target(1, name, 1)]) for name in ("ship", "buoy")}
    for truth in parts.values():
        truth.info = {"scenario": {}}

    with pytest.raises(ValueError, match="category 1 is ship"):
        labels.merge(tmp_path, written(tmp_path, parts))


@pytest.mark.parametrize("graded", [True, False])
def test_an_8_bit_eo_image_carries_its_cameras_sensor_lens_and_processing(
    graded: bool,
) -> None:
    """An exr is the radiance as rendered: the chain never touched it."""
    settings = Camera(
        band="eo",
        hfov_deg=49.0,
        width_px=64,
        height_px=48,
        sensor=Sensor(full_well_e=10500.0, dark_noise_e=3.0),
        lens=Lens(vignetting=True),
    )
    frame = FLAT.astype(np.uint8) if graded else FLAT
    (image,) = added(frame=frame, settings=settings).images
    expected = (settings.sensor, settings.lens, settings.processing)
    assert (image.sensor, image.lens, image.processing) == (
        expected if graded else (None, None, None)
    )


def test_an_image_without_its_camera_carries_no_settings() -> None:
    (image,) = added().images
    assert (image.sensor, image.lens, image.processing) == (None, None, None)


@pytest.mark.parametrize("faint", [{}, {"t1": 0.3}], ids=["plain", "faint"])
def test_a_target_out_of_frame_is_dropped_with_where_it_was(
    faint: dict[str, float],
) -> None:
    (gone,) = added(targets=[target(1)], faint=faint).dropped

    assert (gone.image_id, gone.name, gone.reason) == (1, "t1", "no_pixels")
    assert gone.range_m == pytest.approx(1000.0)
    assert gone.bearing_deg == pytest.approx(0.0)


@pytest.mark.parametrize(("offset", "faint"), [(0.0, True), (8.0, False)])
def test_only_a_target_in_its_backgrounds_noise_is_faint(
    offset: float, faint: bool
) -> None:
    index = patch(4)

    found = labels.faint(index, noisy(index, offset), [target(1)])

    assert ("t1" in found) is faint


def test_a_faint_target_is_dropped_with_its_excess_and_box() -> None:
    truth = added(patch(4), [target(1)], faint={"t1": 0.3})

    assert not truth.annotations
    (gone,) = truth.dropped
    assert (gone.reason, gone.excess_de) == ("low_contrast", 0.3)
    assert (gone.bbox, gone.area) == ((20, 12, 4, 4), 16)


@pytest.mark.parametrize(
    ("elsewhere", "expected"),
    [({"t1"}, [{}, {}]), (set(), [{"t1": 0.3}, {}])],
    ids=["seen by the other camera", "out of the other camera's frame"],
)
def test_a_target_faint_in_one_camera_is_kept_when_another_sees_it(
    elsewhere: set[str], expected: list[dict[str, float]]
) -> None:
    faints = [{"t1": 0.3}, {}]

    found = labels.agreed(faints, [{"t1"}, elsewhere])

    assert found == expected


def test_a_merge_renumbers_the_dropped_images(tmp_path) -> None:
    parts = {str(seed): added(targets=[target(1)]) for seed in (7, 8)}
    for truth in parts.values():
        truth.info = {"scenario": {}}

    merged = labels.merge(tmp_path, written(tmp_path, parts))

    assert [d.image_id for d in merged.dropped] == [1, 2]


def recorded(tmp_path: Path, info: dict) -> Path:
    labels.Labels(info=info).write(tmp_path)
    return tmp_path


def test_a_camera_takes_the_agc_its_thermal_frames_recorded(tmp_path: Path) -> None:
    settings = Camera(
        band="ir", hfov_deg=24.0, width_px=64, height_px=48, agc=Agc(dde=1.2)
    )
    ir = camera().model_copy(update={"band": "ir"})
    added(cam=ir, frame=FLAT.astype(np.uint16), settings=settings).write(tmp_path)
    found = labels.agcs(tmp_path)
    assert found[ir.name] == Agc(dde=1.2)
    assert found["elsewhere"] == Agc()


def test_a_merged_set_is_refused(tmp_path: Path) -> None:
    folder = recorded(tmp_path, {"scenarios": {"0": {}, "1": {}}})
    with pytest.raises(ValueError, match="each seed's folder"):
        labels.agcs(folder)
