"""labels.json from an index map and a calibration, without a render."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from seascape.config import Agc, Camera, Lens, Sensor
from seascape.outputs import labels
from seascape.outputs.calibration import CameraCalibration
from seascape.physics import waves

RADIUS_M = waves.earth_radius_m(0.13)
FLAT = np.ones((48, 64, 3))


def camera(
    width: int = 64,
    height: int = 48,
    hfov_deg: float = 49.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    at_m: tuple[float, float, float] = (0.0, 0.0, 50.0),
) -> CameraCalibration:
    """Facing north, then pitched about its x and rolled about its z."""
    f = width / 2 / math.tan(math.radians(hfov_deg) / 2)
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


def test_an_annotation_records_when_the_engines_stopped() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10:13, 20:25] = 1
    truth = labels.Labels()
    stopped = target(1)._replace(engine_off_min=45.0)

    truth.add(camera(), 0.0, index, FLAT, [stopped], RADIUS_M, faint={})

    (found,) = truth.annotations
    assert found.engine_off_min == 45.0


def test_a_box_covers_whole_pixels_from_its_top_left_corner() -> None:
    """COCO's pixel i spans [i, i + 1), so an inclusive box grows by one."""
    index = np.zeros((48, 64), dtype=int)
    index[10:13, 20:25] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M, faint={})

    (found,) = truth.annotations
    assert found.bbox == (20, 10, 5, 3)
    assert found.area == 15
    assert not found.truncated


def test_a_box_on_the_frame_edge_is_truncated() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[40:, 60:] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M, faint={})

    assert truth.annotations[0].truncated


def test_only_a_target_in_frame_is_labelled_or_categorised() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(
        camera(),
        0.0,
        index,
        FLAT,
        [target(1), target(2, "buoy", 6)],
        RADIUS_M,
        faint={},
    )

    assert [a.name for a in truth.annotations] == ["t1"]
    assert [c.name for c in truth.categories] == ["ship"]


def test_targets_of_one_category_share_it() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1), target(2)], RADIUS_M, faint={})

    assert {a.category_id for a in truth.annotations} == {1}
    assert [(c.name, c.supercategory) for c in truth.categories] == [("ship", "vessel")]


def test_a_category_keeps_its_own_id_whatever_came_first() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10], index[20, 20] = 1, 2
    truth = labels.Labels()

    truth.add(
        camera(),
        0.0,
        index,
        FLAT,
        [target(1, "buoy", 6), target(2)],
        RADIUS_M,
        faint={},
    )

    assert [(c.id, c.name) for c in truth.categories] == [(6, "buoy"), (1, "ship")]
    assert [a.category_id for a in truth.annotations] == [6, 1]


def test_an_image_carries_its_cameras_field_of_view() -> None:
    truth = labels.Labels()

    truth.add(
        camera(hfov_deg=42.0),
        0.0,
        np.zeros((48, 64), dtype=int),
        FLAT,
        [],
        RADIUS_M,
        faint={},
    )

    assert truth.images[0].hfov_deg == pytest.approx(42.0)


def test_an_annotation_carries_its_targets_heading_and_dimensions() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M, faint={})

    (annotation,) = truth.annotations
    assert (annotation.heading_deg, annotation.dims_m) == (90.0, (20.0, 5.0, 4.0))


def test_a_frame_carries_its_time() -> None:
    truth = labels.Labels()

    truth.add(
        camera(), 2.5, np.zeros((48, 64), dtype=int), FLAT, [], RADIUS_M, faint={}
    )

    assert truth.images[0].time_s == 2.5


def test_ranges_run_from_the_camera_to_the_centre_and_the_nearest_waterline() -> None:
    index = np.zeros((48, 64), dtype=int)
    index[10, 10] = 1
    truth = labels.Labels()

    truth.add(
        camera(at_m=(0.0, 0.0, 50.0)), 0.0, index, FLAT, [target(1)], RADIUS_M, faint={}
    )

    found = truth.annotations[0]
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
    into the gap above the hull."""
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
    wrapped round to the right edge."""
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
    it differs from its surroundings as a whole."""
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
    (x - cx)^2 + (y - cy)^2 = 2R cz + cx^2 + cy^2, projected. COCO pixels, by u."""
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
    u, v, _ = np.array(cam.K) @ ahead / ahead[2]
    order = np.argsort(u)
    return np.stack([u[order], v[order]], axis=1) + 0.5


@pytest.mark.parametrize(
    ("pitch_deg", "roll_deg", "at_m"),
    [(0.0, 0.0, (0.0, 0.0, 50.0)), (-6.0, 2.0, (21.5, -68.2, 51.8))],
)
def test_the_horizon_lies_where_rays_graze_the_sea(
    pitch_deg: float, roll_deg: float, at_m: tuple[float, float, float]
) -> None:
    cam = camera(3840, 2160, 49.0, pitch_deg, roll_deg, at_m)
    reference = grazing_circle_px(cam, RADIUS_M)

    points = np.array(labels.horizon_px(cam, RADIUS_M))

    assert len(points) == labels.HORIZON_POINTS
    expected = np.interp(points[:, 0], reference[:, 0], reference[:, 1])
    assert points[:, 1] == pytest.approx(expected, abs=1e-3)


def test_no_horizon_in_a_frame_that_looks_at_the_sea_only() -> None:
    assert labels.horizon_px(camera(pitch_deg=-60.0), RADIUS_M) == []


def test_a_merge_renumbers_reaches_every_file_and_keeps_each_scenario(
    tmp_path,
) -> None:
    folders = []
    for seed, category in ((7, ("ship", 1)), (8, ("buoy", 6))):
        index = np.zeros((48, 64), dtype=int)
        index[10, 10] = 1
        truth = labels.Labels(info={"scenario": {"seed": seed}, "version": "x"})
        truth.add(
            camera(), 0.0, index, FLAT, [target(1, *category)], RADIUS_M, faint={}
        )
        folders.append(tmp_path / str(seed))
        folders[-1].mkdir()
        truth.write(folders[-1])

    merged = labels.merge(tmp_path, folders)

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
    folders = []
    for name in ("ship", "buoy"):
        index = np.zeros((48, 64), dtype=int)
        index[10, 10] = 1
        truth = labels.Labels(info={"scenario": {}})
        truth.add(camera(), 0.0, index, FLAT, [target(1, name, 1)], RADIUS_M, faint={})
        folders.append(tmp_path / name)
        folders[-1].mkdir()
        truth.write(folders[-1])

    with pytest.raises(ValueError, match="category 1 is ship"):
        labels.merge(tmp_path, folders)


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
    truth = labels.Labels()
    frame = FLAT.astype(np.uint8) if graded else FLAT
    truth.add(
        camera(), 0.0, np.zeros((48, 64), int), frame, [], RADIUS_M, settings, faint={}
    )
    (image,) = truth.images
    expected = (settings.sensor, settings.lens, settings.processing)
    assert (image.sensor, image.lens, image.processing) == (
        expected if graded else (None, None, None)
    )


def test_an_image_without_its_camera_carries_no_settings() -> None:
    truth = labels.Labels()
    truth.add(camera(), 0.0, np.zeros((48, 64), int), FLAT, [], RADIUS_M, faint={})
    (image,) = truth.images
    assert (image.sensor, image.lens, image.processing) == (None, None, None)


def test_a_target_out_of_frame_is_dropped_with_where_it_was() -> None:
    truth = labels.Labels()

    truth.add(
        camera(),
        0.0,
        np.zeros((48, 64), dtype=int),
        FLAT,
        [target(1)],
        RADIUS_M,
        faint={},
    )

    (gone,) = truth.dropped
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
    index = patch(4)
    truth = labels.Labels()

    truth.add(camera(), 0.0, index, FLAT, [target(1)], RADIUS_M, faint={"t1": 0.3})

    assert not truth.annotations
    (gone,) = truth.dropped
    assert (gone.reason, gone.excess_de) == ("low_contrast", 0.3)
    assert (gone.bbox, gone.area) == ((20, 12, 4, 4), 16)


def test_a_faint_target_out_of_frame_is_dropped_for_having_no_pixels() -> None:
    truth = labels.Labels()

    truth.add(
        camera(), 0.0, np.zeros((48, 64), int), FLAT, [target(1)], RADIUS_M,
        faint={"t1": 0.3},
    )  # fmt: skip

    (gone,) = truth.dropped
    assert gone.reason == "no_pixels"


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
    folders = []
    for seed in (7, 8):
        truth = labels.Labels(info={"scenario": {"seed": seed}})
        truth.add(
            camera(),
            0.0,
            np.zeros((48, 64), dtype=int),
            FLAT,
            [target(1)],
            RADIUS_M,
            faint={},
        )
        folders.append(tmp_path / str(seed))
        folders[-1].mkdir()
        truth.write(folders[-1])

    merged = labels.merge(tmp_path, folders)

    assert [d.image_id for d in merged.dropped] == [1, 2]


def recorded(tmp_path: Path, info: dict) -> Path:
    labels.Labels(info=info).write(tmp_path)
    return tmp_path


def ir(dde: float) -> dict:
    """A recorded rig, with a field an older seascape had and this one does not."""
    return {
        "rigs": {"bow": {"retired_m": 1.0, "cameras": {"ir": {"agc": {"dde": dde}}}}}
    }


def test_a_camera_takes_the_agc_its_scenario_recorded(tmp_path: Path) -> None:
    found = labels.agcs(recorded(tmp_path, {"scenario": ir(1.2)}))
    assert found["bow_ir"] == Agc(dde=1.2)
    assert found["bow_eo"] == Agc()


def test_seeds_that_drew_different_agcs_are_not_toned_as_one(tmp_path: Path) -> None:
    folder = recorded(tmp_path, {"scenarios": {"0": ir(1.2), "1": ir(0.9)}})
    with pytest.raises(ValueError, match="bow_ir"):
        labels.agcs(folder)
