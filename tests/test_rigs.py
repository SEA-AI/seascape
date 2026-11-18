"""Two rigs of several cameras: where each sits and points, and the panoramas stitched
from them. One global Blender session, so it is built once.
"""

import itertools
import json
import math
from pathlib import Path
from typing import NamedTuple

import bpy
import cv2
import numpy as np
import pytest
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
from pydantic import ValidationError

from seascape import angles
from seascape.assets import Hull, manifest
from seascape.blender import scene, sea
from seascape.blender.blend import mesh_objects
from seascape.blender.rigs import _hold, boresight_deg
from seascape.blender.vessels import waterline_crossings, waterline_m
from seascape.config import BANDS, Agc, Microbolometer, Mount, Rig, Scenario
from seascape.outputs import agc, canvas, panorama
from seascape.outputs.calibration import Calibration, CameraCalibration
from seascape.physics import waves
from tests.scenarios import two_rigs

SCENARIO: Scenario = two_rigs()
MOUNTS = [pytest.param(mount, id=mount.name) for mount in SCENARIO.mounts]
RIGS = [pytest.param(name, rig, id=name) for name, rig in SCENARIO.rigs.items()]


@pytest.fixture(scope="module", autouse=True)
def built() -> scene.Built:
    return scene.build(SCENARIO, "eo")


def first_rig_mounts() -> list[Mount]:
    """The mounts of the first camera's band on its rig."""
    first = SCENARIO.mounts[0]
    return [
        m
        for m in SCENARIO.mounts
        if m.rig_name == first.rig_name and m.camera.band == first.camera.band
    ]


def black_frame(mount: Mount) -> np.ndarray:
    return np.zeros((mount.camera.height_px, mount.camera.width_px, 3), np.uint8)


def targets() -> list[bpy.types.Object]:
    return [obj for obj in bpy.data.objects if obj.name.startswith("target_")]


@pytest.mark.parametrize(("name", "rig"), RIGS)
def test_a_rigs_cameras_all_sit_at_its_mount_point(
    name: str, rig: Rig, built: scene.Built
) -> None:
    """Cameras yawed on the centreline pass every bearing check with no baseline."""
    expected = (rig.offset_x_m, rig.offset_y_m, rig.height_m)

    places = {
        tuple(_in_ship_frame(built, built.cameras[mount.name]).translation)
        for mount in SCENARIO.mounts
        if mount.rig_name == name
    }

    assert len(places) == 1
    assert places.pop() == pytest.approx(expected)


def test_the_ownship_is_at_the_origin(built: scene.Built) -> None:
    """On the water there, which the tall waves raise."""
    root = built.root
    tall, _ = sea.displaced((SCENARIO.wind_waves(), SCENARIO.swell_waves()))
    t_s = SCENARIO.outputs.times_s[bpy.context.scene.frame_current]
    rest = waves.gerstner_rest_m(tall, np.zeros(()), np.zeros(()), t_s)
    ridden = waves.heave_m(tall, waves.Waterplane(*map(float, rest)), t_s)

    assert tuple(root.location) == pytest.approx((0.0, 0.0, ridden))


def test_the_hull_takes_the_attitude_it_was_given(built: scene.Built) -> None:
    """Pins both signs and the order."""
    roll = math.radians(SCENARIO.ownship.roll_deg)
    pitch = math.radians(SCENARIO.ownship.pitch_deg)
    rotation = built.root.matrix_world.to_3x3()

    bow = rotation @ Vector((0.0, 1.0, 0.0))
    starboard = rotation @ Vector((1.0, 0.0, 0.0))

    assert bow.z == pytest.approx(math.sin(pitch))
    assert starboard.z == pytest.approx(-math.cos(pitch) * math.sin(roll))


def test_every_target_sits_at_the_configured_range() -> None:
    assert SCENARIO.targets is not None

    ranges_m = [
        math.hypot(target.location.x, target.location.y) for target in targets()
    ]

    assert ranges_m == pytest.approx([SCENARIO.targets.range_m] * len(ranges_m))


def test_hulls_of_one_asset_share_its_meshes(built: scene.Built) -> None:
    """Clones share datablocks; copied meshes show only when a build runs out of RAM."""
    assert SCENARIO.targets is not None
    for asset, roots in built.targets.items():
        meshes = {frozenset(p.data.name for p in mesh_objects([a])) for a in roots}

        assert len(meshes) == 1, asset
    assert len(targets()) == SCENARIO.targets.count


def test_the_calibration_projects_every_target_where_blender_draws_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built: scene.Built
) -> None:
    """Read back from disk, so a matrix that does not survive JSON fails here too."""
    written = Calibration(
        cameras=[
            scene.calibrate(built, m, f"{m.name}.png", 0.0) for m in SCENARIO.mounts
        ]
    ).write(tmp_path)
    render = bpy.context.scene.render

    projected = 0
    for camera in Calibration.read(written.parent).cameras:
        w, h = camera.width_px, camera.height_px
        # world_to_camera_view takes its aspect from the scene's resolution.
        monkeypatch.setattr(render, "resolution_x", w)
        monkeypatch.setattr(render, "resolution_y", h)
        world_to_cam = np.linalg.inv(camera.extrinsics["world"])
        for target in targets():
            uv = world_to_camera_view(
                bpy.context.scene, built.cameras[camera.name], target.location
            )
            if not _in_frame(uv):
                continue
            x, y, z, _ = world_to_cam @ (*target.location, 1.0)
            u, v, _ = np.array(camera.K) @ (x, y, z) / z
            # Blender's view runs 0-1 across pixel edges, bottom up.
            assert (u, v) == pytest.approx(
                (uv.x * w - 0.5, (1.0 - uv.y) * h - 0.5), abs=1e-3
            ), camera.name
            projected += 1

    assert projected, "no target in any frame: the assertions above ran on nothing"


def test_a_waterline_rings_its_hull_within_its_length(built: scene.Built) -> None:
    """World metres: in the mesh's own frame it would sit at the origin, far off."""
    for asset, roots in built.targets.items():
        for root in roots:
            centre = np.array(root.matrix_world.translation.xy)
            reach = np.linalg.norm(
                waterline_m(root, waterline_crossings(root)) - centre, axis=1
            )

            hull = manifest()[asset]
            assert isinstance(hull, Hull), asset
            assert 0.0 < reach.max() <= hull.length_m / 2, root.name


def test_a_calibration_with_fields_it_does_not_know_still_reads(
    tmp_path: Path, built: scene.Built
) -> None:
    mount = SCENARIO.mounts[0]
    record = scene.calibrate(built, mount, "", 0.0).model_dump()
    (tmp_path / "calibration.json").write_text(
        json.dumps({"cameras": [{**record, "serial": "X"}], "rig": "Y"})
    )

    (camera,) = Calibration.read(tmp_path).cameras

    assert camera.name == mount.name


@pytest.mark.parametrize("mount", MOUNTS)
def test_in_its_rig_a_camera_points_exactly_as_asked(
    mount: Mount, built: scene.Built
) -> None:
    axis = np.array(scene.calibrate(built, mount, "", 0.0).extrinsics["head"])[:3, 2]

    bearing = math.degrees(math.atan2(axis[0], axis[1]))
    elevation = math.degrees(math.asin(axis[2]))

    assert (bearing, elevation) == pytest.approx(
        (mount.camera.yaw_deg, mount.camera.pitch_deg), abs=1e-4
    )


def test_a_calibration_records_its_camera_s_pitch(built: scene.Built) -> None:
    mount = SCENARIO.band_mounts("ir")[0]
    sensor = Microbolometer(pitch_um=17.0, f_number=1.0)
    pitched = mount._replace(
        camera=mount.camera.model_copy(update={"microbolometer": sensor})
    )
    assert scene.calibrate(built, pitched, "ir.png", 0.0).pitch_um == 17.0


@pytest.mark.parametrize("mount", MOUNTS)
def test_a_panorama_puts_each_principal_point_on_its_boresight(
    mount: Mount, built: scene.Built
) -> None:
    """A slip in the stitcher's frame flips or mirrors the panorama."""
    k, r = panorama.pose(scene.calibrate(built, mount, "", 0.0), "world", 0.0)
    warper = cv2.PyRotationWarper("spherical", 1.0)

    # Spherical: u is the bearing, v the angle down from straight up.
    u, v = warper.warpPoint((float(k[0, 2]), float(k[1, 2])), k, r)

    bearing_deg, elevation_deg = boresight_deg(built.cameras[mount.name])
    # A stern camera reads either end of (-180, 180].
    assert angles.wrapped_deg(math.degrees(u) - bearing_deg) == pytest.approx(
        0.0, abs=1e-3
    )
    assert 90.0 - math.degrees(v) == pytest.approx(elevation_deg, abs=1e-3)


def test_a_panorama_lays_its_cameras_out_in_yaw_order(
    tmp_path: Path, built: scene.Built
) -> None:
    mounts = sorted(first_rig_mounts()[:3], key=lambda m: m.camera.yaw_deg)
    for channel, mount in enumerate(mounts):
        frame = black_frame(mount)
        frame[..., channel] = 255
        cv2.imwrite(str(tmp_path / f"{mount.name}.png"), frame)
    Calibration(
        cameras=[scene.calibrate(built, m, f"{m.name}.png", 0.0) for m in mounts]
    ).write(tmp_path)

    paths = panorama.panoramas(tmp_path, "cylindrical", "head", max_width=400)

    image = cv2.imread(str(paths[0]))
    assert image is not None
    row = image[image.shape[0] // 2]
    columns = [np.flatnonzero(row[:, c] > 127).mean() for c in range(len(mounts))]
    assert columns == sorted(columns)


def test_a_panorama_centres_on_what_its_cameras_see(built: scene.Built) -> None:
    """The bow's eo at 0 and 40 deg and its ir at 0: their optical axes sum to 13 deg,
    the middle of what they see is 20 deg.
    """
    mounts = [
        m for m in SCENARIO.mounts if m.rig_name == "bow" and m.camera.yaw_deg >= 0.0
    ]
    cameras = [scene.calibrate(built, m, f"{m.name}.png", 0.0) for m in mounts]

    assert panorama.axis_deg(cameras, "head") == pytest.approx(20.0)


class Overlay(NamedTuple):
    image: np.ndarray
    layout: panorama.Layout
    ir: np.ndarray  # the IR's column, BGR as int
    folder: Path
    cameras: list[CameraCalibration]


@pytest.fixture(scope="module")
def overlay(tmp_path_factory: pytest.TempPathFactory, built: scene.Built) -> Overlay:
    """The bow's eo at 0 and 40 deg in blue and its ir at 0 in red, given IR first:
    the bands' own axes differ, so only one canvas lines them up, and the stack is
    the overlay's order, not the cameras'.
    """
    folder = tmp_path_factory.mktemp("overlay")
    mounts = [
        m for m in SCENARIO.mounts if m.rig_name == "bow" and m.camera.yaw_deg >= 0.0
    ]
    for mount in mounts:
        frame = black_frame(mount)
        frame[..., 0 if mount.camera.band == "eo" else 2] = 255
        cv2.imwrite(str(folder / f"{mount.name}.png"), frame)
    cameras = [scene.calibrate(built, m, f"{m.name}.png", 0.0) for m in mounts]
    cameras.sort(key=lambda c: c.band != "ir")
    image, layout = panorama.overlaid(folder, cameras, "cylindrical", "head", 600)
    centre = panorama.column(layout, 0.0)  # the IR looks along the head's +Y
    assert centre is not None
    return Overlay(image, layout, image[:, round(centre)].astype(int), folder, cameras)


def test_groups_stitched_among_their_rig_share_one_canvas(overlay: Overlay) -> None:
    """Alone, the eo pair and the ir would each get their own axis and extent."""
    eo, ir = ([c for c in overlay.cameras if c.band == b] for b in ("eo", "ir"))
    eo_image, eo_layout = panorama.stitch(
        overlay.folder, eo, "cylindrical", "head", 600, among=overlay.cameras
    )
    ir_image, ir_layout = panorama.stitch(
        overlay.folder, ir, "cylindrical", "head", 600, among=overlay.cameras
    )
    assert (ir_image.shape, ir_layout) == (eo_image.shape, eo_layout)


MIXED = np.array([255 * (1 - panorama.OPACITY), 0, 255 * panorama.OPACITY])  # BGR
IR_COLOUR = canvas.palette(len(BANDS))[BANDS.index("ir")][::-1]


def test_an_overlay_mixes_the_ir_over_the_eo(overlay: Overlay) -> None:
    # Outlines and names are other colours, so only the mix matches; 1 is the
    # blender's and addWeighted's rounding to uint8.
    assert (np.abs(overlay.ir - MIXED).max(axis=1) <= 1).any()


def test_an_overlay_lays_the_ir_opaque_where_no_eo_lies_beneath(
    overlay: Overlay,
) -> None:
    """The ir is level and the eo pitched down, so the ir reaches above them."""
    assert (np.abs(overlay.ir - (0, 0, 255)).max(axis=1) <= 1).any()


def test_an_overlay_keeps_the_ir_within_its_frame(overlay: Overlay) -> None:
    eo_only = panorama.column(overlay.layout, 40.0)  # eo_s's boresight
    assert eo_only is not None
    blue, _, red = overlay.image[overlay.image.shape[0] // 2, round(eo_only)]
    assert blue > 127 and red == 0


def test_an_overlay_names_a_later_band_over_its_bottom_edge(overlay: Overlay) -> None:
    c = round(panorama.column(overlay.layout, 0.0) or 0)  # the IR's name is centred
    rows = np.flatnonzero((overlay.image[:, c] == IR_COLOUR).all(axis=1))
    top, bottom = rows[0], rows[-1]  # its outline
    # Judgement: 3 px clears the outline's stroke, 2 px wide here.
    name = rows[(rows > top + 3) & (rows < bottom - 3)]
    assert name.size and name.mean() > (top + bottom) / 2


def test_an_overlay_outlines_each_frame(overlay: Overlay) -> None:
    beside = panorama.column(overlay.layout, -10.0)  # inside the IR, past its name
    assert beside is not None
    assert (overlay.image[:, round(beside)] == IR_COLOUR).all(axis=1).any()


def test_an_overlay_writes_one_file_per_rig_whatever_the_rigs_are_called(
    tmp_path: Path, built: scene.Built
) -> None:
    """Rig `bow` with eo and ir, and rig `bow_eo` with ir: a name of rig and bands
    would give both `bow_eo_ir`.
    """
    first = SCENARIO.mounts[0]
    mounts = [m for m in SCENARIO.mounts if m.rig_name == first.rig_name]
    for mount in mounts:
        cv2.imwrite(str(tmp_path / f"{mount.name}.png"), black_frame(mount))
    cameras = [scene.calibrate(built, m, f"{m.name}.png", 0.0) for m in mounts]
    ir = next(c for c in cameras if c.band == "ir")
    twin = ir.model_copy(update={"rig": f"{ir.rig}_eo", "name": f"{ir.name}_twin"})
    Calibration(cameras=[*cameras, twin]).write(tmp_path)

    paths = panorama.panoramas(tmp_path, "cylindrical", "head", 400, overlay=True)

    assert len(set(paths)) == 2


def test_a_calibration_refuses_a_band_seascape_has_not(built: scene.Built) -> None:
    mount = SCENARIO.mounts[0]
    camera = scene.calibrate(built, mount, f"{mount.name}.png", 0.0)
    with pytest.raises(ValidationError):
        CameraCalibration.model_validate({**camera.model_dump(), "band": "uv"})


def test_a_panorama_refuses_a_sequence(tmp_path: Path, built: scene.Built) -> None:
    mount = SCENARIO.mounts[0]
    Calibration(
        cameras=[
            scene.calibrate(built, mount, f"{mount.name}_{i}.png", 0.0) for i in (0, 1)
        ]
    ).write(tmp_path)
    with pytest.raises(ValueError, match="a sequence"):
        panorama.panoramas(tmp_path, "cylindrical", "head")


def test_a_thermal_panorama_is_one_grey_per_temperature(
    tmp_path: Path, built: scene.Built
) -> None:
    mounts = first_rig_mounts()[:2]
    thermal, white = tmp_path / "thermal", tmp_path / "white"
    calibration = Calibration(
        cameras=[scene.calibrate(built, m, f"{m.name}.png", 0.0) for m in mounts]
    )
    for folder in (thermal, white):
        folder.mkdir()
        calibration.write(folder)
    for hot_k, mount in zip((300.0, 290.0), mounts, strict=True):
        t_k = np.full((mount.camera.height_px, mount.camera.width_px), 290.0)
        t_k[:8] = hot_k  # only the first frame has anything warmer than the sea
        cv2.imwrite(str(thermal / f"{mount.name}.png"), agc.counts(t_k))
        frame = np.full((mount.camera.height_px, mount.camera.width_px, 3), 255)
        cv2.imwrite(str(white / f"{mount.name}.png"), frame.astype(np.uint8))

    image, coverage = (
        cv2.imread(str(panorama.panoramas(f, "cylindrical", "head", max_width=400)[0]))
        for f in (thermal, white)
    )
    assert image is not None and coverage is not None
    middle = image.shape[0] // 2
    sea = image[middle, :, 0][coverage[middle, :, 0] > 127]
    black = Agc().detail_headroom  # the sea is flat: no detail goes below it
    assert sea.size > 0
    assert sea.min() >= black - 8, "the sea lost its detail headroom"
    assert sea.max() <= black + 8, "the second frame's sea came out grey"


def test_the_width_is_capped_and_never_raised(
    tmp_path: Path, built: scene.Built
) -> None:
    mount = SCENARIO.mounts[0]
    cv2.imwrite(str(tmp_path / "frame.png"), black_frame(mount))
    camera = scene.calibrate(built, mount, "frame.png", 0.0)

    native, _ = panorama.stitch(tmp_path, [camera], "cylindrical", "head")
    capped, _ = panorama.stitch(tmp_path, [camera], "cylindrical", "head", 101)
    uncapped, _ = panorama.stitch(tmp_path, [camera], "cylindrical", "head", 10**6)

    assert capped.shape[1] == 101
    assert uncapped.shape == native.shape


@pytest.mark.parametrize("projection", list(panorama.PROJECTIONS))
def test_a_bearing_lands_where_the_stitch_put_it(
    tmp_path: Path, projection: str, built: scene.Built
) -> None:
    """A dot right of each principal point, found in the stitch, against the column
    its ray's bearing maps to. Off centre, so a mirrored mapping finds no dot; on a
    pitched camera, so the mapping is held independent of elevation.
    """
    cameras = [
        scene.calibrate(built, m, f"{m.name}.png", 0.0) for m in first_rig_mounts()
    ]
    for camera in cameras:
        frame = np.zeros((camera.height_px, camera.width_px, 3), np.uint8)
        cx, cy = round(camera.K[0][2] + camera.width_px / 4), round(camera.K[1][2])
        frame[cy - 6 : cy + 7, cx - 6 : cx + 7] = 255
        cv2.imwrite(str(tmp_path / camera.image), frame)

    image, layout = panorama.stitch(tmp_path, cameras, projection, "head", 2000)

    for camera in cameras:
        dot = (camera.K[0][2] + camera.width_px / 4, camera.K[1][2], 1.0)
        ray = (
            np.array(camera.extrinsics["head"])[:3, :3] @ np.linalg.inv(camera.K) @ dot
        )
        x, y, _ = ray
        expected = panorama.column(layout, math.degrees(math.atan2(x, y)))
        assert expected is not None
        window = image[:, max(0, round(expected) - 40) : round(expected) + 41, 0]
        found = max(0, round(expected) - 40) + int(window.sum(axis=0).argmax())
        assert found == pytest.approx(expected, abs=1.5), camera.name


def test_the_ruler_ticks_under_the_image_where_column_says(
    tmp_path: Path, built: scene.Built
) -> None:
    mount = SCENARIO.mounts[0]
    cv2.imwrite(str(tmp_path / "frame.png"), black_frame(mount))
    camera = scene.calibrate(built, mount, "frame.png", 0.0)
    image, layout = panorama.stitch(tmp_path, [camera], "cylindrical", "head", 1000)
    # The labelled bearing nearest the camera's axis, so it is in frame.
    x, y, _ = np.array(camera.extrinsics["head"])[:3, 2]
    bearing = panorama.LABEL_DEG * round(
        math.degrees(math.atan2(x, y)) / panorama.LABEL_DEG
    )

    ruled = panorama.ruled(image, layout)

    strip = ruled[image.shape[0] :]
    assert (ruled[: image.shape[0]] == image).all()
    tick = strip[:16, round(panorama.column(layout, bearing) or -1)]
    assert (tick == canvas.INK).all()


def test_rectilinear_refuses_a_camera_behind_its_plane(built: scene.Built) -> None:
    """Two cameras back to back: no plane faces both, whatever their field."""
    camera = scene.calibrate(built, SCENARIO.mounts[0], "", 0.0)
    turned = Matrix.Rotation(math.pi, 4, "Z") @ Matrix(camera.extrinsics["world"])
    behind = camera.model_copy(
        update={"extrinsics": {"world": tuple(map(tuple, turned))}}
    )

    with pytest.raises(ValueError, match="rectilinear cannot show"):
        panorama.stitch(Path(), [camera, behind], "rectilinear", "world")


def test_a_frame_the_calibration_lacks_is_named(built: scene.Built) -> None:
    camera = scene.calibrate(built, SCENARIO.mounts[0], "", 0.0)
    headless = camera.model_copy(
        update={"extrinsics": {"world": camera.extrinsics["world"]}}
    )

    with pytest.raises(ValueError, match="'head'"):
        panorama.stitch(Path(), [headless], "cylindrical", "head")


def test_a_max_width_under_a_pixel_is_an_error(built: scene.Built) -> None:
    camera = scene.calibrate(built, SCENARIO.mounts[0], "", 0.0)

    with pytest.raises(ValueError, match="max width"):
        panorama.stitch(Path(), [camera], "cylindrical", "head", 0)


def test_a_shrunk_frame_keeps_its_principal_point_at_its_centre(
    built: scene.Built,
) -> None:
    camera = scene.calibrate(built, SCENARIO.mounts[0], "", 0.0)
    k, _ = panorama.pose(camera, "head", 0.0)

    (w, h), k = panorama._shrunk((camera.width_px, camera.height_px), k, 0.25)

    assert (k[0, 2], k[1, 2]) == pytest.approx(((w - 1) / 2, (h - 1) / 2))
    assert k[0, 0] == pytest.approx(camera.K[0][0] * w / camera.width_px)


def _in_ship_frame(built: scene.Built, obj: bpy.types.Object) -> Matrix:
    """The rig is specified on the hull, so it is measured there."""
    return built.root.matrix_world.inverted() @ obj.matrix_world


def _in_frame(uv) -> bool:
    """Behind the camera `world_to_camera_view` still lands x, y in range; z decides."""
    return 0.0 <= uv.x <= 1.0 and 0.0 <= uv.y <= 1.0 and uv.z > 0.0


def test_a_pan_past_the_stern_keys_no_spin_back() -> None:
    """A pan from 170 to 190 deg turns 20 deg, not 340 the other way round."""
    # Beside the module's built scene, which the other tests share.
    head = bpy.data.objects.new("head", None)
    bpy.context.scene.collection.objects.link(head)
    rig = next(iter(two_rigs().rigs.values()))
    times = [0.0, 1.0, 2.0]
    # The axis, yaw and pan, sweeps 170 to 190 deg whatever the rig's own yaw.
    _hold(head, lambda t: rig.pose(170.0 - rig.yaw_deg + 10.0 * t), times)
    yaws = []
    for frame in range(len(times)):
        bpy.context.scene.frame_set(frame)
        yaws.append(head.rotation_euler[2])
    bpy.context.scene.frame_set(0)
    bpy.data.objects.remove(head)
    # Each frame pans 10 deg; a spin back steps about 340.
    assert max(abs(b - a) for a, b in itertools.pairwise(yaws)) < math.radians(15.0)
