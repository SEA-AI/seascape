"""Two rigs of several cameras: where each sits and points, and the panoramas stitched
from them. One global Blender session, so it is built once."""

import json
import math
from pathlib import Path

import bpy
import cv2
import numpy as np
import pytest
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

from seascape.assets import Hull, manifest
from seascape.blender import scene, sea
from seascape.blender.rigs import boresight_deg
from seascape.blender.vessels import waterline_m
from seascape.config import Agc, Mount, Rig, Scenario
from seascape.outputs import agc, panorama
from seascape.outputs.calibration import Calibration
from seascape.outputs.montage import INK
from seascape.physics import waves
from tests.scenarios import two_rigs

SCENARIO: Scenario = two_rigs()
MOUNTS = [pytest.param(mount, id=mount.name) for mount in SCENARIO.mounts]
RIGS = [pytest.param(name, rig, id=name) for name, rig in SCENARIO.rigs.items()]


@pytest.fixture(scope="module", autouse=True)
def built() -> scene.Built:
    return scene.build(SCENARIO, "eo")


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
    anchor = built.vessel
    tall, _ = sea.displaced((SCENARIO.wind_waves(), SCENARIO.swell_waves()))
    t_s = SCENARIO.outputs.times_s[bpy.context.scene.frame_current]
    rest = waves.gerstner_rest_m(tall, np.zeros(()), np.zeros(()), t_s)
    ridden = waves.heave_m(tall, *map(float, rest), 0.0, 0.0, 0.0, t_s)

    assert tuple(anchor.location) == pytest.approx((0.0, 0.0, ridden))


def test_the_hull_takes_the_attitude_it_was_given(built: scene.Built) -> None:
    """Pins both signs and the order."""
    roll = math.radians(SCENARIO.ownship.roll_deg)
    pitch = math.radians(SCENARIO.ownship.pitch_deg)
    rotation = built.vessel.matrix_world.to_3x3()

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
    for asset, anchors in built.targets.items():
        meshes = {
            frozenset(p.data.name for p in a.children_recursive if p.type == "MESH")
            for a in anchors
        }

        assert len(meshes) == 1, asset
    assert len(targets()) == SCENARIO.targets.count


def test_the_calibration_projects_every_target_where_blender_draws_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built: scene.Built
) -> None:
    """Read back from disk, so a matrix that does not survive JSON fails here too."""
    written = Calibration(
        cameras=[scene.calibrate(built, m, f"{m.name}.png") for m in SCENARIO.mounts]
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
    for asset, anchors in built.targets.items():
        for anchor in anchors:
            centre = np.array(anchor.matrix_world.translation.xy)
            reach = np.linalg.norm(waterline_m(anchor) - centre, axis=1)

            hull = manifest()[asset]
            assert isinstance(hull, Hull), asset
            assert 0.0 < reach.max() <= hull.length_m / 2, anchor.name


def test_a_calibration_with_fields_it_does_not_know_still_reads(
    tmp_path: Path, built: scene.Built
) -> None:
    mount = SCENARIO.mounts[0]
    record = scene.calibrate(built, mount, "").model_dump()
    (tmp_path / "calibration.json").write_text(
        json.dumps({"cameras": [{**record, "serial": "X"}], "rig": "Y"})
    )

    (camera,) = Calibration.read(tmp_path).cameras

    assert camera.name == mount.name


@pytest.mark.parametrize("mount", MOUNTS)
def test_in_its_rig_a_camera_points_exactly_as_asked(
    mount: Mount, built: scene.Built
) -> None:
    axis = np.array(scene.calibrate(built, mount, "").extrinsics["rig"])[:3, 2]

    bearing = math.degrees(math.atan2(axis[0], axis[1]))
    elevation = math.degrees(math.asin(axis[2]))

    assert (bearing, elevation) == pytest.approx(
        (mount.camera.yaw_deg, mount.camera.pitch_deg), abs=1e-4
    )


@pytest.mark.parametrize("mount", MOUNTS)
def test_a_panorama_puts_each_principal_point_on_its_boresight(
    mount: Mount, built: scene.Built
) -> None:
    """A slip in the stitcher's frame flips or mirrors the panorama."""
    k, r = panorama.pose(scene.calibrate(built, mount, ""), "world", 0.0)
    warper = cv2.PyRotationWarper("spherical", 1.0)

    # Spherical: u is the bearing, v the angle down from straight up.
    u, v = warper.warpPoint((float(k[0, 2]), float(k[1, 2])), k, r)

    assert (math.degrees(u), 90.0 - math.degrees(v)) == pytest.approx(
        boresight_deg(built.cameras[mount.name]), abs=1e-3
    )


def test_a_panorama_lays_its_cameras_out_in_yaw_order(
    tmp_path: Path, built: scene.Built
) -> None:
    first = SCENARIO.mounts[0]
    mounts = [
        m
        for m in SCENARIO.mounts
        if m.rig_name == first.rig_name and m.camera.band == first.camera.band
    ][:3]
    mounts.sort(key=lambda m: m.camera.yaw_deg)
    for channel, mount in enumerate(mounts):
        frame = np.zeros((mount.camera.height_px, mount.camera.width_px, 3), np.uint8)
        frame[..., channel] = 255
        cv2.imwrite(str(tmp_path / f"{mount.name}.png"), frame)
    Calibration(
        cameras=[scene.calibrate(built, m, f"{m.name}.png") for m in mounts]
    ).write(tmp_path)

    paths = panorama.panoramas(tmp_path, "cylindrical", "rig", max_width=400)

    image = cv2.imread(str(paths[0]))
    assert image is not None
    row = image[image.shape[0] // 2]
    columns = [np.flatnonzero(row[:, c] > 127).mean() for c in range(len(mounts))]
    assert columns == sorted(columns)


def test_a_panorama_refuses_a_sequence(tmp_path: Path, built: scene.Built) -> None:
    mount = SCENARIO.mounts[0]
    Calibration(
        cameras=[scene.calibrate(built, mount, f"{mount.name}_{i}.png") for i in (0, 1)]
    ).write(tmp_path)
    with pytest.raises(ValueError, match="a sequence"):
        panorama.panoramas(tmp_path, "cylindrical", "rig")


def test_a_thermal_panorama_is_one_grey_per_temperature(
    tmp_path: Path, built: scene.Built
) -> None:
    first = SCENARIO.mounts[0]
    mounts = [
        m
        for m in SCENARIO.mounts
        if m.rig_name == first.rig_name and m.camera.band == first.camera.band
    ][:2]
    thermal, white = tmp_path / "thermal", tmp_path / "white"
    calibration = Calibration(
        cameras=[scene.calibrate(built, m, f"{m.name}.png") for m in mounts]
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
        cv2.imread(str(panorama.panoramas(f, "cylindrical", "rig", max_width=400)[0]))
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
    frame = np.zeros((mount.camera.height_px, mount.camera.width_px, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    camera = scene.calibrate(built, mount, "frame.png")

    native, _ = panorama.stitch(tmp_path, [camera], "cylindrical", "rig")
    capped, _ = panorama.stitch(tmp_path, [camera], "cylindrical", "rig", 101)
    uncapped, _ = panorama.stitch(tmp_path, [camera], "cylindrical", "rig", 10**6)

    assert capped.shape[1] == 101
    assert uncapped.shape == native.shape


@pytest.mark.parametrize("projection", list(panorama.PROJECTIONS))
def test_a_bearing_lands_where_the_stitch_put_it(
    tmp_path: Path, projection: str, built: scene.Built
) -> None:
    """A dot right of each principal point, found in the stitch, against the column
    its ray's bearing maps to. Off centre, so a mirrored mapping finds no dot; on a
    pitched camera, so the mapping is held independent of elevation."""
    first = SCENARIO.mounts[0]
    mounts = [
        m
        for m in SCENARIO.mounts
        if m.rig_name == first.rig_name and m.camera.band == first.camera.band
    ]
    cameras = [scene.calibrate(built, m, f"{m.name}.png") for m in mounts]
    for camera in cameras:
        frame = np.zeros((camera.height_px, camera.width_px, 3), np.uint8)
        cx, cy = round(camera.K[0][2] + camera.width_px / 4), round(camera.K[1][2])
        frame[cy - 6 : cy + 7, cx - 6 : cx + 7] = 255
        cv2.imwrite(str(tmp_path / camera.image), frame)

    image, layout = panorama.stitch(tmp_path, cameras, projection, "rig", 2000)

    for camera in cameras:
        dot = (camera.K[0][2] + camera.width_px / 4, camera.K[1][2], 1.0)
        ray = np.array(camera.extrinsics["rig"])[:3, :3] @ np.linalg.inv(camera.K) @ dot
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
    frame = np.zeros((mount.camera.height_px, mount.camera.width_px, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    camera = scene.calibrate(built, mount, "frame.png")
    image, layout = panorama.stitch(tmp_path, [camera], "cylindrical", "rig", 1000)
    # The labelled bearing nearest the camera's axis, so it is in frame.
    x, y, _ = np.array(camera.extrinsics["rig"])[:3, 2]
    bearing = panorama.LABEL_DEG * round(
        math.degrees(math.atan2(x, y)) / panorama.LABEL_DEG
    )

    ruled = panorama.ruled(image, layout)

    strip = ruled[image.shape[0] :]
    assert (ruled[: image.shape[0]] == image).all()
    tick = strip[:16, round(panorama.column(layout, bearing) or -1)]
    assert (tick == INK).all()


def test_rectilinear_refuses_a_camera_behind_its_plane(built: scene.Built) -> None:
    """Two cameras back to back: no plane faces both, whatever their field."""
    camera = scene.calibrate(built, SCENARIO.mounts[0], "")
    turned = Matrix.Rotation(math.pi, 4, "Z") @ Matrix(camera.extrinsics["world"])
    behind = camera.model_copy(
        update={"extrinsics": {"world": tuple(map(tuple, turned))}}
    )

    with pytest.raises(ValueError, match="rectilinear cannot show"):
        panorama.stitch(Path(), [camera, behind], "rectilinear", "world")


def test_a_frame_the_calibration_lacks_is_named(built: scene.Built) -> None:
    camera = scene.calibrate(built, SCENARIO.mounts[0], "")

    with pytest.raises(ValueError, match="'deck'"):
        panorama.stitch(Path(), [camera], "cylindrical", "deck")


def test_a_max_width_under_a_pixel_is_an_error(built: scene.Built) -> None:
    camera = scene.calibrate(built, SCENARIO.mounts[0], "")

    with pytest.raises(ValueError, match="max width"):
        panorama.stitch(Path(), [camera], "cylindrical", "rig", 0)


def test_a_shrunk_frame_keeps_its_principal_point_at_its_centre(
    built: scene.Built,
) -> None:
    camera = scene.calibrate(built, SCENARIO.mounts[0], "")
    frame = np.zeros((camera.height_px, camera.width_px, 3), np.uint8)
    k, _ = panorama.pose(camera, "rig", 0.0)

    small, k = panorama._shrink(frame, k, 0.25)

    h, w = small.shape[:2]
    assert (k[0, 2], k[1, 2]) == pytest.approx(((w - 1) / 2, (h - 1) / 2))
    assert k[0, 0] == pytest.approx(camera.K[0][0] * w / camera.width_px)


def _in_ship_frame(built: scene.Built, obj: bpy.types.Object) -> Matrix:
    """The rig is specified on the hull, so it is measured there."""
    return built.vessel.matrix_world.inverted() @ obj.matrix_world


def _in_frame(uv) -> bool:
    """Behind the camera `world_to_camera_view` still lands x, y in range; z decides."""
    return 0.0 <= uv.x <= 1.0 and 0.0 <= uv.y <= 1.0 and uv.z > 0.0
