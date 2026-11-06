"""ArUco markers, read off a render."""

import math
from itertools import chain
from pathlib import Path

import bpy
import cv2
import numpy as np
import pytest
from mathutils import Vector

from seascape import render
from seascape.blender import markers, scene
from seascape.config import ARUCO_DICTIONARY, MARKER_M, Agc, Object
from seascape.outputs import agc
from seascape.outputs.calibration import Calibration
from tests.scenarios import target, variant

SCENARIO = variant(
    rig={"height_m": 2.5},
    objects=[
        target("container", 51.0, -3.0, steady=True, aruco_id=1),
        target("container", 81.0, 3.0, steady=True, aruco_id=2),
    ],
    outputs={"format": "png", "samples": {"eo": 4, "ir": 4}},
)


def _corners_m(aruco_id: int) -> np.ndarray:
    """The marker's outer corners in the world, in ArUco's order: top left first,
    clockwise as seen.
    """
    (panel,) = (o for o in bpy.data.objects if o.get("aruco_id") == aruco_id)
    margin = markers.BITS + 2 + 2 * markers.QUIET
    # A grid of corners, row by row from the top left as seen.
    last = margin * (margin + 1)
    quad = np.array(
        [
            panel.matrix_world @ panel.data.vertices[i].co
            for i in (0, margin, last + margin, last)
        ]
    )
    centre = quad.mean(axis=0)
    return centre + (quad - centre) * (markers.BITS + 2) / margin


def _project(camera, world_m: np.ndarray) -> np.ndarray:
    """Pixels, centres at integers as OpenCV has them."""
    seen = np.linalg.inv(camera.extrinsics["world"]) @ np.c_[world_m, np.ones(4)].T
    return (np.array(camera.K) @ (seen[:3] / seen[2]))[:2].T


def _cells(corners_px: np.ndarray, aruco_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Each inner module's centre, the light ones then the dark."""
    side = markers.BITS + 2
    to_px = cv2.getPerspectiveTransform(
        np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32),
        corners_px.astype(np.float32),
    )
    image = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY), aruco_id, side
    )
    inner = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    px = cv2.perspectiveTransform(
        np.array([[[c + 0.5, r + 0.5] for r, c in inner]], np.float32), to_px
    )[0]
    light = np.array([image[r, c] > 127 for r, c in inner])
    return px[light], px[~light]


def _detected(grey: np.ndarray) -> dict[int, np.ndarray]:
    """Each marker the frame shows, by id: its corners in pixels, to subpixel."""
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    quads, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(grey)
    if ids is None:
        return {}
    return {int(i): q.reshape(4, 2) for q, i in zip(quads, ids.ravel(), strict=True)}


@pytest.mark.render
def test_every_camera_reads_every_marker_where_it_stands(tmp_path: Path) -> None:
    render.render(SCENARIO, tmp_path)

    for camera in Calibration.read(tmp_path).cameras:
        path = tmp_path / camera.image
        t_k = agc.kelvin(path)
        if t_k is None:
            frame = cv2.imread(str(path))
            assert frame is not None, path
            value = grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            setting = Agc()
            value, grey = t_k, agc.grey(t_k, agc.transfer([t_k], setting), setting)
        seen = _detected(grey)
        for aruco_id in (1, 2):
            expected = _project(camera, _corners_m(aruco_id))
            found = seen.get(aruco_id)
            assert found is not None, (camera.name, aruco_id)
            assert np.abs(found - expected).max() <= 1.0, (camera.name, aruco_id)
            light, dark = (
                np.mean([value[round(y), round(x)] for x, y in cells])
                for cells in _cells(expected, aruco_id)
            )
            assert light - dark >= (markers.CONTRAST_K if t_k is not None else 100), (
                camera.name,
                aruco_id,
            )


def test_aruco_id_spans_the_dictionary() -> None:
    bound = Object.model_json_schema()["properties"]["aruco_id"]["anyOf"][0]
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    assert bound["exclusiveMaximum"] == len(dictionary.bytesList)


SYNC = variant(
    # Pitched down, so each marker stands on its camera's centre ray; held level, so
    # the camera stays on it.
    rig={"height_m": 2.5, "pitch_deg": -1.5},
    ownship={"steady": True},
    cameras={
        "eo": {"hfov_deg": 30.0, "width_px": 320, "height_px": 240},
        # Its marker stands inside eo's view, which must not show it.
        "eo_s": {
            "band": "eo",
            "hfov_deg": 15.0,
            "width_px": 320,
            "height_px": 240,
            "yaw_deg": 10.0,
        },
    },
    sync={"marker_px": 60.0},
    outputs={"format": "png", "bands": ["eo"], "samples": {"eo": 4}},
)


@pytest.mark.render
def test_sync_stands_each_camera_s_marker_where_it_sees_it_at_its_size(
    tmp_path: Path,
) -> None:
    render.render(SYNC, tmp_path)

    ids = {mount.name: aruco_id for aruco_id, mount in enumerate(SYNC.mounts)}
    for camera in Calibration.read(tmp_path).cameras:
        frame = cv2.imread(str(tmp_path / camera.image), cv2.IMREAD_GRAYSCALE)
        assert frame is not None, camera.image
        seen = _detected(frame)
        assert list(seen) == [ids[camera.name]], camera.name
        found = seen[ids[camera.name]]
        side_px = np.linalg.norm(found - np.roll(found, 1, axis=0), axis=1).mean()
        assert side_px == pytest.approx(60.0, rel=0.05), camera.name
        centre_px = ((camera.width_px - 1) / 2, (camera.height_px - 1) / 2)
        assert found.mean(axis=0) == pytest.approx(centre_px, abs=1.0), camera.name


def test_sync_stands_each_marker_where_its_built_camera_points() -> None:
    # Off the ownship's origin and turned, so neither the origin nor the nominal
    # bearing would do.
    scenario = variant(
        rig={"height_m": 2.5, "offset_x_m": 8.0, "yaw_deg": 30.0, "pitch_deg": -4.0},
        ownship={"steady": True},
        cameras={
            "eo_s": {
                "band": "eo",
                "hfov_deg": 15.0,
                "width_px": 320,
                "height_px": 240,
                "yaw_deg": -40.0,
            }
        },
        sync={"marker_px": 60.0},
    )
    built = scene.build(scenario)

    panels = {o["aruco_id"]: o for o in bpy.data.objects if "aruco_id" in o}
    margined = (markers.BITS + 2 + 2 * markers.QUIET) / (markers.BITS + 2)
    for aruco_id, mount in enumerate(scenario.mounts):
        if mount.camera.band != "eo":
            continue
        camera = built.cameras[mount.name].matrix_world
        forward = camera.to_3x3() @ Vector((0.0, 0.0, -1.0))
        panel = panels[aruco_id]
        corners = np.array([panel.matrix_world @ v.co for v in panel.data.vertices])
        toward = Vector(corners.mean(axis=0)) - camera.translation
        assert math.degrees(forward.angle(toward)) == pytest.approx(0.0, abs=0.01), (
            mount.name
        )
        side_m = np.ptp(corners[:, 2]) / margined
        half_rad = math.radians(mount.camera.hfov_deg) / 2
        pixel_rad = 2 * math.tan(half_rad) / mount.camera.width_px
        assert side_m / toward.length == pytest.approx(60.0 * pixel_rad, rel=1e-3), (
            mount.name
        )


@pytest.mark.parametrize("elevation_rad", [0.0, 0.1])
def test_a_marker_that_cannot_be_centred_stands_at_its_widest(
    elevation_rad: float,
) -> None:
    out_m, side_m = markers.on_ray(2.5, elevation_rad, 0.74, 0.001)
    assert side_m == pytest.approx(MARKER_M)
    assert out_m == pytest.approx(MARKER_M * math.cos(elevation_rad) / 0.001)


def test_an_eye_below_its_marker_is_an_error() -> None:
    with pytest.raises(ValueError, match="raise the rig"):
        markers.on_ray(0.5, -0.1, 0.74, 0.01)


def test_a_camera_renders_only_its_own_marker_and_every_shared_object() -> None:
    scenario = SYNC.model_copy(
        update={"objects": [*SYNC.objects, target("container", 120.0, -20.0)]}
    )
    built = scene.build(scenario)

    for mount in scenario.band_mounts("eo"):
        scene.look_through(built.cameras, mount, np.random.default_rng(0))
        for anchor in chain(*built.targets.values()):
            shown = anchor.get("camera", mount.name) == mount.name
            for part in [anchor, *anchor.children_recursive]:
                assert part.hide_render is not shown, (mount.name, part.name)
