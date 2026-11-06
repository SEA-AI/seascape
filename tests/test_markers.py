"""ArUco markers, read off a render."""

from pathlib import Path

import bpy
import cv2
import numpy as np
import pytest

from seascape import render
from seascape.blender import markers
from seascape.config import ARUCO_DICTIONARY, Agc, Object
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
