"""Laying rendered frames out for review."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seascape.config import Scenario
from seascape.outputs import montage
from tests.scenarios import two_rigs

FLAT = (90, 110, 130)
MARKS = {
    "bow_eo_s": (255, 0, 0),
    "bow_ir": (0, 255, 0),
    "stern_eo": (0, 0, 255),
    "stern_ir": (255, 0, 255),
}


@pytest.fixture
def rigs() -> Scenario:
    return two_rigs(outputs={"format": "png"})


def frames(scenario: Scenario, into: Path) -> Path:
    """Stand-ins for a render, one per camera, named as `render` names them."""
    into.mkdir(parents=True, exist_ok=True)
    for mount in scenario.mounts:
        colour = MARKS.get(mount.name, FLAT)
        Image.new("RGB", (64, 36), colour).save(into / f"{mount.name}.png")
    return into


def sheet(scenario: Scenario, into: Path) -> np.ndarray:
    return np.asarray(Image.open(montage.compose(scenario, frames(scenario, into))))


def turned(scenario: Scenario, rig: str, yaw_deg: float, **cameras: float) -> Scenario:
    """`scenario` with `rig` and its `cameras` turned, validated again."""
    data = scenario.model_dump()
    data["rigs"][rig]["yaw_deg"] = yaw_deg
    for name, camera_yaw_deg in cameras.items():
        data["rigs"][rig]["cameras"][name]["yaw_deg"] = camera_yaw_deg
    return Scenario.model_validate(data)


def span(picture: np.ndarray, colour: tuple[int, int, int], axis: int) -> np.ndarray:
    """The columns (axis 0) or rows (axis 1) holding `colour`."""
    return np.flatnonzero((picture == colour).all(axis=2).any(axis=axis))


def test_a_band_gets_its_own_row(rigs, tmp_path) -> None:
    picture = sheet(rigs, tmp_path)

    eo, ir = MARKS["bow_eo_s"], MARKS["bow_ir"]
    assert span(picture, eo, 1).max() < span(picture, ir, 1).min()


def test_the_caption_sits_under_the_frame_and_never_on_it(rigs, tmp_path) -> None:
    picture = sheet(rigs, tmp_path)
    frame = span(picture, MARKS["stern_eo"], 0)

    assert (picture[: montage.TILE_H, frame] == MARKS["stern_eo"]).all()
    assert not (picture[montage.TILE_H :, frame] == MARKS["stern_eo"]).all(-1).any()


def test_a_frame_sits_under_the_camera_that_sees_its_bearing(rigs, tmp_path) -> None:
    picture = sheet(turned(rigs, "bow", 0.0, ir=35.0), tmp_path)

    under = span(picture, MARKS["bow_eo_s"], 0)
    assert span(picture, MARKS["bow_ir"], 0).mean() == pytest.approx(under.mean())


def test_frames_either_side_of_astern_sit_together(rigs, tmp_path) -> None:
    """Cut at astern, they would land at opposite ends of the sheet."""
    picture = sheet(turned(rigs, "stern", 180.0, eo=-15.0, ir=15.0), tmp_path)
    eo, ir = span(picture, MARKS["stern_eo"], 0), span(picture, MARKS["stern_ir"], 0)

    assert 0 < ir.min() - eo.max() <= montage.GUTTER + 1


def test_frames_sharing_a_bearing_sit_side_by_side(rigs, tmp_path) -> None:
    """Covering one with the other would hide a whole camera."""
    picture = sheet(turned(rigs, "stern", 0.0), tmp_path)
    bow, stern = span(picture, MARKS["bow_ir"], 0), span(picture, MARKS["stern_ir"], 0)

    assert bow.size == stern.size
    assert bow.max() < stern.min()


def test_a_frame_no_column_sees_takes_its_own(rigs, tmp_path) -> None:
    """Abeam to port, the stern thermal sees what no colour camera does."""
    picture = sheet(turned(rigs, "stern", 180.0, ir=90.0), tmp_path)
    alone = span(picture, MARKS["stern_ir"], 0)

    assert (picture[: montage.TILE_H, alone] == montage.MATTE).all()


def test_the_caption_fan_lights_the_frames_own_wedge(rigs, tmp_path) -> None:
    """bow_eo_s looks to starboard: its wedge is right of the bow, drawn up."""
    picture = sheet(rigs, tmp_path)
    left = span(picture, MARKS["bow_eo_s"], 0).min()
    centre = montage.GUTTER + montage.FAN_R
    caption = picture[montage.TILE_H : montage.TILE_H + montage.CAPTION_H, left:]
    fan = caption[:, : 2 * centre]

    rows, cols = np.nonzero((fan == montage.LIT).all(axis=2))
    assert cols.mean() > centre
    assert rows.mean() < montage.CAPTION_H // 2


def test_a_missing_frame_names_itself(rigs, tmp_path) -> None:
    """Silently dropping it gives a montage that looks complete and is not."""
    into = frames(rigs, tmp_path)
    absent = next(into.glob("*.png"))
    absent.unlink()

    with pytest.raises(FileNotFoundError, match=absent.stem):
        montage.compose(rigs, into)
