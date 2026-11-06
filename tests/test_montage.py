"""Laying rendered frames out for review."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seascape.config import Scenario
from seascape.outputs import montage
from tests.scenarios import two_rigs


@pytest.fixture
def rigs() -> Scenario:
    return two_rigs(outputs={"format": "png"})


def frames(scenario: Scenario, into: Path, size: tuple[int, int] = (64, 36)) -> Path:
    """Stand-ins for a render, one per camera, named as `render` names them."""
    into.mkdir(parents=True, exist_ok=True)
    for mount in scenario.mounts:
        Image.new("RGB", size, (90, 110, 130)).save(into / f"{mount.name}.png")
    return into


MARKS = {"stern_eo": (255, 0, 0), "stern_ir": (0, 255, 0), "bow_ir": (0, 0, 255)}


def marked(scenario: Scenario, into: Path) -> np.ndarray:
    """The sheet, with the frames in MARKS in their own colours."""
    frames(scenario, into)
    for name, colour in MARKS.items():
        size = Image.open(into / f"{name}.png").size
        Image.new("RGB", size, colour).save(into / f"{name}.png")
    return np.asarray(Image.open(montage.compose(scenario, into)))


def turned(scenario: Scenario, rig: str, yaw_deg: float, **cameras: float) -> Scenario:
    """`scenario` with `rig` and its `cameras` turned, validated again."""
    data = scenario.model_dump()
    data["rigs"][rig]["yaw_deg"] = yaw_deg
    for name, camera_yaw_deg in cameras.items():
        data["rigs"][rig]["cameras"][name]["yaw_deg"] = camera_yaw_deg
    return Scenario.model_validate(data)


def span(sheet: np.ndarray, name: str, axis: int) -> np.ndarray:
    """The columns (axis 0) or rows (axis 1) a marked frame covers."""
    return np.flatnonzero((sheet == MARKS[name]).all(axis=2).any(axis=axis))


def test_a_band_gets_its_own_row(rigs, tmp_path) -> None:
    sheet = marked(rigs, tmp_path)

    assert span(sheet, "stern_eo", 1).max() < span(sheet, "stern_ir", 1).min()


def test_the_caption_sits_under_the_frame_and_never_on_it(rigs, tmp_path) -> None:
    into = frames(rigs, tmp_path)
    flat = (90, 110, 130)

    sheet = Image.open(montage.compose(rigs, into))

    assert sheet.getpixel((0, montage.TILE_H // 2)) == flat  # frame, untouched
    assert sheet.getpixel((0, montage.TILE_H + 2)) == montage.MATTE  # caption band


def test_a_frame_sits_under_its_bearing_as_wide_as_it_sees(rigs, tmp_path) -> None:
    """The thermal looking astern sits under the colour camera looking astern, as
    wide as its share of that camera's hfov.
    """
    sheet = marked(rigs, tmp_path)
    eo, ir = span(sheet, "stern_eo", 0), span(sheet, "stern_ir", 0)
    hfov = {name: c.hfov_deg for name, c in rigs.rigs["stern"].cameras.items()}

    assert ir.mean() == pytest.approx(eo.mean(), abs=1)
    assert ir.size == pytest.approx(eo.size * hfov["ir"] / hfov["eo"], abs=1)


def test_frames_either_side_of_astern_sit_together(rigs, tmp_path) -> None:
    """Cut at astern, they would land at opposite ends of the sheet."""
    sheet = marked(turned(rigs, "stern", 180.0, eo=-15.0, ir=15.0), tmp_path)
    eo, ir = span(sheet, "stern_eo", 0), span(sheet, "stern_ir", 0)
    px_per_deg = eo.size / rigs.rigs["stern"].cameras["eo"].hfov_deg

    assert ir.mean() - eo.mean() == pytest.approx(30.0 * px_per_deg, abs=2)


def test_frames_sharing_a_bearing_sit_side_by_side(rigs, tmp_path) -> None:
    """Covering one with the other would hide a whole camera."""
    sheet = marked(turned(rigs, "stern", 0.0), tmp_path)
    bow, stern = span(sheet, "bow_ir", 0), span(sheet, "stern_ir", 0)

    assert bow.size == stern.size
    assert bow.max() < stern.min()


def test_a_missing_frame_names_itself(rigs, tmp_path) -> None:
    """Silently dropping it gives a montage that looks complete and is not."""
    into = frames(rigs, tmp_path)
    absent = next(into.glob("*.png"))
    absent.unlink()

    with pytest.raises(FileNotFoundError, match=absent.stem):
        montage.compose(rigs, into)
