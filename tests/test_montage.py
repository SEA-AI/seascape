"""Laying rendered frames out for review."""

from pathlib import Path

import pytest
from PIL import Image

from seascape import montage
from seascape.config import Scenario
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


def test_a_montage_carries_every_camera(rigs, tmp_path) -> None:
    into = frames(rigs, tmp_path)

    out = montage.compose(rigs, into)

    assert out == into / "montage.png"
    assert Image.open(out).size[0] > 0


def test_a_band_gets_its_own_row(rigs, tmp_path) -> None:
    into = frames(rigs, tmp_path)

    height = Image.open(montage.compose(rigs, into)).height

    rows = len({mount.camera.band for mount in rigs.mounts})
    assert height == rows * (montage.TILE_H + montage.CAPTION_H) + montage.GUTTER


def test_the_caption_sits_under_the_frame_and_never_on_it(rigs, tmp_path) -> None:
    into = frames(rigs, tmp_path)
    flat = (90, 110, 130)

    sheet = Image.open(montage.compose(rigs, into))

    assert sheet.getpixel((0, montage.TILE_H // 2)) == flat  # frame, untouched
    assert sheet.getpixel((0, montage.TILE_H + 2)) == montage.MATTE  # caption band


def test_a_missing_frame_names_itself(rigs, tmp_path) -> None:
    """Silently dropping it gives a montage that looks complete and is not."""
    into = frames(rigs, tmp_path)
    absent = next(into.glob("*.png"))
    absent.unlink()

    with pytest.raises(FileNotFoundError, match=absent.stem):
        montage.compose(rigs, into)


def test_an_unrendered_scenario_is_an_error(rigs, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        montage.compose(rigs, tmp_path)
