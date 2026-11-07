"""Laying rendered frames out for review."""

import colorsys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from seascape.config import Camera, Mount, Rig, Scenario
from seascape.outputs import montage
from tests.scenarios import two_rigs

type Rgb = tuple[int, int, int]


def mount(rig: str, camera: str, bearing_deg: float, hfov_deg: float) -> Mount:
    """`camera` alone on `rig`, looking along `bearing_deg`."""
    optics = Camera.model_validate(
        {
            "band": "eo",
            "hfov_deg": hfov_deg,
            "width_px": 640,
            "height_px": 480,
            "yaw_deg": bearing_deg,
        }
    )
    carrier = Rig.model_validate({"height_m": 5.0, "cameras": {camera: optics}})
    return Mount(rig, carrier, camera, optics)


def names(mounts: Sequence[Mount]) -> list[list[str]]:
    return [[m.name for m in column] for column in montage.columns(mounts)]


AHEAD = [
    mount("bow", "eo_p", -40.0, 49.0),
    mount("bow", "eo", 0.0, 49.0),
    mount("bow", "eo_s", 40.0, 49.0),
]


def test_a_frame_joins_the_column_nearest_its_bearing() -> None:
    """At 22°, both eo and eo_s see it; eo_s is nearer."""
    ir = mount("bow", "ir", 22.0, 24.0)

    assert names([*AHEAD, ir]) == [["bow_eo_p"], ["bow_eo"], ["bow_ir", "bow_eo_s"]]


def test_a_frame_no_column_sees_takes_its_own() -> None:
    """Abeam to port, the stern thermal sees what no colour camera does."""
    ir = mount("stern", "ir", -90.0, 24.0)

    assert ["stern_ir"] in names([*AHEAD, ir])


def test_frames_either_side_of_astern_sit_together() -> None:
    """Cut at astern, they would land at opposite ends of the sheet."""
    eo, ir = mount("stern", "eo", 165.0, 49.0), mount("stern", "ir", 195.0, 24.0)

    laid = names([*AHEAD, eo, ir])
    assert abs(laid.index(["stern_eo"]) - laid.index(["stern_ir"])) == 1


@pytest.mark.parametrize(
    "cameras",
    [
        [*AHEAD, mount("bow", "ir", 22.0, 24.0)],
        # Equally wide and in each other's view: either could head the column.
        [
            mount("a", "eo", 0.0, 49.0),
            mount("b", "eo", 20.0, 49.0),
            mount("c", "ir", 40.0, 24.0),
        ],
    ],
)
def test_the_columns_ignore_the_order_cameras_come_in(cameras) -> None:
    assert names(cameras[::-1]) == names(cameras)


@pytest.fixture
def rigs() -> Scenario:
    return two_rigs(outputs={"format": "png"})


def frames(scenario: Scenario, into: Path) -> dict[str, Rgb]:
    """Stand-ins for a render, named as `render` names them, each its own colour and
    a tenth of its camera's size.

    Fully saturated, so no caption's grey text or fan matches one.
    """
    colours: dict[str, Rgb] = {}
    for i, each in enumerate(scenario.mounts):
        r, g, b = colorsys.hsv_to_rgb(i / len(scenario.mounts), 1.0, 1.0)
        colours[each.name] = (round(255 * r), round(255 * g), round(255 * b))
        size = (each.camera.width_px // 10, each.camera.height_px // 10)
        Image.new("RGB", size, colours[each.name]).save(into / f"{each.name}.png")
    return colours


def sheet(scenario: Scenario, into: Path) -> tuple[np.ndarray, dict[str, Rgb]]:
    colours = frames(scenario, into)
    return np.asarray(Image.open(montage.compose(scenario, into))), colours


def span(picture: np.ndarray, colour: Rgb, axis: int) -> np.ndarray:
    """The columns (axis 0) or rows (axis 1) holding `colour`."""
    return np.flatnonzero((picture == colour).all(axis=2).any(axis=axis))


def test_a_band_gets_its_own_row_centred_in_its_column(rigs, tmp_path) -> None:
    picture, colours = sheet(rigs, tmp_path)
    eo, ir = colours["bow_eo"], colours["bow_ir"]

    assert span(picture, eo, 1).max() < span(picture, ir, 1).min()
    centre = span(picture, eo, 0).mean()
    assert span(picture, ir, 0).mean() == pytest.approx(centre, abs=0.5)  # whole pixels


def test_no_frame_covers_another(rigs, tmp_path) -> None:
    """Both thermals look ahead, so they share a column; each shows whole."""
    data = rigs.model_dump()
    data["rigs"]["stern"]["yaw_deg"] = 0.0
    picture, colours = sheet(Scenario.model_validate(data), tmp_path)

    for each in rigs.mounts:
        aspect = each.camera.width_px / each.camera.height_px
        shown = span(picture, colours[each.name], 0).size
        assert shown == round(aspect * montage.TILE_H), each.name


def test_the_caption_sits_under_the_frame_and_never_on_it(rigs, tmp_path) -> None:
    picture, colours = sheet(rigs, tmp_path)
    frame = span(picture, colours["stern_eo"], 0)

    assert (picture[: montage.TILE_H, frame] == colours["stern_eo"]).all()
    assert not (picture[montage.TILE_H :, frame] == colours["stern_eo"]).all(-1).any()


def test_the_caption_fan_lights_the_frames_own_wedge(rigs, tmp_path) -> None:
    """bow_eo_s looks to starboard: its wedge is right of the bow, drawn up."""
    picture, colours = sheet(rigs, tmp_path)
    left = span(picture, colours["bow_eo_s"], 0).min()
    centre = montage.GUTTER + montage.FAN_R
    caption = picture[montage.TILE_H : montage.TILE_H + montage.CAPTION_H, left:]
    fan = caption[:, : 2 * centre]

    rows, cols = np.nonzero((fan == montage.LIT).all(axis=2))
    assert cols.mean() > centre
    assert rows.mean() < montage.CAPTION_H // 2


def test_a_missing_frame_names_itself(rigs, tmp_path) -> None:
    """Silently dropping it gives a montage that looks complete and is not."""
    frames(rigs, tmp_path)
    absent = next(tmp_path.glob("*.png"))
    absent.unlink()

    with pytest.raises(FileNotFoundError, match=absent.stem):
        montage.compose(rigs, tmp_path)
