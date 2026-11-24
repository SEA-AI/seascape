"""Laying rendered frames out for review."""

import colorsys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from seascape.config import Camera, Mount, Rig, Scenario, load
from seascape.outputs import agc, labels, montage
from seascape.render import video
from tests.scenarios import SWEEP_WATCHKEEPER, two_rigs

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


def test_frames_either_side_of_astern_sit_together_in_their_own_columns() -> None:
    """Cut at astern, they would land at opposite ends of the sheet. No colour camera
    ahead sees either.
    """
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


def videos(
    into: Path, names: Sequence[str], times_s: Sequence[float]
) -> dict[str, Rgb]:
    """64 x 48 frames per name, each its own colour, and labels.json timing them."""
    colours, images = {}, []
    for i, name in enumerate(names):
        r, g, b = colorsys.hsv_to_rgb(i / len(names), 1.0, 1.0)
        colours[name] = (round(255 * r), round(255 * g), round(255 * b))
        (into / name).mkdir()
        for time_s in times_s:
            Image.new("RGB", (64, 48), colours[name]).save(
                into / f"{name}/{len(images):04d}.png"
            )
            images.append(
                labels.Image(
                    id=len(images) + 1,
                    file_name=f"{name}/{len(images):04d}.png",
                    width=64,
                    height=48,
                    camera=name,
                    band="eo",
                    hfov_deg=1.0,
                    time_s=time_s,
                    horizon_px=[],
                )
            )
    labels.Labels(images=images).write(into)
    return colours


def played(paths: list[Path]) -> list[np.ndarray]:
    return [np.asarray(Image.open(path).convert("RGB")) for path in paths]


def showing(picture: np.ndarray, colour: Rgb) -> np.ndarray:
    """Where `picture` shows `colour`."""
    return (picture == colour).all(axis=2)


def test_the_preview_video_lays_out_every_camera_as_the_sheet_does(
    rigs, tmp_path
) -> None:
    colours = videos(tmp_path, [m.name for m in rigs.mounts], [0.0, 0.1, 0.2])

    shown = played(montage.video_frames(rigs, tmp_path, tmp_path / "out"))

    assert len(shown) == 3
    rows = {
        n: np.flatnonzero(showing(shown[0], c).any(axis=1)) for n, c in colours.items()
    }
    assert rows["bow_eo"].max() < rows["bow_ir"].min()  # a row per band
    assert rows["bow_eo"].size == pytest.approx(montage.TILE_H, abs=2)


def test_a_thermal_clip_keeps_its_tone_from_frame_to_frame(rigs, tmp_path) -> None:
    """Its AGC eases toward each frame: a frame after a hotter one shows apart from
    the same frame after itself.
    """
    ramp = np.linspace(280.0, 300.0, 64) * np.ones((48, 1))
    shown = []
    for name, first in (("after_hotter", ramp + 30.0), ("after_itself", ramp)):
        into = tmp_path / name
        into.mkdir()
        videos(into, [m.name for m in rigs.mounts], [0.0, 0.1])
        # Each camera's frames are numbered in turn: its first is the even one.
        for path in into.glob("*/*.png"):
            t_k = first if int(path.stem) % 2 == 0 else ramp
            cv2.imwrite(str(path), agc.counts(t_k))
        shown.append(played(montage.video_frames(rigs, into, into / "out"))[1])

    after_hotter, after_itself = shown
    assert not np.array_equal(after_hotter, after_itself)


@pytest.mark.parametrize("video", ["mast_RGB1", "mast_T1"])
def test_the_preview_video_shows_a_rig_s_videos_not_its_cameras(
    tmp_path, video: str
) -> None:
    """Watchkeeper records an undistortion of rgb1 and a stitch of t2 and t1: one
    tile each.
    """
    colours = videos(tmp_path, ["mast_RGB1", "mast_T1"], [0.0, 0.1])

    out = montage.video_frames(load(SWEEP_WATCHKEEPER), tmp_path, tmp_path / "out")

    columns = np.flatnonzero(showing(played(out)[0], colours[video]).any(axis=0))
    assert columns.size == columns.max() - columns.min() + 1


def test_preview_frames_have_even_sides_for_h264(
    rigs, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(montage, "_sheet", lambda *_: Image.new("RGB", (63, 47)))
    videos(tmp_path, [m.name for m in rigs.mounts], [0.0])

    (frame,) = montage.video_frames(rigs, tmp_path, tmp_path / "out")

    assert Image.open(frame).size == (64, 48)


def test_a_render_without_frames_has_no_preview_video(rigs, tmp_path) -> None:
    labels.Labels(images=[]).write(tmp_path)

    with pytest.raises(FileNotFoundError, match="no frames"):
        montage.video_frames(rigs, tmp_path, tmp_path / "out")


def test_the_preview_video_plays_a_sheet_per_frame(rigs, tmp_path) -> None:
    videos(tmp_path, [m.name for m in rigs.mounts], [0.0, 0.1, 0.2])

    out = video.montage_video(rigs, tmp_path, tmp_path / "montage.mp4")

    capture, count = cv2.VideoCapture(str(out)), 0
    while capture.read()[0]:
        count += 1
    assert count == 3
