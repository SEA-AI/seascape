import numpy as np
import pytest
from PIL import Image

from seascape.outputs import preview
from tests.scenarios import two_rigs


def frame(grey: int, size: tuple[int, int] = (1000, 600)) -> Image.Image:
    return Image.new("RGB", size, (grey, grey, grey))


def mark_pixels(marked: Image.Image | np.ndarray, grey: int) -> np.ndarray:
    """Where `marked` differs from the flat frame under the mark."""
    return np.asarray(marked).astype(int)[..., 0] != grey


def test_the_mark_is_white_on_dark() -> None:
    marked = np.asarray(preview.watermark(frame(20)))
    assert marked[mark_pixels(marked, 20)].max() > 20


def test_the_mark_is_black_on_light() -> None:
    marked = np.asarray(preview.watermark(frame(230)))
    assert marked[mark_pixels(marked, 230)].min() < 230


def test_the_mark_lets_the_picture_show_through() -> None:
    marked = np.asarray(preview.watermark(frame(20)))
    assert 20 < marked[mark_pixels(marked, 20)].max() < 255


def test_a_strip_below_holds_the_mark_and_leaves_the_picture_alone() -> None:
    picture = frame(20)
    marked = np.asarray(preview.watermark(picture, below=True))
    assert marked.shape[1] == picture.width and marked.shape[0] > picture.height
    assert (marked[: picture.height] == 20).all()
    assert mark_pixels(marked[picture.height :], 24).any()


def test_a_frame_smaller_than_the_mark_still_takes_it() -> None:
    assert mark_pixels(preview.watermark(frame(20, (100, 80))), 20).any()


def test_the_mark_sits_bottom_right_clear_of_the_edge() -> None:
    marked = preview.watermark(frame(20))
    rows, cols = np.nonzero(mark_pixels(marked, 20))
    height = round(preview.MARK_SHARE * 600)
    assert rows.max() - rows.min() + 1 == height
    # The Brand Book's clear space is the cursor's width: a share of the mark's.
    assert marked.height - 1 - rows.max() == pytest.approx(
        preview.CURSOR_SHARE * (cols.max() - cols.min() + 1), abs=3
    )
    assert cols.min() > marked.width / 2 and rows.min() > marked.height / 2


def test_the_mark_never_shrinks_below_the_brand_minimum() -> None:
    rows = np.nonzero(mark_pixels(preview.watermark(frame(20, (400, 300))), 20))[0]
    assert rows.max() - rows.min() + 1 == preview.MIN_MARK_PX


@pytest.mark.parametrize("colour", ["white", "black"])
def test_the_cursor_share_matches_the_logo_files(colour: str) -> None:
    alpha = np.asarray(preview._logo(colour))[..., 3] > 0
    columns = alpha.any(axis=0)
    # The cursor is the last opaque run of columns.
    last = np.flatnonzero(~columns)[-1] + 1
    assert (len(columns) - last) / len(columns) == pytest.approx(
        preview.CURSOR_SHARE, abs=0.002
    )


def test_a_clip_shows_each_cameras_middle_frame(tmp_path) -> None:
    scenario = two_rigs(outputs={"format": "png", "duration_s": 0.3, "fps": 10})
    for mount in scenario.mounts:
        (tmp_path / mount.name).mkdir()
        for n in range(3):
            (tmp_path / mount.name / f"{n:04d}.png").touch()
    shown = preview.shown(scenario, tmp_path)
    assert {m.name for m, _ in shown} == {m.name for m in scenario.mounts}
    assert all(path.name == "0001.png" for _, path in shown)


def test_a_clip_camera_with_no_frames_names_its_folder(tmp_path) -> None:
    scenario = two_rigs(outputs={"format": "png", "duration_s": 0.3, "fps": 10})
    first = scenario.mounts[0].name
    (tmp_path / first).mkdir()
    with pytest.raises(FileNotFoundError, match=first):
        preview.shown(scenario, tmp_path)
