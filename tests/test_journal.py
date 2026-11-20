"""truth.jsonl: a render's truth appended a frame at a time, without a render."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from seascape.outputs import journal, labels
from seascape.outputs.calibration import Calibration, CameraCalibration
from seascape.physics import earth

INFO = {"scenario": {"seed": 7}}
# Facing north, level, 50 m up; the pose that puts a horizon in its frame.
CAMERA = CameraCalibration(
    name="C",
    band="eo",
    image="C.png",
    rig="bow",
    width_px=64,
    height_px=48,
    K=((70.0, 0.0, 31.5), (0.0, 70.0, 23.5), (0.0, 0.0, 1.0)),
    extrinsics={
        "world": (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 50.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    },
)


def _target(pass_index: int) -> labels.Target:
    return labels.Target(
        pass_index=pass_index,
        name=f"t{pass_index}",
        category_id=1,
        category="ship",
        supercategory="vessel",
        centre_m=(0.0, 1000.0),
        waterline_m=np.array([[0.0, 990.0], [5.0, 1010.0]]),
        heading_deg=90.0,
        dims_m=(20.0, 5.0, 4.0),
    )


def _label(truth: labels.Labels, time_s: float) -> None:
    """One frame where t1 is seen and t2 is not, so it adds to images, annotations,
    dropped and, the first time, categories.
    """
    index = np.zeros((48, 64), dtype=int)
    index[20:24, 30:34] = 1
    truth.add(
        CAMERA,
        time_s,
        index,
        np.ones((48, 64, 3)),
        [_target(1), _target(2)],
        earth.earth_radius_m(0.13),
        faint={},
    )


def _leftover(folder: Path, frames: int) -> labels.Labels:
    """The truth of `frames` frames, and in `folder` the journal a render killed
    outright leaves of them: a `Journal` whose block never ended.
    """
    truth = labels.Labels(info=INFO)
    appended = journal.Journal(folder, truth)
    for frame in range(frames):
        _label(truth, frame / 10)
        appended.append([CAMERA], [CAMERA] if frame == 0 else [])
    appended._file.close()  # as the kill does
    return truth


def test_the_journal_reads_back_as_the_truth_it_was_appended_from(
    tmp_path: Path,
) -> None:
    truth = _leftover(tmp_path, 2)

    calibration, read = journal.read(tmp_path)

    assert read == truth
    assert calibration == Calibration(cameras=[CAMERA] * 2, raw=[CAMERA])


def test_only_the_first_frame_s_line_holds_the_info(tmp_path: Path) -> None:
    _leftover(tmp_path, 2)
    lines = (tmp_path / journal.FILENAME).read_text().splitlines()

    assert [journal.Step.model_validate_json(line).info for line in lines] == [INFO, {}]


def test_a_torn_last_line_is_left_out(tmp_path: Path) -> None:
    """What a render killed mid-write leaves, here inside a two-byte character."""
    _leftover(tmp_path, 2)
    path = tmp_path / journal.FILENAME
    path.write_bytes(path.read_bytes() + '{"info": {"sea": "ö'.encode()[:-1])

    _, read = journal.read(tmp_path)

    assert len(read.images) == 2


def test_text_any_locale_or_line_separator_would_break_reads_back(
    tmp_path: Path,
) -> None:
    truth = labels.Labels(info={"note": "海\u2028sea"})

    with journal.Journal(tmp_path, truth) as appended:
        _label(truth, 0.0)
        appended.append([CAMERA], [])
        _, read = journal.read(tmp_path)

    assert read == truth


def test_a_torn_line_before_the_last_is_an_error(tmp_path: Path) -> None:
    _leftover(tmp_path, 1)
    path = tmp_path / journal.FILENAME
    path.write_text('{"cameras": [{"na\n' + path.read_text())

    with pytest.raises(ValidationError):
        journal.read(tmp_path)


def test_finish_writes_both_files_and_deletes_the_journal(tmp_path: Path) -> None:
    truth = _leftover(tmp_path, 2)

    written = journal.finish(tmp_path)

    assert written == [tmp_path / "calibration.json", tmp_path / "labels.json"]
    assert sorted(tmp_path.iterdir()) == written
    assert labels.Labels.read(tmp_path) == truth
    assert len(Calibration.read(tmp_path).cameras) == 2


def test_a_journal_without_a_frame_writes_nothing(tmp_path: Path) -> None:
    _leftover(tmp_path, 0)

    written = journal.finish(tmp_path)

    assert written == []
    assert not any(tmp_path.iterdir())


def test_a_block_that_raises_still_writes_the_frames_it_finished(
    tmp_path: Path,
) -> None:
    truth = labels.Labels(info=INFO)

    with pytest.raises(RuntimeError), journal.Journal(tmp_path, truth) as appended:
        _label(truth, 0.0)
        appended.append([CAMERA], [])
        raise RuntimeError("the GPU died")

    assert len(labels.Labels.read(tmp_path).images) == 1
    assert not (tmp_path / journal.FILENAME).exists()


def test_a_journal_that_cannot_finish_notes_it_on_the_blocks_own_error(
    tmp_path: Path,
) -> None:
    with (
        pytest.raises(RuntimeError, match="the GPU died") as raised,
        journal.Journal(tmp_path, labels.Labels()),
    ):
        (tmp_path / journal.FILENAME).unlink()
        raise RuntimeError("the GPU died")

    assert "left unfinished" in raised.value.__notes__[0]


def test_its_block_ending_writes_the_truth_over_an_earlier_render_s_journal(
    tmp_path: Path,
) -> None:
    _leftover(tmp_path, 2)
    truth = labels.Labels(info=INFO)

    with journal.Journal(tmp_path, truth) as appended:
        _label(truth, 0.0)
        appended.append([CAMERA], [])

    assert appended.written == [tmp_path / "calibration.json", tmp_path / "labels.json"]
    assert labels.Labels.read(tmp_path) == truth
