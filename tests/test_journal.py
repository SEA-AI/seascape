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


def _label(
    truth: labels.Labels, time_s: float, camera: CameraCalibration = CAMERA
) -> None:
    """One frame where t1 is seen and t2 is not, so it adds to images, annotations,
    dropped and, the first time, categories.
    """
    index = np.zeros((48, 64), dtype=int)
    index[20:24, 30:34] = 1
    truth.add(
        camera,
        time_s,
        index,
        np.ones((48, 64, 3)),
        [_target(1), _target(2)],
        earth.earth_radius_m(0.13),
        faint={},
    )


def _leftover(folder: Path, frames: int) -> labels.Labels:
    """The truth of `frames` frames, and in `folder` the journal a render killed
    outright leaves of them: the lines `Journal` writes, never finished.
    """
    truth = labels.Labels(info=INFO)
    steps = []
    for frame in range(frames):
        before = len(truth.images), len(truth.annotations), len(truth.dropped)
        _label(truth, frame / 10)
        images, annotations, dropped = before
        steps.append(
            journal.Step(
                info=INFO if frame == 0 else {},
                cameras=[CAMERA],
                raw=[CAMERA] if frame == 0 else [],
                images=truth.images[images:],
                annotations=truth.annotations[annotations:],
                dropped=truth.dropped[dropped:],
                categories=truth.categories if frame == 0 else [],
            )
        )
    (folder / journal.FILENAME).write_text(
        "".join(step.model_dump_json() + "\n" for step in steps)
    )
    return truth


def test_the_journal_reads_back_as_the_truth_it_was_appended_from(
    tmp_path: Path,
) -> None:
    truth = labels.Labels(info=INFO)

    with journal.Journal(tmp_path, truth) as appended:
        _label(truth, 0.0)
        appended.append([CAMERA], [CAMERA])
        _label(truth, 0.1)
        appended.append([CAMERA], [])
        calibration, read = journal.read(tmp_path)

    assert read == truth
    assert calibration == Calibration(cameras=[CAMERA] * 2, raw=[CAMERA])


def test_each_frame_is_one_line_holding_only_what_it_added(tmp_path: Path) -> None:
    truth = labels.Labels(info=INFO)

    with journal.Journal(tmp_path, truth) as appended:
        _label(truth, 0.0)
        appended.append([CAMERA], [])
        _label(truth, 0.1)
        appended.append([CAMERA], [])
        lines = (tmp_path / journal.FILENAME).read_text().splitlines()
    steps = [journal.Step.model_validate_json(line) for line in lines]

    assert [len(s.images) for s in steps] == [1, 1]
    assert [len(s.annotations) for s in steps] == [1, 1]
    assert [len(s.dropped) for s in steps] == [1, 1]
    assert [len(s.categories) for s in steps] == [1, 0]
    assert [s.info for s in steps] == [INFO, {}]


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


def test_its_block_ending_writes_the_truth(tmp_path: Path) -> None:
    truth = labels.Labels(info=INFO)

    with journal.Journal(tmp_path, truth) as appended:
        _label(truth, 0.0)
        appended.append([CAMERA], [])

    assert appended.written == [tmp_path / "calibration.json", tmp_path / "labels.json"]
    assert labels.Labels.read(tmp_path) == truth


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


def test_a_journal_left_by_an_earlier_render_is_replaced(tmp_path: Path) -> None:
    _leftover(tmp_path, 2)
    truth = labels.Labels(info=INFO)

    with journal.Journal(tmp_path, truth) as appended:
        _label(truth, 0.0)
        appended.append([CAMERA], [])

    assert len(labels.Labels.read(tmp_path).images) == 1


def _part(folder: Path, name: str) -> None:
    """A finished render of camera `name` in `folder`, as `--parallel` leaves one
    band's: its frame, truth and log.
    """
    camera = CAMERA.model_copy(update={"name": name, "image": f"{name}.png"})
    truth = labels.Labels(info=INFO)
    _label(truth, 0.0, camera)
    folder.mkdir()
    truth.write(folder)
    Calibration(cameras=[camera]).write(folder)
    (folder / camera.image).write_bytes(b"frame")
    (folder / "blender.log").write_text(f"{name}\n")


def test_gathered_parts_read_as_one_render_of_the_scene(tmp_path: Path) -> None:
    parts = [tmp_path / ".eo", tmp_path / ".ir"]
    _part(parts[0], "A")
    _part(parts[1], "B")

    written = journal.gather(tmp_path, parts, {"seed": 1})

    truth = labels.Labels.read(tmp_path)
    assert [(i.id, i.file_name) for i in truth.images] == [(1, "A.png"), (2, "B.png")]
    assert [(a.id, a.image_id) for a in truth.annotations] == [(1, 1), (2, 2)]
    assert truth.info["scenario"] == {"seed": 1}
    assert [c.name for c in Calibration.read(tmp_path).cameras] == ["A", "B"]
    assert written == [
        tmp_path / "A.png",
        tmp_path / "B.png",
        tmp_path / "calibration.json",
        tmp_path / "labels.json",
    ]
    assert (tmp_path / "blender.log").read_text() == "A\nB\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "A.png",
        "B.png",
        "blender.log",
        "calibration.json",
        "labels.json",
    ]


@pytest.mark.parametrize("held_by", ["another part", "the folder"])
def test_a_file_of_one_name_moves_nothing(tmp_path: Path, held_by: str) -> None:
    parts = [tmp_path / ".eo", tmp_path / ".ir"]
    _part(parts[0], "A")
    _part(parts[1], "A" if held_by == "another part" else "B")
    if held_by == "the folder":
        (tmp_path / "A.png").write_bytes(b"an earlier frame")

    with pytest.raises(FileExistsError, match=r"A\.png"):
        journal.gather(tmp_path, parts, {})

    assert (parts[0] / "A.png").exists()
    assert not (tmp_path / "labels.json").exists()
