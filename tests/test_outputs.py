"""A render's frames as its camera writes them."""

from pathlib import Path

from seascape import outputs


def test_a_sequence_s_frames_come_in_order_past_frame_9999(tmp_path: Path) -> None:
    frames = [9999, 10000, 1001]
    for frame in frames:
        path = tmp_path / f"{outputs.stem('eo', frame)}.png"
        path.parent.mkdir(exist_ok=True)
        path.touch()

    written = outputs.written(tmp_path, "eo", "png")

    assert [int(p.stem) for p in written] == sorted(frames)
