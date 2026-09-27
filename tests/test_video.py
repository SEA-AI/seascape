"""Encoding a render's frames as one video per camera."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image as Picture

from seascape import agc, video
from seascape.labels import Image, Labels

# cv2 decodes as BT.601 whatever the bt709 tag says: saturated colours shift, greys
# don't.
GREYS = [(0, 0, 0), (64, 64, 64), (200, 200, 200), (255, 255, 255), (230, 220, 200)]


def run(
    into: Path,
    colours: list[tuple[int, int, int]],
    times: list[float],
    size: tuple[int, int] = (64, 36),
) -> Path:
    """Stand-ins for a render: flat frames from one camera, and their labels."""
    (into / "port").mkdir()
    images = []
    for i, (colour, time_s) in enumerate(zip(colours, times, strict=True)):
        name = f"port/{i:04d}.png"
        Picture.new("RGB", size, colour).save(into / name)
        images.append(
            Image(
                id=i + 1,
                file_name=name,
                width=size[0],
                height=size[1],
                camera="port",
                band="eo",
                time_s=time_s,
                horizon_px=[],
            )
        )
    Labels(images=images).write(into)
    return into


def decoded(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while (frame := capture.read()[1]) is not None:
        frames.append(frame[..., ::-1])
    return frames, capture.get(cv2.CAP_PROP_FPS)


def test_a_camera_becomes_one_video_at_its_name(tmp_path) -> None:
    folder = run(tmp_path, GREYS[:4], [0.0, 0.1, 0.2, 0.3])

    assert video.encode(folder) == [folder / "port.mp4"]
    frames, fps = decoded(folder / "port.mp4")
    assert len(frames) == 4
    assert fps == pytest.approx(10.0)


def test_the_colours_pass_through_in_time_order(tmp_path) -> None:
    times = [0.4, 0.0, 0.2, 0.6, 0.8]  # labels.json order is not time order
    folder = run(tmp_path, GREYS, times)

    frames, _ = decoded(video.encode(folder)[0])

    expected = [colour for _, colour in sorted(zip(times, GREYS, strict=True))]
    for frame, colour in zip(frames, expected, strict=True):
        assert np.abs(frame.astype(int) - colour).max() <= 4


def test_a_16_bit_thermal_frame_is_toned_not_flattened(tmp_path) -> None:
    """The sequencer would show centikelvin as one flat grey."""
    folder = run(tmp_path, GREYS[:3], [0.0, 0.1, 0.2])
    t_k = np.repeat([280.0, 300.0], [18, 18])[:, None] * np.ones(64)
    for path in (folder / "port").glob("*.png"):
        cv2.imwrite(str(path), agc.counts(t_k))

    frames, _ = decoded(video.encode(folder)[0])

    for frame in frames:
        assert frame[0, 0, 0] <= 4 and frame[-1, 0, 0] >= 251


def test_one_frame_is_not_a_video(tmp_path) -> None:
    with pytest.raises(ValueError, match="one frame"):
        video.encode(run(tmp_path, GREYS[:1], [0.0]))


def test_an_uneven_time_step_names_itself(tmp_path) -> None:
    with pytest.raises(ValueError, match="constant time step"):
        video.encode(run(tmp_path, GREYS[:3], [0.0, 0.1, 0.3]))


def test_a_missing_frame_names_itself(tmp_path) -> None:
    folder = run(tmp_path, GREYS[:3], [0.0, 0.1, 0.2])
    (folder / "port/0001.png").unlink()

    with pytest.raises(FileNotFoundError, match=r"0001\.png"):
        video.encode(folder)


def test_an_odd_frame_size_cannot_be_h264(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="divisible by 2"):
        video.encode(run(tmp_path, GREYS[:2], [0.0, 0.1], size=(63, 36)))
