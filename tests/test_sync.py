"""The sync preview's settings and its check of the labels."""

from pathlib import Path

import pytest

from seascape import sync
from seascape.config import Crop, load
from seascape.outputs.labels import Annotation, Image, Labels
from tests.scenarios import SYNC_SENTRY, SYNC_WATCHKEEPER


@pytest.mark.parametrize("path", [SYNC_SENTRY, SYNC_WATCHKEEPER])
def test_a_preview_shrinks_every_camera_and_crop_over_the_same_span(path: Path) -> None:
    full = load(path)
    quick = load(path, [sync.preview(full)])

    assert quick.outputs.fps == sync.PREVIEW_FPS
    assert quick.outputs.duration_s == full.outputs.duration_s
    for rig, small in zip(full.rigs.values(), quick.rigs.values(), strict=True):
        for name, camera in rig.cameras.items():
            shrunk_px = camera.width_px // sync.PREVIEW_SHRINK
            assert small.cameras[name].width_px == shrunk_px
            assert small.cameras[name].hfov_deg == camera.hfov_deg
        for name, video in rig.videos.items():
            if isinstance(video, Crop):
                sides = ("left_px", "right_px", "top_px", "bottom_px")
                cut = {s: getattr(video, s) // sync.PREVIEW_SHRINK for s in sides}
                assert small.videos[name] == video.model_copy(update=cut)
            else:
                assert small.videos[name] == video


def seen(*frames: list[bool]) -> Labels:
    """One video, a frame per entry, a container per flag: True is cut."""
    images = [
        Image.model_construct(id=i, camera="mast_T1", time_s=i / 3)
        for i in range(len(frames))
    ]
    annotations = [
        Annotation.model_construct(image_id=i, truncated=cut)
        for i, frame in enumerate(frames)
        for cut in frame
    ]
    return Labels.model_construct(images=images, annotations=annotations)


def test_whole_containers_in_every_frame_pass() -> None:
    sync.check(seen([False, False], [False, False]))


@pytest.mark.parametrize(
    ("frames", "error"),
    [
        (([False], [True]), "mast_T1 at 0.33 s: a container touches"),
        (([False, False], [False]), "mast_T1 at 0.33 s sees 1 containers, 2 at first"),
        (([], []), "mast_T1 at 0.00 s sees no container"),
    ],
)
def test_a_frame_that_loses_or_cuts_a_container_fails(
    frames: tuple[list[bool], ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        sync.check(seen(*frames))
