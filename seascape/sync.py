"""The sync check's preview: a short, coarse render of a sync scenario, and the check
that every recorded video keeps its containers whole in every frame.
"""

from collections import Counter
from typing import Any

from seascape.config import Crop, Scenario
from seascape.outputs.labels import Labels

# Judgement: a thirtieth of the frames; a peak between two of them goes unseen.
PREVIEW_FPS = 1
# Judgement: a 1280 px colour camera comes out a GIF's 320 px wide, where a container
# a few metres long is still tens of pixels.
PREVIEW_SHRINK = 4
_CROP_SIDES = ("left_px", "right_px", "top_px", "bottom_px")


def preview(scenario: Scenario) -> dict[str, Any]:
    """Overrides that render `scenario` at `PREVIEW_FPS` and one sample, each camera
    and crop `PREVIEW_SHRINK` times smaller, over the same span of motion.
    """
    rigs = {
        name: {
            "cameras": {
                c: {
                    "width_px": camera.width_px // PREVIEW_SHRINK,
                    "height_px": camera.height_px // PREVIEW_SHRINK,
                }
                for c, camera in rig.cameras.items()
            },
            "videos": {
                v: {
                    side: getattr(video, side) // PREVIEW_SHRINK for side in _CROP_SIDES
                }
                for v, video in rig.videos.items()
                if isinstance(video, Crop)
            },
        }
        for name, rig in scenario.rigs.items()
    }
    outputs = {"fps": PREVIEW_FPS, "samples": {"eo": 1, "ir": 1}}
    return {"rigs": rigs, "outputs": outputs}


def check(seen: Labels) -> None:
    """Fail if a video sees no container, one touching its frame's edge, or a count
    that differs from its first frame's.

    Raises:
        ValueError: The first frame that fails, by video and time.
    """
    shown = Counter(a.image_id for a in seen.annotations)
    cut = Counter(a.image_id for a in seen.annotations if a.truncated)
    first: dict[str, int] = {}
    for image in sorted(seen.images, key=lambda i: i.time_s):
        count = first.setdefault(image.camera, shown[image.id])
        at = f"{image.camera} at {image.time_s:.2f} s"
        if not count:
            raise ValueError(f"{at} sees no container")
        if cut[image.id]:
            raise ValueError(f"{at}: a container touches the frame's edge")
        if shown[image.id] != count:
            raise ValueError(
                f"{at} sees {shown[image.id]} containers, {count} at first"
            )
