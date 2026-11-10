"""EXIF naming how a render was made, for a picture that leaves the server."""

import json
from collections.abc import Sequence

from PIL import ExifTags
from PIL import Image as Picture

# A JPEG segment's length is 16 bits (ITU-T T.81), and EXIF is one segment.
# Judgement: the margin left for the tags besides the description.
EXIF_MAX_BYTES = 60_000


def exif(
    scenario: str, seed: int, overrides: Sequence[str], camera: str | None = None
) -> Picture.Exif:
    """EXIF naming the `scenario`, `seed`, `overrides` and `camera` of a picture.

    Overrides too long for the segment are replaced by a note that they are left out.
    """
    made: dict[str, object] = {
        "scenario": scenario,
        "seed": seed,
        "overrides": list(overrides),
    }
    if camera:
        made["camera"] = camera
    text = json.dumps(made)
    if len(text) > EXIF_MAX_BYTES:
        omitted = {"scenario": scenario, "seed": seed, "overrides_omitted": True}
        text = json.dumps(omitted | ({"camera": camera} if camera else {}))
    tags = Picture.Exif()
    tags[ExifTags.Base.Software] = "seascape"
    tags[ExifTags.Base.ImageDescription] = text
    return tags
