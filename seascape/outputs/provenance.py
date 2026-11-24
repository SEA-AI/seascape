"""How a render was made: the commit that reproduces it, and EXIF naming it for a
picture that leaves the server.
"""

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

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


def commit() -> str | None:
    """The commit this module's work tree is at, else the image's, else None."""
    here = Path(__file__)
    try:
        # A wheel in another project's .venv sits in a work tree that does not track it.
        _git(here.parent, "ls-files", "--error-unmatch", here.name)
        # --exclude: a tag would otherwise replace the hash with its own name.
        return _git(
            here.parent, "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*"
        )
    except (OSError, subprocess.CalledProcessError):
        # The Dockerfile's build arg: an image carries no .git.
        return os.environ.get("SEASCAPE_COMMIT") or None


def clean(commit: str | None) -> bool:
    """Whether `commit`, as `commit()` names one, can be checked out to render the
    same again.
    """
    return commit is not None and not commit.endswith("-dirty")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()
