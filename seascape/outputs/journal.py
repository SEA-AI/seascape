"""A render's truth as it goes: one line of JSON per frame, appended.

Rewriting calibration.json and labels.json after every frame costs time and disk in
the square of the frame count: at 30 fps a minute of four cameras ends near 200 MB,
written some 3600 times. Each frame is appended here once instead, and the two files
are written once from the lines. Renders of a scene's bands apart gather into one
folder here too, as one render writes it.
"""

import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from pydantic import Field, ValidationError

from seascape.config import Scenario
from seascape.model import Model
from seascape.outputs.calibration import FILENAME as CALIBRATION
from seascape.outputs.calibration import Calibration, CameraCalibration
from seascape.outputs.labels import FILENAME as LABELS
from seascape.outputs.labels import Annotation, Category, Dropped, Image, Labels, concat

FILENAME = "truth.jsonl"

_BLENDER_LOG = "blender.log"


class Step(Model):
    """What one frame adds: its cameras, and the labels new since the last."""

    info: dict[str, Any] = Field(default_factory=dict)
    cameras: list[CameraCalibration] = Field(default_factory=list)
    raw: list[CameraCalibration] = Field(default_factory=list)
    images: list[Image] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    dropped: list[Dropped] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)


class Journal:
    """Appends each frame's `Step` to `FILENAME` in a folder, and as its block ends
    writes calibration.json and labels.json from them.

    It writes them also when the block raises, so a render that dies keeps every frame
    it finished; a render killed outright leaves the journal for `finish`. A frame's
    labels are what `truth` gained since the last, so their ids stay those it numbered.
    """

    def __init__(self, folder: Path, truth: Labels) -> None:
        self.path = folder / FILENAME
        self.written: list[Path] = []
        self._truth = truth
        self._seen = (0, 0, 0, 0)  # images, annotations, dropped, categories
        self._info_written = False
        # Not appended to: a folder's journal is its render's alone. Bytes, so the
        # locale's encoding never applies and only LF ends a line.
        self._file = self.path.open("wb")

    def append(
        self, cameras: list[CameraCalibration], raw: list[CameraCalibration]
    ) -> None:
        """Write one frame: `cameras` as written, `raw` as rendered, and whatever
        labels the truth gained since the last frame.
        """
        truth = self._truth
        images, annotations, dropped, categories = self._seen
        step = Step(
            info={} if self._info_written else truth.info,
            cameras=cameras,
            raw=raw,
            images=truth.images[images:],
            annotations=truth.annotations[annotations:],
            dropped=truth.dropped[dropped:],
            categories=truth.categories[categories:],
        )
        self._file.write(step.model_dump_json().encode() + b"\n")
        # Flushed per frame, so a killed render leaves at most its last line torn.
        self._file.flush()
        self._info_written = self._info_written or bool(truth.info)
        self._seen = (
            len(truth.images),
            len(truth.annotations),
            len(truth.dropped),
            len(truth.categories),
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        try:
            # Closing flushes, and can fail as the append before it did.
            self._file.close()
            self.written = finish(self.path.parent)
        except Exception as failed:
            if error is None:
                raise
            # The render's own error is the one to report; the journal stays.
            error.add_note(f"{self.path} was left unfinished: {failed!r}")


def _steps(folder: Path) -> list[Step]:
    """The journal's lines; a torn last one, from a render killed mid-write, is left
    out.
    """
    # Split as bytes: a value may hold U+2028, which str.splitlines also splits at,
    # and a torn last line may end inside a character.
    lines = (folder / FILENAME).read_bytes().splitlines()
    steps = []
    for number, line in enumerate(lines, 1):
        try:
            steps.append(Step.model_validate_json(line))
        except ValidationError:
            if number < len(lines):
                raise
    return steps


def _assemble(steps: list[Step]) -> tuple[Calibration, Labels]:
    labels = Labels(info=next((s.info for s in steps if s.info), {}))
    cameras, raw = [], []
    for step in steps:
        cameras += step.cameras
        raw += step.raw
        labels.images += step.images
        labels.annotations += step.annotations
        labels.dropped += step.dropped
        labels.categories += step.categories
    return Calibration(cameras=cameras, raw=raw), labels


def read(folder: Path) -> tuple[Calibration, Labels]:
    """The calibration and labels the journal in `folder` holds."""
    return _assemble(_steps(folder))


def finish(folder: Path) -> list[Path]:
    """Write calibration.json and labels.json from the journal in `folder`, then
    delete it; the files written, none if it holds no frame.
    """
    written = []
    if steps := _steps(folder):
        calibration, labels = _assemble(steps)
        written = [calibration.write(folder), labels.write(folder)]
    (folder / FILENAME).unlink()
    return written


def gather(folder: Path, parts: Sequence[Path], scenario: Scenario) -> list[Path]:
    """Move `parts`, finished renders of a scene's bands, into `folder` as one render
    of `scenario` writes it, then delete them; the truth files written. Each part's
    Blender log appends to the folder's.

    Raises:
        FileExistsError: Two parts, or a part and `folder`, hold a file of one name;
            nothing has moved.
    """
    joined = concat({part.name: Labels.read(part) for part in parts})
    # Each part's scenario renders some bands; the truth records the whole scene's.
    joined.info["scenario"] = scenario.model_dump(mode="json")
    cameras = Calibration.concat([Calibration.read(part) for part in parts])
    moves = []
    for part in parts:
        moves += [
            item
            for item in part.iterdir()
            if item.name not in {LABELS, CALIBRATION, _BLENDER_LOG}
        ]
    counts = Counter(item.name for item in moves)
    if twice := sorted(name for name, count in counts.items() if count > 1):
        raise FileExistsError(f"{folder}: more than one part holds {twice}")
    if there := sorted(name for name in counts if (folder / name).exists()):
        raise FileExistsError(f"{folder} already holds {there}")
    for item in moves:
        item.rename(folder / item.name)
    with (folder / _BLENDER_LOG).open("a") as log:
        for part in parts:
            if (part / _BLENDER_LOG).exists():
                log.write((part / _BLENDER_LOG).read_text(errors="replace"))
    written = [cameras.write(folder), joined.write(folder)]
    for part in parts:
        shutil.rmtree(part)
    return written
