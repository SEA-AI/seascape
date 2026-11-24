"""A recording's W&B run: its render's progress and previews as they come, then the
artifacts it publishes.
"""

from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    import wandb


class Run:
    """One command's live W&B run, named `name`, and the wandb module that opened it."""

    def __init__(self, live: "wandb.Run", wandb: ModuleType, name: str) -> None:
        self._run, self._wandb, self.name = live, wandb, name

    def progress(self, *, done: int, s_per_image: float, eta_s: float) -> None:
        """Log `done` images, the wall time per image and the time left."""
        self._run.log({"images_done": done, "s_per_image": s_per_image, "eta_s": eta_s})

    def log(self, key: str, shown: Path | Image.Image) -> None:
        """Log `shown` under `key`: an mp4 at its path as a video, W&B copying it as it
        logs, else a picture or the one at its path.
        """
        if isinstance(shown, Path) and shown.suffix == ".mp4":
            self._run.log({key: self._wandb.Video(str(shown))})
        else:
            picture = str(shown) if isinstance(shown, Path) else shown
            self._run.log({key: self._wandb.Image(picture)})

    def publish(
        self,
        info: dict[str, Any],
        scenario: Path,
        recordings: Iterable[Path],
    ) -> None:
        """Log artifact `name`, of type `recording`, holding `recordings`, and use
        artifact `<name>-scenario`, holding `scenario` as `<name>.toml`, as its input.

        Args:
            info: The render's labels.json info: the run's config and the
                recording's metadata.
            scenario: The TOML that rendered it, resolved.
            recordings: Folders, each kept whole as `<parent>/<name>`.
        """
        name = self.name
        self._run.config.update(info, allow_val_change=True)
        source = self._wandb.Artifact(f"{name}-scenario", type="scenario")
        source.add_file(str(scenario), name=f"{name}.toml")
        self._run.use_artifact(source)
        recording = self._wandb.Artifact(
            name,
            type="recording",
            description=f"seascape {info['version']} at {info['commit']}",
            metadata=info,
        )
        for folder in recordings:
            recording.add_dir(str(folder), name=f"{folder.parent.name}/{folder.name}")
        self._run.log_artifact(recording).wait()


@contextmanager
def run(project: str, entity: str | None, name: str) -> Generator[Run]:
    """A run of `project` named `name`, in `entity` or wandb's default, while the
    block runs; W&B records the command line and commit itself.

    Args:
        project: The W&B project.
        entity: The W&B team or user; None for wandb's default.
        name: The run's name, and its artifacts'.

    Yields:
        The live run.

    Raises:
        ValueError: wandb is not installed, or starts no run: logged out, offline or
            a wrong entity.
    """
    try:
        import wandb  # noqa: PLC0415 - an optional extra
    except ImportError as error:
        raise ValueError(
            "publishing needs wandb: uv sync --extra wandb, then wandb login"
        ) from error
    try:
        live = wandb.init(project=project, entity=entity, name=name)
    except wandb.errors.Error as error:
        raise ValueError(f"W&B started no run: {error}") from error
    with live:
        # Offline, a run starts but cannot use an artifact: refuse before the render.
        if live.offline:
            raise ValueError("W&B is offline: publishing needs WANDB_MODE online")
        yield Run(live, wandb, name)
