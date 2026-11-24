"""A render's frames as its camera writes them, and the files beside them."""

from pathlib import Path

from seascape.config import ImageFormat


def stem(camera: str, frame: int | None) -> str:
    """`camera`'s image path in a render's folder, without the extension."""
    return camera if frame is None else f"{camera}/{frame:04d}"


def written(folder: Path, camera: str, fmt: ImageFormat) -> list[Path]:
    """`camera`'s images in `folder` in order: its still, or its sequence's frames."""
    still = folder / f"{stem(camera, None)}.{fmt}"
    return [still] if still.exists() else sorted((folder / camera).glob(f"*.{fmt}"))
