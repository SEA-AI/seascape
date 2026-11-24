"""A render's frames as its camera writes them, and the files beside them."""

from pathlib import Path

from seascape.config import ImageFormat


def stem(camera: str, frame: int | None) -> str:
    """`camera`'s image path in a render's folder, without the extension."""
    return camera if frame is None else f"{camera}/{frame:04d}"


def written(folder: Path, camera: str, fmt: ImageFormat) -> list[Path]:
    """`camera`'s images in `folder` in order: its still, or its sequence's frames."""
    still = folder / f"{stem(camera, None)}.{fmt}"
    if still.exists():
        return [still]
    # By number: `stem` pads to four digits, so frame 10000 sorts before 1001 as text.
    return sorted((folder / camera).glob(f"*.{fmt}"), key=lambda p: int(p.stem))
