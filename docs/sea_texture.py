"""The LWIR sea's texture, measured alike on renders and on footage.

uv run python -m docs.sea_texture [--references] [IMAGE ...]

Per image, after stretching its 1st to 99th percentile to 0 to 1: the horizon row,
the sea's step over the sky just across it, and the sea's median, and the std and
skew of its high-passed texture, in far, mid and near thirds of the rows below.
--references adds SEA.AI's thermal frames from the private SEA-AI/.github, fetched
once into the asset cache.
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

from seascape.assets import CACHE

REFERENCES = (
    "sentry_twfov.png",
    "sentry_tnfov.png",
    "sailing_tl_640.png",
    "sailing_tr_640.png",
    "example_1_IR.jpg",
    "example_2_IR.jpg",
    "example_3_IR.jpg",
)
# The last sky row of a reference whose largest jump is not its horizon: haze spreads
# example_3's over many rows, and a streak of glitter jumps further.
HORIZONS = {"example_3_IR.jpg": 240}
# The high-pass box, and the rows either side of the horizon its step is taken over.
BOX_PX, STEP_ROWS = 9, 5
# A jpg's frame line: a row or two that jumps further than any horizon.
BORDER_ROWS = 2
THIRDS = ("far", "mid", "near")


def reference(name: str) -> Path:
    path = CACHE / "references" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = subprocess.run(
            [
                *("gh", "api", f"repos/SEA-AI/.github/contents/assets/{name}"),
                *("-H", "Accept: application/vnd.github.raw"),
            ],
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
        # Renamed into place, so an interrupted write never takes the cache name.
        partial = path.with_suffix(".part")
        partial.write_bytes(raw)
        partial.replace(path)
    return path


def normalised(path: Path) -> np.ndarray:
    """The frame as float grey, its 1st percentile 0 and its 99th 1, unclipped."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"cannot read {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
    frame = image.astype(np.float32)
    low, high = np.percentile(frame, (1, 99))
    return (frame - low) / max(high - low, 1e-6)


def skew(x: np.ndarray) -> float:
    d = x - x.mean()
    return float((d**3).mean() / max((d**2).mean() ** 1.5, 1e-12))


def last_sky_row(frame: np.ndarray) -> int:
    """The row above the largest jump in row medians, away from the frame's edge."""
    jumps = np.abs(np.diff(np.median(frame, axis=1)))
    inner = jumps[BORDER_ROWS:-BORDER_ROWS]
    return int(np.argmax(inner)) + BORDER_ROWS


def measure(frame: np.ndarray, horizon: int | None = None) -> dict[str, float]:
    horizon = last_sky_row(frame) if horizon is None else horizon
    if not STEP_ROWS - 1 <= horizon < frame.shape[0] - BOX_PX // 2 - 3:
        raise ValueError(f"a horizon at row {horizon} leaves no sky or no sea thirds")
    sky = frame[horizon + 1 - STEP_ROWS : horizon + 1]
    sea = frame[horizon + 1 : horizon + 1 + STEP_ROWS]
    texture = frame - cv2.blur(frame, (BOX_PX, BOX_PX))
    rows = np.array_split(np.arange(horizon + 1 + BOX_PX // 2, frame.shape[0]), 3)
    out = {"horizon": horizon, "step": float(np.median(sea) - np.median(sky))}
    for third, r in zip(THIRDS, rows, strict=True):
        out[f"{third} med"] = float(np.median(frame[r]))
        out[f"{third} std"] = float(texture[r].std())
        out[f"{third} skew"] = skew(texture[r])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    if not args.images and not args.references:
        parser.error("give an image or --references")
    # An image given by path keeps it, so a before and an after of one name differ.
    images = [(str(path), path, None) for path in args.images]
    if args.references:
        images += [(n, reference(n), HORIZONS.get(n)) for n in REFERENCES]
    columns = ["horizon", "step"] + [
        f"{t} {s}" for s in ("med", "std", "skew") for t in THIRDS
    ]
    width = max(len(label) for label, _, _ in images)
    print(f"{'image':<{width}}" + "".join(f"{c:>10}" for c in columns))
    for label, path, horizon in images:
        row = measure(normalised(path), horizon)
        cells = [f"{row['horizon']:>10d}"] + [f"{row[c]:>10.3f}" for c in columns[1:]]
        print(f"{label:<{width}}" + "".join(cells))


if __name__ == "__main__":
    main()
