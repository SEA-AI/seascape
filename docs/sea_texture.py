"""The LWIR sea's and targets' texture, measured alike on renders and on footage.

uv run python -m docs.sea_texture [--references] [IMAGE | RENDER_DIR ...]

Per image, after stretching its 1st to 99th percentile to 0 to 1: the horizon row,
the sea's step over the sky just across it, and the sea's median, and the std and
skew of its high-passed texture, in far, mid and near thirds of the rows below.
Per target, in that stretched frame quantised to 8 bits: the histogram and GLCM
entropy and the Sobel gradient variance inside its box, from a render folder's
labels.json or, for footage, from `FOOTAGE_BOXES`. --references adds SEA.AI's
thermal frames from the private SEA-AI/.github, fetched once into the asset cache.
"""

import argparse
import json
import math
import subprocess
from collections.abc import Sequence
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

# Picked by eye: (x, y, w, h), tight to the hull and superstructure, masts and sails
# left out. sailing_tr_640 shows only the clipped edge of a sail.
FOOTAGE_BOXES = {
    "sentry_twfov.png": ((329, 250, 38, 9),),
    "sentry_tnfov.png": ((351, 236, 155, 37),),
    "example_1_IR.jpg": ((560, 470, 168, 53),),
    "example_2_IR.jpg": ((545, 260, 15, 6),),
    "example_3_IR.jpg": ((57, 218, 54, 39), (360, 274, 20, 5)),
    "sailing_tl_640.png": ((412, 278, 72, 24),),
}
# The target metrics follow Liiv et al. 2026, "Training with synthetic data for drone
# detection in thermal imagery", arXiv:2608.17799, appendix C.1: 8-bit grey, and no
# histogram or GLCM statistic on a crop under 400 px.
MIN_CROP_PX = 400
# GLCM offsets as (dy, dx): 1 px at 0, 45, 90 and 135 degrees, their four
# orientations, as scikit-image's graycomatrix takes them.
GLCM_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
# Their noise floor on Sobel magnitudes, which they leave unstated: the response to a
# step of one grey level, which a 3x3 Sobel weighs 1 + 2 + 1 across.
SOBEL_FLOOR = 4.0

Box = tuple[float, float, float, float]


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


def grey8(frame: np.ndarray) -> np.ndarray:
    """A normalised frame as the 8-bit grey the target metrics read."""
    return np.clip(np.rint(frame * 255), 0, 255).astype(np.uint8)


def histogram_entropy(crop: np.ndarray) -> float:
    """Shannon entropy of an 8-bit crop's 256-bin histogram, in bits."""
    counts = np.bincount(crop.ravel(), minlength=256)
    p = counts[counts > 0] / crop.size
    return float(-(p * np.log2(p)).sum())


def glcm_entropy(crop: np.ndarray) -> float:
    """Entropy of an 8-bit crop's grey-level co-occurrence matrix, averaged over
    `GLCM_OFFSETS`, in nats as scikit-image's graycoprops reports it."""
    h, w = crop.shape
    entropies = []
    for dy, dx in GLCM_OFFSETS:
        a = crop[max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)]
        b = crop[max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)]
        pairs = a.astype(np.int64).ravel() * 256 + b.ravel()
        counts = np.bincount(pairs, minlength=256 * 256)
        p = counts[counts > 0] / pairs.size
        entropies.append(-(p * np.log(p)).sum())
    return float(np.mean(entropies))


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def gradient_variance(magnitude: np.ndarray) -> float:
    """Variance of the Sobel magnitudes at or over `SOBEL_FLOOR`; 0 with none."""
    kept = magnitude[magnitude >= SOBEL_FLOOR]
    return float(kept.var()) if kept.size else 0.0


def target_metrics(image: np.ndarray, box: Box) -> dict[str, float]:
    """An 8-bit frame's statistics inside a COCO box. An entropy is nan on a crop
    under `MIN_CROP_PX`. Gradients are taken over the frame, so the box's edge sees
    its true neighbours."""
    x, y, w, h = box
    rows = slice(max(0, math.floor(y)), math.ceil(y + h))
    cols = slice(max(0, math.floor(x)), math.ceil(x + w))
    crop = image[rows, cols]
    big = crop.size >= MIN_CROP_PX
    return {
        "px": crop.size,
        "hist H": histogram_entropy(crop) if big else math.nan,
        "GLCM H": glcm_entropy(crop) if big else math.nan,
        "grad var": gradient_variance(sobel_magnitude(image)[rows, cols]),
    }


def render_targets(run: Path) -> list[tuple[str, Path, list[Box]]]:
    """A render folder's LWIR frames, each with its labelled boxes."""
    labels = json.loads((run / "labels.json").read_text())
    return [
        (
            str(run / image["file_name"]),
            run / image["file_name"],
            [a["bbox"] for a in labels["annotations"] if a["image_id"] == image["id"]],
        )
        for image in labels["images"]
        if image["band"] == "ir"
    ]


def print_sea(images: list[tuple[str, Path, int | None]]) -> None:
    columns = ["horizon", "step"] + [
        f"{t} {s}" for s in ("med", "std", "skew") for t in THIRDS
    ]
    width = max(len(label) for label, _, _ in images)
    print(f"{'image':<{width}}" + "".join(f"{c:>10}" for c in columns))
    for label, path, horizon in images:
        row = measure(normalised(path), horizon)
        cells = [f"{row['horizon']:>10d}"] + [f"{row[c]:>10.3f}" for c in columns[1:]]
        print(f"{label:<{width}}" + "".join(cells))


def print_targets(frames: Sequence[tuple[str, Path, Sequence[Box]]]) -> None:
    columns = ("px", "hist H", "GLCM H", "grad var")
    width = max(len(label) for label, _, _ in frames)
    print(f"{'target':<{width}}{'box':>20}" + "".join(f"{c:>10}" for c in columns))
    for label, path, boxes in frames:
        image = grey8(normalised(path))
        for box in boxes:
            row = target_metrics(image, box)
            cells = [f"{row['px']:>10d}"] + [
                f"{'-':>10}" if math.isnan(row[c]) else f"{row[c]:>10.2f}"
                for c in columns[1:]
            ]
            where = " ".join(f"{v:g}" for v in box)
            print(f"{label:<{width}}{where:>20}" + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    if not args.paths and not args.references:
        parser.error("give an image, a render folder or --references")
    # An image given by path keeps it, so a before and an after of one name differ.
    images = [(str(path), path, None) for path in args.paths if not path.is_dir()]
    targets = [t for path in args.paths if path.is_dir() for t in render_targets(path)]
    if args.references:
        images += [(n, reference(n), HORIZONS.get(n)) for n in REFERENCES]
        footage = [(n, reference(n), b) for n, b in FOOTAGE_BOXES.items()]
        targets = footage + targets
    if images:
        print_sea(images)
    if images and targets:
        print()
    if targets:
        print_targets(targets)


if __name__ == "__main__":
    main()
