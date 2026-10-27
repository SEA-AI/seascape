"""The LWIR sea's and targets' texture, measured alike on renders and on footage.

uv run python -m docs.sea_texture [--references] [IMAGE | RENDER_DIR ...]

Per image, after stretching its 1st to 99th percentile to 0 to 1: the horizon row,
the sea's step over the sky just across it, and its noise: the std of the sky's
high-passed texture in its median row. In far, mid and near thirds of the rows
below: the sea's median, and its high-passed texture's std, skew, lx (the lag across
at which it correlates under 1/2) and gx/gy (its RMS step across over down), each
with the sky's noise taken out, so a noisy camera and a clean render compare by the
sea's structure alone.
Per target, in that stretched frame quantised to 8 bits: the histogram and GLCM
entropy and the Sobel gradient variance inside its box, from a render folder's
labels.json or, for footage, from `FOOTAGE_BOXES`. --references adds SEA.AI's
thermal frames from the private SEA-AI/.github, fetched once into the asset cache.
"""

import argparse
import math
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from seascape import labels
from seascape.assets import CACHE

# The frames whose sea is measured: raw counts, so the sky's noise is the sea's. The
# jpgs went through an AGC that gives the sea a different gain from the sky's, so
# they serve only as targets.
REFERENCES = (
    "sentry_twfov.png",
    "sentry_tnfov.png",
    "sailing_tl_640.png",
    "sailing_tr_640.png",
)
# The high-pass box, and the rows either side of the horizon its step is taken over.
BOX_PX, STEP_ROWS = 9, 5
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


def last_sky_row(frame: np.ndarray) -> int:
    """The row above the largest jump in row medians."""
    return int(np.argmax(np.abs(np.diff(np.median(frame, axis=1)))))


def products(texture: np.ndarray, lag: int, axis: int) -> np.ndarray:
    """Per row, the mean product of zero-mean `texture` with itself `lag` px along
    `axis`."""
    n = texture.shape[axis]
    a = texture.take(range(n - lag), axis=axis)
    b = texture.take(range(lag, n), axis=axis)
    return (a * b).mean(axis=1)


def structure(sea: np.ndarray, sky: np.ndarray) -> dict[str, float]:
    """The std, skew, lx and gx/gy of `sea`'s high-passed texture less `sky`'s, nan
    where nothing is left. Noise independent of the scene adds its covariance at every
    lag and its third cumulant, so the sky's are subtracted; the sky is taken as
    smooth under the high pass, and its noise as the sea's."""
    sea, sky = sea - sea.mean(), sky - sky.mean()

    def c(lag: int, axis: int) -> float:
        # The sky's median row, as a cloud or a mast lifts a few.
        sky_c = np.median(products(sky, lag, axis))
        return float(products(sea, lag, axis).mean() - sky_c)

    var = c(0, 0)
    if var <= 0:
        return dict.fromkeys(("std", "skew", "lx", "gx/gy"), math.nan)
    # A step's mean square is 2 (C(0) - C(1)).
    across, down = var - c(1, 1), var - c(1, 0)
    return {
        "std": math.sqrt(var),
        "skew": ((sea**3).mean() - np.median((sky**3).mean(axis=1))) / var**1.5,
        "lx": next(
            (lag for lag in range(1, sea.shape[1]) if c(lag, 1) < var / 2), math.nan
        ),
        "gx/gy": math.sqrt(across / down) if across > 0 < down else math.nan,
    }


def measure(frame: np.ndarray) -> dict[str, float]:
    """The sky's, then each third's, rows as `__doc__` lists them. A row whose high
    pass reaches the horizon or the frame's edge is left out of either."""
    horizon = last_sky_row(frame)
    reach = BOX_PX // 2
    bottom = frame.shape[0] - reach
    if not 2 * reach < horizon < bottom - reach - 3:
        raise ValueError(f"a horizon at row {horizon} leaves no sky or no sea thirds")
    sky = frame[horizon + 1 - STEP_ROWS : horizon + 1]
    sea = frame[horizon + 1 : horizon + 1 + STEP_ROWS]
    texture = frame - cv2.blur(frame, (BOX_PX, BOX_PX))
    # Clouds in these rows overstate the noise, so a third under a cloudy sky can
    # read nan.
    noise = texture[reach : horizon + 1 - reach]
    rows = np.array_split(np.arange(horizon + 1 + reach, bottom), 3)
    out = {
        "horizon": horizon,
        "step": float(np.median(sea) - np.median(sky)),
        "noise": math.sqrt(np.median((noise - noise.mean()).var(axis=1))),
    }
    for third, r in zip(THIRDS, rows, strict=True):
        out[f"{third} med"] = float(np.median(frame[r]))
        for metric, value in structure(texture[r], noise).items():
            out[f"{third} {metric}"] = value
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


def target_metrics(
    image: np.ndarray, box: Box, magnitude: np.ndarray | None = None
) -> dict[str, float]:
    """An 8-bit frame's statistics inside a COCO box. An entropy is nan on a crop
    under `MIN_CROP_PX`. Gradients are taken over the frame, so the box's edge sees
    its true neighbours; `magnitude` is the frame's `sobel_magnitude`, if taken."""
    if magnitude is None:
        magnitude = sobel_magnitude(image)
    x, y, w, h = box
    rows = slice(max(0, math.floor(y)), math.ceil(y + h))
    cols = slice(max(0, math.floor(x)), math.ceil(x + w))
    crop = image[rows, cols]
    big = crop.size >= MIN_CROP_PX
    return {
        "px": crop.size,
        "hist H": histogram_entropy(crop) if big else math.nan,
        "GLCM H": glcm_entropy(crop) if big else math.nan,
        "grad var": gradient_variance(magnitude[rows, cols]),
    }


def render_targets(run: Path) -> list[tuple[str, Path, list[Box]]]:
    """A render folder's LWIR frames, each with its labelled boxes."""
    found = labels.Labels.model_validate_json((run / labels.FILENAME).read_text())
    return [
        (
            str(run / image.file_name),
            run / image.file_name,
            [a.bbox for a in found.annotations if a.image_id == image.id],
        )
        for image in found.images
        if image.band == "ir"
    ]


def print_sea(images: list[tuple[str, Path]]) -> None:
    columns = ("horizon", "step", "noise", "med", "std", "skew", "lx", "gx/gy")
    width = max(len(label) for label, _ in images)
    print(f"{'image':<{width}}{'third':>6}" + "".join(f"{c:>8}" for c in columns))
    for label, path in images:
        row = measure(normalised(path))
        for third in THIRDS:
            cells = [
                f"{row[c] if c in row else row[f'{third} {c}']:>8.3f}"
                for c in columns[1:]
            ]
            print(f"{label:<{width}}{third:>6}{row['horizon']:>8d}" + "".join(cells))


def print_targets(frames: Sequence[tuple[str, Path, Sequence[Box]]]) -> None:
    columns = ("px", "hist H", "GLCM H", "grad var")
    width = max(len(label) for label, _, _ in frames)
    print(f"{'target':<{width}}{'box':>20}" + "".join(f"{c:>10}" for c in columns))
    for label, path, boxes in frames:
        image = grey8(normalised(path))
        magnitude = sobel_magnitude(image)
        for box in boxes:
            row = target_metrics(image, box, magnitude)
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
    images = [(str(path), path) for path in args.paths if not path.is_dir()]
    targets = [t for path in args.paths if path.is_dir() for t in render_targets(path)]
    if args.references:
        images += [(n, reference(n)) for n in REFERENCES]
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
