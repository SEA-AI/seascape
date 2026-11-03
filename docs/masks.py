"""docs/outputs.md's figure: labels.json's masks read through supervision.

uv run --with supervision python -m docs.masks docs/masks.jpg
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

from docs.hero import GAP_PX, SCENARIO, captioned, render
from docs.outline import outlined
from seascape.outputs import labels

PANEL_PX = (640, 480)
MARGIN = 1.3
CAPTIONS = {"eo": "EO", "ir": "LWIR"}


def spotlight(image: np.ndarray, found: sv.Detections) -> np.ndarray:
    return outlined(
        sv.BackgroundOverlayAnnotator(opacity=0.8).annotate(image, found), found
    )


def around(image: np.ndarray, found: sv.Detections) -> tuple[np.ndarray, sv.Detections]:
    """`image` and `found` cut to the panel's aspect around every mask."""
    if not len(found):
        return image, found
    x0, y0 = found.xyxy[:, :2].min(axis=0)
    x1, y1 = found.xyxy[:, 2:].max(axis=0)
    w = max((x1 - x0) * MARGIN, 2 * PANEL_PX[0] / 16)
    h = w * PANEL_PX[1] / PANEL_PX[0]
    shrink = min(1, image.shape[1] / w, image.shape[0] / h)
    w, h = w * shrink, h * shrink
    left = int(np.clip((x0 + x1 - w) / 2, 0, image.shape[1] - w))
    top = int(np.clip((y0 + y1 - h) / 2, 0, image.shape[0] - h))
    window = np.s_[top : top + int(h), left : left + int(w)]
    mask = found.mask[(slice(None), *window)]
    return image[window], sv.Detections(xyxy=sv.mask_to_xyxy(mask), mask=mask)


def main(figure: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        render(SCENARIO, ["sky.sun_elevation_deg = 35"], run)
        dataset = sv.DetectionDataset.from_coco(
            str(run), str(run / "labels.json"), force_masks=True
        )
        bands = {
            i.file_name: i.band
            for i in labels.Labels.model_validate_json(
                (run / labels.FILENAME).read_text()
            ).images
        }
        panels = []
        for path, image, found in dataset:
            panel = Image.fromarray(spotlight(*around(image, found))[..., ::-1])
            panel = panel.resize(PANEL_PX, Image.Resampling.LANCZOS)
            panels.append(captioned(panel, CAPTIONS[bands[Path(path).name]]))
    sheet = Image.new(
        "RGB", (sum(p.width for p in panels) + GAP_PX * (len(panels) - 1), PANEL_PX[1])
    )
    left = 0
    for panel in panels:
        sheet.paste(panel, (left, 0))
        left += panel.width + GAP_PX
    sheet.save(figure, quality=85)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
