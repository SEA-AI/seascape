"""Draws labels.json's masks on a frame."""

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from seascape.outputs import brand, labels


def masks(
    image: labels.Image,
    annotations: list[labels.Annotation],
    box: tuple[float, float, float, float],
    size: tuple[int, int],
) -> np.ndarray:
    """Each mask, cropped to `box` and resized to `size`; a pixel it touches stays."""
    resolution = (image.width, image.height)
    return np.stack(
        [
            np.asarray(
                Image.fromarray(
                    sv.rle_to_mask(a.segmentation.counts, resolution).astype(np.uint8)
                    * 255
                ).resize(size, Image.Resampling.BOX, box=box)
            )
            > 0
            for a in annotations
        ]
    )


def outlined(pic: np.ndarray, found: sv.Detections) -> np.ndarray:
    """`pic` with each target haloed, boxed and outlined."""
    if not len(found):
        return pic
    red = sv.Color.from_hex(brand.FOCUS_RED)
    # `found` carries no class ids, which the default lookup needs.
    by_index = sv.ColorLookup.INDEX
    grown = sv.Detections(
        xyxy=found.xyxy,
        mask=np.stack(
            [cv2.dilate(m.view(np.uint8), np.ones((3, 3))) > 0 for m in found.mask]
        ),
    )
    pic = sv.HaloAnnotator(
        color=red, opacity=0.6, kernel_size=25, color_lookup=by_index
    ).annotate(pic, found)
    pic = sv.PolygonAnnotator(color=red, thickness=1, color_lookup=by_index).annotate(
        pic, grown
    )
    return sv.BoxAnnotator(color=red, thickness=1, color_lookup=by_index).annotate(
        pic, found
    )


def lit(img: Image.Image, found: np.ndarray) -> Image.Image:
    """`img` with the targets in `found` drawn on."""
    detections = sv.Detections(xyxy=sv.mask_to_xyxy(found), mask=found)
    pic = outlined(np.asarray(img)[..., ::-1].copy(), detections)
    return Image.fromarray(pic[..., ::-1])
