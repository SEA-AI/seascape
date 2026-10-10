"""The primer's figures: each a row of renders of `primer.toml`, one `--set` apart.

uv run python -m docs.primer.figures            # every figure
uv run python -m docs.primer.figures wind haze  # some
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from docs.hero import SKIES, captioned, render
from seascape import waves
from seascape.calibration import Calibration
from seascape.config import load

HERE = Path(__file__).parent
PRIMER = HERE / "primer.toml"
GAP_PX = 6
SHIP = (
    '{ preset = "multipurpose_freighter", range_m = %s, '
    "bearing_deg = 0.0, heading_deg = 250.0 }"
)
SUN_AHEAD = ["sky.sun_bearing_deg = 0.0", "sky.sun_elevation_deg = 12.0"]
LWIR = ['outputs.bands = ["ir"]', "sky.t_air_k = 288.0"]
FULL_HD = "rigs.bow.cameras.eo = { hfov_deg = 45.0, width_px = 1920, height_px = 1080 }"
CROP_PX, CROP_SCALE = (192, 56), 4
HULL_DOWN_KM = (15, 30, 40)
TELE = "rigs.bow.cameras.eo = { hfov_deg = 1.5, width_px = 640, height_px = 360 }"

FIGURES: dict[str, tuple[list[str], list[tuple[str, list[str]]]]] = {
    "samples": (
        SUN_AHEAD,
        [
            (f"{n} sample{'s' * (n > 1)}", [f"outputs.samples.eo = {n}"])
            for n in (1, 4, 64)
        ],
    ),
    "wind": (
        SUN_AHEAD,
        [(f"wind {u} m/s", [f"sea.wind_speed_mps = {u}.0"]) for u in (2, 7, 14)],
    ),
    "eo-ir": (
        [*SKIES["cumulus"], f"objects = [{SHIP % 2000.0}]"],
        [
            (name, [f'outputs.bands = ["{band}"]'])
            for name, band in (("EO", "eo"), ("LWIR", "ir"))
        ],
    ),
    "lwir": (
        LWIR,
        [
            (f"sea {label} air", [f"sea.t_sea_k = {t}"])
            for label, t in (
                ("5 K under", 283.0),
                ("equal to", 288.0),
                ("5 K over", 293.0),
            )
        ],
    ),
    "haze": (
        [f"objects = [{SHIP % 2000.0}]"],
        [(f"visibility {v} km", [f"sky.visibility_km = {v}.0"]) for v in (42, 10, 3)],
    ),
    "horizon": (
        # Clear air and a mast-top camera, so the sea hides most of the far hull.
        [TELE, "rigs.bow.height_m = 30.0", "sky.visibility_km = 1000.0"],
        [
            (f"ship at {r} km", [f"objects = [{SHIP % (r * 1000.0)}]"])
            for r in HULL_DOWN_KM
        ],
    ),
}


def frame(overrides: list[str], out: Path, caption: str) -> Image.Image:
    render(PRIMER, overrides, out)
    (image,) = json.loads((out / "labels.json").read_text())["images"]
    return captioned(Image.open(out / image["file_name"]).convert("RGB"), caption, 18)


def crops() -> None:
    """The sea at 100 m and at 1 km in one full-size frame, pixels blown up."""
    radius_m = waves.earth_radius_m(load(PRIMER).sea.refraction_k)
    with tempfile.TemporaryDirectory() as tmp:
        render(PRIMER, [FULL_HD], Path(tmp))
        (camera,) = Calibration.read(Path(tmp)).cameras
        full = Image.open(Path(tmp) / camera.image).convert("RGB")
    pose = np.array(camera.extrinsics["world"])
    ahead = pose[:2, 2] / np.hypot(*pose[:2, 2])
    (w, h), size = CROP_PX, (CROP_PX[0] * CROP_SCALE, CROP_PX[1] * CROP_SCALE)
    sheet = Image.new("RGB", (size[0], 2 * size[1] + GAP_PX), "white")
    for i, (d_m, caption) in enumerate(((100.0, "100 m away"), (1000.0, "1 km away"))):
        east_m, north_m = pose[:2, 3] + d_m * ahead
        sea = np.array([east_m, north_m, waves.sea_z_m(east_m, north_m, radius_m)])
        u, v, z = np.array(camera.K) @ pose[:3, :3].T @ (sea - pose[:3, 3])
        left, top = round(u / z - w / 2), round(v / z - h / 2)
        panel = full.crop((left, top, left + w, top + h))
        panel = captioned(panel.resize(size, Image.Resampling.NEAREST), caption, 18)
        sheet.paste(panel, (0, i * (size[1] + GAP_PX)))
    sheet.save(HERE / "crops.jpg", quality=90)


def figure(name: str) -> None:
    shared, panels = FIGURES[name]
    with tempfile.TemporaryDirectory() as tmp:
        frames = [
            frame(shared + sets, Path(tmp) / str(i), caption)
            for i, (caption, sets) in enumerate(panels)
        ]
    width = sum(f.width for f in frames) + GAP_PX * (len(frames) - 1)
    sheet = Image.new("RGB", (width, frames[0].height), "white")
    x = 0
    for f in frames:
        sheet.paste(f, (x, 0))
        x += f.width + GAP_PX
    sheet.save(HERE / f"{name}.jpg", quality=85)


if __name__ == "__main__":
    for name in sys.argv[1:] or [*FIGURES, "crops"]:
        crops() if name == "crops" else figure(name)
        print(name)
