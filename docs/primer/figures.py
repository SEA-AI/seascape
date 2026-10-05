"""The primer's figures: each a row of renders of `primer.toml`, one `--set` apart.

uv run python docs/primer/figures.py            # every figure
uv run python docs/primer/figures.py wind haze  # some
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
GAP_PX = 6
SHIP = (
    '{ preset = "container_ship", range_m = %s, '
    "bearing_deg = 0.0, heading_deg = 250.0 }"
)
SUN_AHEAD = ["sky.sun_bearing_deg = 0.0", "sky.sun_elevation_deg = 12.0"]
LWIR = ['outputs.bands = ["ir"]', "sky.t_air_k = 288.0"]
TELE = (
    'rig.pods = [{ name = "bow", yaw_deg = 0.0, cameras = ['
    '{ kind = "eo", hfov_deg = 1.5, width_px = 640, height_px = 360 }] }]'
)

# name: (shared overrides, [(caption, overrides), ...])
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
    "sun": (
        [],
        [(f"sun at {e}°", [f"sky.sun_elevation_deg = {e}.0"]) for e in (60, 10, 2)],
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
        [TELE, "rig.height_m = 30.0", "sky.visibility_km = 1000.0"],
        [
            (f"ship at {r} km", [f"objects = [{SHIP % (r * 1000.0)}]"])
            for r in (15, 30, 45)
        ],
    ),
}


def render(overrides: list[str], out: Path) -> Path:
    sets = [arg for o in overrides for arg in ("--set", o)]
    subprocess.run(
        ["seascape", "render", str(HERE / "primer.toml"), "-o", str(out), *sets],
        check=True,
        capture_output=True,
    )
    return next(p for p in sorted(out.glob("*_0.*")) if p.suffix in (".jpg", ".png"))


def captioned(path: Path, caption: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=18)
    left, top, right, bottom = draw.textbbox((10, 8), caption, font=font)
    draw.rectangle([left - 4, top - 3, right + 4, bottom + 3], fill="black")
    draw.text((10, 8), caption, fill="white", font=font)
    return img


def figure(name: str) -> None:
    shared, panels = FIGURES[name]
    with tempfile.TemporaryDirectory() as tmp:
        frames = [
            captioned(render(shared + sets, Path(tmp) / str(i)), caption)
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
    for name in sys.argv[1:] or FIGURES:
        figure(name)
        print(name)
