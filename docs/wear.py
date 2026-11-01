"""docs/scenarios.md's figure: one freighter as delivered and worn, in EO and LWIR.

uv run python -m docs.wear docs/wear.jpg
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image

from docs.hero import GAP_PX, captioned, render

SCENARIO = Path(__file__).with_name("wear.toml")
HULL = (
    'asset = "multipurpose_freighter", range_m = 260.0, bearing_deg = 0.0, '
    "heading_deg = 90.0"
)
WEARS = {
    "as delivered": [f"objects = [{{ {HULL} }}]"],
    "wear 0.8": [f"objects = [{{ {HULL}, wear = 0.8 }}]"],
}
WIDTH_PX = 640


def main(out: Path) -> None:
    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for caption, overrides in WEARS.items():
            run = Path(tmp) / caption
            render(SCENARIO, overrides, run)
            for band in ("eo", "ir"):
                img = Image.open(run / f"bow_{band}.jpg").convert("RGB")
                img = img.resize((WIDTH_PX, WIDTH_PX * img.height // img.width))
                tiles.append(captioned(img, f"{band.upper()}, {caption}"))
    eo_clean, ir_clean, eo_worn, ir_worn = tiles
    rows = [(eo_clean, eo_worn), (ir_clean, ir_worn)]
    height = sum(row[0].height for row in rows) + GAP_PX
    sheet = Image.new("RGB", (2 * WIDTH_PX + GAP_PX, height), "white")
    y = 0
    for left, right in rows:
        sheet.paste(left, (0, y))
        sheet.paste(right, (WIDTH_PX + GAP_PX, y))
        y += left.height + GAP_PX
    sheet.save(out, quality=88)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
