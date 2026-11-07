"""The air and sea each named atmosphere has at the surface.

LOWTRAN7's model atmospheres (AFGL-TR-86-0110) and those measured at Helgoland;
the header of `data/lowtran_surface.csv` says how each was taken.
"""

from importlib import resources
from typing import Any, Literal

import numpy as np

DATA = resources.files("seascape.physics") / "data"
_SURFACE_CSV = "lowtran_surface.csv"

type Atmosphere = Literal[
    "tropical",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
    "us_standard",
    "north_sea",
    "north_sea_winter",
    "north_sea_summer",
]
# Judgement: European waters, as measured at Helgoland.
ATMOSPHERE: Atmosphere = "north_sea"


def load_table(csv: str, **kwargs: Any) -> np.ndarray:
    """The rows of the shipped CSV `csv`, its `#` lines left out; `kwargs` go to
    `np.loadtxt`.
    """
    with (DATA / csv).open() as f:
        return np.loadtxt(f, delimiter=",", comments="#", **kwargs)


def header(csv: str) -> list[str]:
    """A shipped LOWTRAN table's column names after the first: its last comment line."""
    with (DATA / csv).open() as f:
        header = [line for line in f if line.startswith("#")][-1]
    return header.removeprefix("#").strip().split(",")[1:]


def _surface(column: str) -> dict[str, float]:
    names = ["profile", *header(_SURFACE_CSV)]
    with (DATA / _SURFACE_CSV).open() as f:
        rows = np.genfromtxt(f, delimiter=",", dtype=None, names=names)
    return dict(zip(rows["profile"].tolist(), rows[column].tolist(), strict=True))


SURFACE_AIR_K = _surface("air_k")
SURFACE_SEA_K = _surface("sea_k")

T_SEA_K = SURFACE_SEA_K[ATMOSPHERE]
