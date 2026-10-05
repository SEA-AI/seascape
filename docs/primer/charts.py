"""The primer's charts, each in a light and a dark PNG, from seascape's own physics.

uv run --with matplotlib python docs/primer/charts.py
"""

import math
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from seascape import lwir, waves

OUT = Path(__file__).parent / "charts"
# A temperate sea.
SEA_K = 288.0
# IAU 2015 Resolution B3: the sun's nominal effective temperature.
SUN_K = 5772.0
# The shortest gravity wave, where capillarity takes over: lambda = 2 pi sqrt(sigma /
# rho g), 1.7 cm for seawater (Lamb, Hydrodynamics, section 267).
CAPILLARY_M = 0.017

# SEA.AI's brand colours, light and dark: Focus Red first, then the neutrals.
THEMES = {
    "light": {
        "series": ("#CB0D00", "#06404C", "#7B9194"),
        "surface": "#FFFFFF",
        "text": "#000000",
        "secondary": "#000000",
        "muted": "#7B9194",
        "grid": "#DFDED9",
        "axis": "#7B9194",
        "band": "#DFDED9",
    },
    "dark": {
        "series": ("#CB0D00", "#DFDED9", "#7B9194"),
        "surface": "#0B1731",
        "text": "#FFFFFF",
        "secondary": "#DFDED9",
        "muted": "#7B9194",
        "grid": "#06404C",
        "axis": "#7B9194",
        "band": "#06404C",
    },
}

type Theme = dict
type Chart = Callable[[Axes, Theme], None]


def label_end(
    ax: Axes, t: Theme, x: float, y: float, text: str, offset: tuple = (6, 0)
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        va="center",
        ha="right" if offset[0] < 0 else "center" if offset[0] == 0 else "left",
        color=t["secondary"],
        fontsize=10,
    )


def reference(ax: Axes, t: Theme, y: float, text: str, x: float = 0.01) -> None:
    ax.axhline(y, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        text,
        (x, y),
        ha="left" if x < 0.5 else "right",
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        color=t["muted"],
        fontsize=9.5,
    )


def glow(ax: Axes, t: Theme) -> None:
    lam_um = np.geomspace(0.2, 60, 600)
    for (low, high), name in (((0.4, 0.7), "EO"), ((8, 14), "LWIR")):
        ax.axvspan(low, high, color=t["band"], zorder=0)
        ax.text(
            math.sqrt(low * high), 1.12, name, ha="center", color=t["muted"], size=10
        )
    bodies = ((f"the sun, {SUN_K:.0f} K", SUN_K), (f"the sea, {SEA_K:.0f} K", SEA_K))
    for colour, (name, t_k) in zip(t["series"], bodies, strict=False):
        b = lwir.planck(lam_um * 1e-6, t_k)
        b /= b.max()
        ax.plot(lam_um, b, color=colour, lw=2, label=name)
        label_end(ax, t, lam_um[b.argmax()], 1.0, name, (8, 0))
    ax.set_xscale("log")
    ax.set_xlim(0.2, 60)
    ax.set_ylim(0, 1.2)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
    ax.set_xlabel("wavelength, µm")
    ax.set_yticks([])


def footprint(ax: Axes, t: Theme) -> None:
    height_m, width_px, hfov = 12.0, 1920, math.radians(45)
    radius_m = waves.earth_radius_m(0.13)
    d_m = np.geomspace(20, 8000, 300)
    across = d_m * hfov / width_px
    along = across / np.sin(np.arctan(height_m / d_m) - d_m / (2 * radius_m))
    peak_m = 2 * math.pi * waves.GRAVITY_MS2 / waves.peak_omega_rad_s(7.0) ** 2
    reference(ax, t, peak_m, f"a 7 m/s sea's peak wave, {peak_m:.0f} m")
    reference(ax, t, CAPILLARY_M, "the shortest gravity wave, 1.7 cm", x=0.99)
    for colour, y, name in zip(
        t["series"], (along, across), ("along the view", "across"), strict=False
    ):
        ax.plot(d_m, y, color=colour, lw=2, label=name)
        label_end(ax, t, d_m[-1], y[-1], name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, 8000)
    ax.set_xticks([20, 100, 1000, 8000], ["20 m", "100 m", "1 km", "8 km"])
    ticks = [0.01, 0.1, 1, 10, 100, 1000]
    ax.set_yticks(ticks, ["1 cm", "10 cm", "1 m", "10 m", "100 m", "1 km"])
    ax.set_xlabel("distance from the camera")
    ax.set_ylabel("one pixel's footprint")


def hidden(ax: Axes, t: Theme) -> None:
    radius_m = waves.earth_radius_m(0.13)
    r_km = np.linspace(0, 50, 500)
    for colour, h in zip(t["series"], (3.0, 12.0, 30.0), strict=True):
        horizon_km = waves.horizon_m(h, 0.13) / 1000
        beyond = r_km >= horizon_km
        x = r_km[beyond]
        y = (x - horizon_km) ** 2 * 1e6 / (2 * radius_m)
        ax.plot(x, y, color=colour, lw=2, label=f"camera {h:.0f} m up")
        ax.plot(horizon_km, 0, "o", ms=8, color=colour, mec=t["surface"], mew=2)
        at_km = horizon_km + math.sqrt(2 * radius_m * 25.0) / 1000
        label_end(ax, t, at_km, 25.0, f"{h:.0f} m up", (8, -6))
    seen = (45 - waves.horizon_m(30.0, 0.13) / 1000) ** 2 * 1e6 / (2 * radius_m)
    ax.plot(45, seen, "o", ms=8, color=t["series"][2], mec=t["surface"], mew=2)
    ax.annotate(
        f"the 45 km ship above:\n{seen:.0f} m hidden",
        (45, seen),
        xytext=(49.5, 6),
        textcoords="data",
        ha="right",
        color=t["secondary"],
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": t["muted"], "lw": 1},
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 80)
    ax.set_xlabel("range to the target, km")
    ax.set_ylabel("height hidden by the sea, m")


CHARTS: dict[str, tuple[str, str, Chart]] = {
    "glow": (
        "The sun glows in the visible, the sea in the thermal",
        "Planck's law, each curve scaled to its peak. Shaded: what each camera sees.",
        glow,
    ),
    "footprint": (
        "By 1 km, one pixel is as long as the biggest waves",
        "1920 px across 45°, 12 m up. A wave shorter than the footprint is "
        "drawn as roughness.",
        footprint,
    ),
    "hidden": (
        "Raise the camera and the horizon moves out",
        "How much of a target the curve hides. Each line starts at its horizon.",
        hidden,
    ),
}


def draw(name: str, theme: str) -> None:
    t = THEMES[theme]
    title, subtitle, chart = CHARTS[name]
    with mpl.rc_context(
        {
            # From Google Fonts; matplotlib finds it once installed.
            "font.family": ["Barlow Semi Condensed", "DejaVu Sans"],
            "font.size": 10.5,
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "axes.edgecolor": t["axis"],
            "axes.labelcolor": t["secondary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": t["axis"],
            "ytick.color": t["axis"],
            "xtick.labelcolor": t["secondary"],
            "ytick.labelcolor": t["secondary"],
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
        chart(ax, t)
        # The gridlines carry the y scale.
        ax.minorticks_off()
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.suptitle(
            title, x=0.01, ha="left", color=t["text"], fontsize=14, fontweight="bold"
        )
        ax.set_title(subtitle, loc="left", color=t["muted"], fontsize=10, pad=10)
        # Twice the width it displays at, for high-density screens.
        fig.savefig(OUT / f"{name}-{theme}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in CHARTS:
        for theme in THEMES:
            draw(name, theme)
