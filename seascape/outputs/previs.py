"""What a previs of a scenario labels and when it samples the run, and its HUD."""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from seascape import angles
from seascape.outputs import brand

# Judgement: the top view's room round everything it frames, as a share of its span.
MARGIN = 0.12
# Judgement: an object other than the ownship grows until its footprint spans
# `OBJECT_PX`, so a buoy shows its kind's shape from a few hundred metres, but never
# past `GROW` times its size, past which it reads as a toy.
GROW = 5.0
OBJECT_PX = 12
# Judgement, in pixels of a 1000 px tall view, scaled with the view: a hairline's
# width, and a label's cap height.
LINE_PX, LABEL_PX = 1.0, 13.0
# Judgement: a compass tick every `TICK_STEP_DEG` off the bow, its label every 90 deg
# this far past the outer ring, in pixels as above.
TICK_STEP_DEG, TICK_LABEL_PX = 30, 28.0

type Box = tuple[float, float, float, float]
type Px = tuple[float, float]


def ring_ranges_m(ranges_m: Iterable[float], merge: float = 0.05) -> list[float]:
    """The ranges to ring, each run of `ranges_m` within `merge` of its smallest
    merged into its mean.

    Args:
        ranges_m: Each object's range, in any order.
        merge: The share above a run's smallest range that still joins the run.
            Judgement: within 5 % their rings lie a few pixels apart and read as one,
            and `distance`'s two significant figures err by as much.

    Returns:
        The rings' ranges, ascending.
    """
    groups: list[list[float]] = []
    for r in sorted(ranges_m):
        if groups and r <= groups[-1][0] * (1.0 + merge):
            groups[-1].append(r)
        else:
            groups.append([r])
    return [float(np.mean(g)) for g in groups]


def distance(m: float) -> str:
    """A distance as a label.

    Args:
        m: Metres.

    Returns:
        `m` to two significant figures, in m, or in km from 1 km.
    """
    m = float(f"{m:.2g}")
    return f"{m / 1000:g} km" if m >= 1000.0 else f"{m:g} m"


@dataclass(frozen=True, slots=True)
class TopView:
    """An orthographic camera straight down, bow at the start up."""

    centre_m: np.ndarray
    heading_rad: float
    m_per_px: float

    def matrix(self, height_m: float) -> np.ndarray:
        """The camera's pose, looking down its -Z as a Blender camera looks.

        Args:
            height_m: The camera's height above the water.

        Returns:
            `T_world_camera`, (4, 4): its +X to starboard, its +Y the bow.
        """
        T = np.eye(4)
        T[:3, :3] = angles.turned(math.degrees(self.heading_rad), 0.0)
        T[:3, 3] = (*self.centre_m, height_m)
        return T


def top_view(
    points_m: np.ndarray,
    heading_rad: float,
    size_px: tuple[int, int],
    radius_m: np.ndarray | float = 0.0,
) -> TopView:
    """The top view that frames circles with `MARGIN` round them.

    Args:
        points_m: (n, 2) east and north of their centres.
        heading_rad: The bearing that runs up the view.
        size_px: The view's width and height.
        radius_m: Their radii, (n,) or one for all.

    Returns:
        The view, centred on the points.
    """
    R = angles.turned(math.degrees(heading_rad), 0.0)
    starboard, bow = R[:2, 0], R[:2, 1]
    across, along = points_m @ starboard, points_m @ bow
    low = np.array([(across - radius_m).min(), (along - radius_m).min()])
    high = np.array([(across + radius_m).max(), (along + radius_m).max()])
    span = (high - low) * (1.0 + 2.0 * MARGIN)
    mid = (low[0] + high[0]) / 2 * starboard + (low[1] + high[1]) / 2 * bow
    # Judgement: a floor, so a lone point still frames a few tens of metres.
    m_per_px = max(float(span[0]) / size_px[0], float(span[1]) / size_px[1], 0.05)
    return TopView(mid, heading_rad, m_per_px)


def samples(output_fps: float, count: int, fps: float) -> tuple[np.ndarray, float]:
    """Output frames evenly spaced, as one render call takes them.

    Args:
        output_fps: The scenario's frame rate.
        count: How many output frames the run has.
        fps: The rate asked for; the step is the nearest whole one to it.

    Returns:
        Every step-th frame's index, shape (n,), and the rate they play at in real
        time.
    """
    step = max(1, round(output_fps / fps))
    return np.arange(0, count, step), output_fps / step


def grown(m_per_px: float, footprint_m: float) -> float:
    """How much to scale an object so it reads at a distance.

    Args:
        m_per_px: What a pixel spans where the object is.
        footprint_m: The object's longest side.

    Returns:
        The factor that makes it `OBJECT_PX` across, between 1 and `GROW`.
    """
    return min(GROW, max(1.0, OBJECT_PX * m_per_px / footprint_m))


def nice_m(most_m: float) -> float:
    """A scale bar's length.

    Args:
        most_m: The longest it may be; positive.

    Returns:
        The largest 1, 2 or 5 times a power of ten up to `most_m`.
    """
    power = 10.0 ** math.floor(math.log10(most_m))
    return max(k * power for k in (1.0, 2.0, 5.0) if k * power <= most_m)


def label_bearing_deg(candidates_deg: np.ndarray, hulls_deg: Sequence[float]) -> float:
    """Where a ring's label sits clear of the hulls.

    Args:
        candidates_deg: Bearings it may sit at, shape (n,).
        hulls_deg: Each hull's bearing.

    Returns:
        The candidate farthest from every hull; the middle one without hulls.
    """
    if not hulls_deg:
        return float(candidates_deg[len(candidates_deg) // 2])
    gaps = np.abs(angles.wrapped_deg(candidates_deg[:, None] - np.array(hulls_deg)))
    return float(candidates_deg[np.argmax(gaps.min(axis=1))])


def overlap(a: Box, b: Box) -> bool:
    """Whether two boxes share any area.

    Args:
        a: (left, top, right, bottom).
        b: Another, in the same pixels.

    Returns:
        True if they overlap; touching edges do not.
    """
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def hud_lines(
    t_s: float, heading_rad: float, speed_mps: float, most: float | None
) -> tuple[str, str, str]:
    """The HUD's text for a frame.

    Args:
        t_s: The time into the run.
        heading_rad: The ownship's heading.
        speed_mps: The ownship's speed.
        most: The most an object in view is grown, if any is.

    Returns:
        The time, the heading and speed, and how much objects are grown.
    """
    # The international knot, exactly: 1852 m an hour.
    mps_per_kn = 1852.0 / 3600.0
    heading_deg = round(math.degrees(heading_rad)) % 360
    grew = "" if most is None else f"objects up to \N{MULTIPLICATION SIGN}{most:.0f}"
    return (
        f"{t_s:.1f} s",
        f"HDG {heading_deg:03d}°   {speed_mps / mps_per_kn:.1f} kn",
        grew,
    )


@cache
def _cap(weight: brand.Weight) -> float:
    """Barlow's cap height per pixel of font size."""
    return -brand.font(100, weight).getbbox("H", anchor="ls")[1] / 100.0


def hud(
    path: Path,
    lines: tuple[str, str, str],
    m_per_px: float | None,
    rings: Sequence[tuple[Px, float]] = (),
    compass: tuple[Px, float] | None = None,
) -> None:
    """Draw the HUD on a picture: bottom left, the time large; each ring's label; and
    from above, the compass.

    Args:
        path: The picture, overwritten.
        lines: `hud_lines`'s text.
        m_per_px: What a pixel spans; with it, a scale bar.
        rings: Each ring's label's centre, in pixels, and its range. A label that
            would overlap the compass's or an earlier one's is left out.
        compass: The ownship's start in pixels and the outer ring's radius in
            pixels, in a view with its bow up.
    """
    # Judgement, in pixels of a 1000 px tall view: the time's and the heading's cap
    # heights, the margin from the view's edge, the lines' gap and a tick's length.
    time_px, nav_px, inset_px, gap_px, tick_px = 34.0, 15.0, 32.0, 14.0, 12.0
    ink, muted, label, ring = "#FFFFFF", "#8C96A2", "#D5DBE2", "#3A4350"
    with Image.open(path) as f:
        image = f.convert("RGB")
    w, h = image.size
    k = h / 1000.0
    pen = ImageDraw.Draw(image)
    thin = max(1, round(LINE_PX * k))

    def text(
        x: float,
        y_mid: float,
        body: str,
        cap_px: float,
        colour: str,
        weight: brand.Weight = "Regular",
    ) -> None:
        size = round(cap_px * k / _cap(weight))
        pen.text(
            (x, y_mid + cap_px * k / 2),
            body,
            colour,
            brand.font(size, weight),
            anchor="ls",
        )

    marks: list[tuple[Px, str]] = []
    if compass is not None:
        (cx, cy), r = compass
        for off in range(0, 360, TICK_STEP_DEG):
            e, n, _ = angles.direction(off, 0.0)
            ends = [(cx + e * at, cy - n * at) for at in (r, r + tick_px * k)]
            pen.line(ends, ring, thin)
            if off % 90 == 0:
                at = r + TICK_LABEL_PX * k
                marks.append(((cx + e * at, cy - n * at), f"{off}°"))
    marks += [(xy, distance(m)) for xy, m in rings]
    face = brand.font(round(LABEL_PX * k / _cap("Regular")))
    drawn: list[Box] = []
    for (x, y), body in marks:
        at = (x, y + LABEL_PX * k / 2)
        box = pen.textbbox(at, body, face, anchor="ms")
        if not any(overlap(box, d) for d in drawn):
            pen.text(at, body, label, face, anchor="ms")
            drawn.append(box)

    time, nav, most = lines
    x, y = inset_px * k, h - inset_px * k
    if m_per_px is not None:
        # Judgement: a scale bar about a fifth of the view wide.
        bar_m = nice_m(0.2 * w * m_per_px)
        x1 = x + bar_m / m_per_px
        pen.line([(x, y), (x1, y)], muted, thin)
        for end in (x, x1):
            pen.line([(end, y), (end, y - 6 * k)], muted, thin)
        text(x1 + 8 * k, y - LABEL_PX / 2 * k, distance(bar_m), LABEL_PX, muted)
        y -= (gap_px + LABEL_PX) * k
    text(x, y - LABEL_PX / 2 * k, most, LABEL_PX, muted)
    y -= (LABEL_PX + gap_px / 2) * k
    text(x, y - nav_px / 2 * k, nav, nav_px, muted)
    y -= (nav_px + gap_px) * k
    text(x, y - time_px / 2 * k, time, time_px, ink, "Medium")
    image.save(path)
