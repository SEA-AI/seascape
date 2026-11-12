"""A product's sync course on one sheet, to check before rendering it.

The sheet stacks a title over two rows: the course from above beside it unrolled as
the rig sees it, then a timeline of which video holds a buoy of each ring, over the
attitudes the recording holds. A wrong mount or turn is quicker to spot here than in
a failed preview.
"""

import functools
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageOps
from PIL.ImageFont import FreeTypeFont

from seascape.angles import wrapped_deg
from seascape.assets import Buoy, Debris
from seascape.blender.scene import Posed, posed_at_start
from seascape.config import Crop, Object, Rig, Scenario, Video, sources
from seascape.motion import axis_deg, held_tilt_deg, panned_deg, rig_track
from seascape.outputs import brand, canvas
from seascape.sync import preview
from seascape.sync.course import (
    BUOY,
    CONTAINER,
    Ring,
    View,
    asset,
    buoy_above_m,
    elevation_deg,
    half_deg,
    one_rig,
    overlap_deg,
    reach_m,
    rings,
    views,
)
from seascape.sync.course import course as placed

# Drawn K times the size, then shrunk: Pillow anti-aliases no line or polygon.
K = 2
WIDTH_PX = 2400 * K
MARGIN_PX = 28 * K
INNER_PX = WIDTH_PX - 2 * MARGIN_PX
GAP_PX = 48 * K
TOP_PX = 640 * K
ABOVE_W_PX = 1000 * K
HEAD_PX = 70 * K
FOOT_PX = 70 * K
ROW_PX = 48 * K
SIGNAL_PX = 120 * K
LEFT_PX = 300 * K
RING_LABEL_PX = 100 * K
PAD_PX = 8 * K
TICK_PX = 12 * K
LINE_PX = 2 * K
DOT_PX = 7 * K
# Seen at about half its size; ONS's data visualisation typography sets a 14 px
# floor.
FONT_PX = 28 * K
MATTE = (255, 255, 255)
INK = (0, 0, 0)
GRID = ImageColor.getrgb(brand.FOG_WHITE)
MUTED = ImageColor.getrgb(brand.SKY_GREY)
LABEL = ImageColor.getrgb(brand.NIGHT_BLUE)
type Colour = tuple[int, int, int]


def _rgb(colour: str) -> Colour:
    r, g, b, *_ = ImageColor.getrgb(colour)
    return r, g, b


# The brand's four colours; its book allows no other, so the grey is the furniture's
# too. Past four videos, canvas's palette.
SERIES = [
    _rgb(c)
    for c in (brand.FOCUS_RED, brand.OCEAN_TEAL, brand.SKY_GREY, brand.NIGHT_BLUE)
]
SHADOW = (0, 0, 0, 40)
PAIR = (0, 0, 0, 60)
CONE_ALPHA = 50
FRAME_ALPHA = 34
# Judgement: the bird's-eye view's radius, past the farthest cone by this share.
REACH = 1.05
# Judgement: points along each edge of a frame.
EDGE_POINTS = 48
# Judgement: a panorama this wide shows every bearing.
ALL_ROUND_DEG = 300.0
# Judgement: the panorama's room past its frames, across and up, the timeline's
# tick, and a signal strip's least half span and its headroom.
ACROSS_PAD_DEG = 5.0
UP_PAD_DEG = 2.0
TICK_S = 10
LEAST_HALF_DEG = 0.5
HEADROOM = 1.15
# ISO 668: every series 1 container is 2438 mm wide, a standard one 2591 mm tall.
CONTAINER_WIDE_M = 2.438
CONTAINER_TALL_M = 2.591


@dataclass(frozen=True, slots=True, kw_only=True)
class _Course:
    """What every part of the sheet reads, worked out once."""

    scenario: Scenario
    name: str
    rig: Rig
    # The rig's axis at the start, clockwise from the bow.
    axis_deg: float
    videos: list[View]
    rings: list[Ring]
    objects: list[Object]
    sight: Callable[[Object], list[tuple[float, float]]]
    colours: dict[str, Colour]


def _course(scenario: Scenario) -> _Course:
    name, rig, _ = one_rig(scenario)
    videos = views(rig)
    names = [v.name for v in videos]
    many = ((int(r), int(g), int(b)) for r, g, b in canvas.palette(len(names)))
    palette = SERIES if len(names) <= len(SERIES) else list(many)
    colours: dict[str, Colour] = dict(zip(names, palette, strict=False))
    return _Course(
        scenario=scenario,
        name=name,
        rig=rig,
        axis_deg=axis_deg(scenario, name)(0.0),
        videos=videos,
        rings=sorted(rings(videos), key=lambda r: r.range_m),
        objects=placed(scenario).objects(scenario.seed),
        sight=preview.sightlines(scenario),
        colours=colours,
    )


@functools.cache
def _font(weight: brand.Weight = "Regular", px: int = FONT_PX) -> FreeTypeFont:
    """The typeface, read once for each weight and size."""
    return brand.font(px, weight)


def _metres(range_m: float) -> str:
    return f"{range_m:.0f} m"


def _tile(size: tuple[int, int], title: str, note: str) -> Image.Image:
    """A blank tile, its title at the top left and its key after it."""
    tile = Image.new("RGB", size, MATTE)
    pen = ImageDraw.Draw(tile)
    pen.text((0, 0), title, font=_font("Medium"), fill=LABEL)
    pen.text(
        (pen.textlength(title, font=_font("Medium")) + GAP_PX, 0),
        note,
        font=_font(),
        fill=MUTED,
    )
    return tile


def _tick(
    draw: ImageDraw.ImageDraw, x: float, top: float, bottom: float, text: str
) -> None:
    """A gridline from `top` to `bottom`, its label centred under it."""
    draw.line([(x, top), (x, bottom)], fill=GRID, width=K)
    draw.text((x, bottom + TICK_PX), text, font=_font(), fill=INK, anchor="ma")


def _dot(draw: ImageDraw.ImageDraw, x: float, y: float, buoy: Object) -> None:
    """`buoy` in its colour, outlined so a white one shows on the sheet."""
    sold = asset(BUOY, Buoy).colours
    fill = ImageColor.getrgb(sold.get(buoy.colour or "", "#FFFFFF"))
    box = (x - DOT_PX, y - DOT_PX, x + DOT_PX, y + DOT_PX)
    draw.ellipse(box, fill=fill, outline=INK, width=1)


def _header(course: _Course) -> Image.Image:
    """The product's name, and each video's colour by its name at the right."""
    tile = Image.new("RGB", (INNER_PX, 2 * FONT_PX + PAD_PX), MATTE)
    pen = ImageDraw.Draw(tile)
    pen.text(
        (0, 0),
        f"{course.rig.model or 'Rig'} sync course",
        font=_font("Medium", 2 * FONT_PX),
        fill=INK,
    )
    x = INNER_PX
    for name, colour in reversed(course.colours.items()):
        pen.text((x, FONT_PX), name, font=_font(), fill=colour, anchor="rm")
        x -= pen.textlength(name, font=_font()) + GAP_PX
    return tile


@dataclass(frozen=True, slots=True)
class _Disc:
    """The bird's-eye view's scale: range on a square-root scale off the rig."""

    centre: tuple[int, int]
    radius_px: float
    reach_m: float

    def _r(self, range_m: float) -> float:
        return self.radius_px * math.sqrt(min(range_m, self.reach_m) / self.reach_m)

    def px(self, range_m: float, bearing_deg: float) -> tuple[float, float]:
        b = math.radians(bearing_deg)
        r = self._r(range_m)
        return self.centre[0] + r * math.sin(b), self.centre[1] - r * math.cos(b)

    def box(self, range_m: float) -> tuple[float, float, float, float]:
        r, (x, y) = self._r(range_m), self.centre
        return x - r, y - r, x + r, y + r

    def wedge(
        self,
        tile: Image.Image,
        range_m: float,
        span_deg: tuple[float, float],
        fill: tuple[int, int, int, int],
        outline: Colour | None = None,
    ) -> None:
        """A wedge off the rig over bearings `span_deg`, blended over `tile`."""
        layer = Image.new("RGBA", tile.size, (0, 0, 0, 0))
        start, end = (b - 90.0 for b in span_deg)  # Pillow's 0 is east
        ImageDraw.Draw(layer).pieslice(
            self.box(range_m), start, end, fill=fill, outline=outline, width=LINE_PX
        )
        tile.paste(layer, mask=layer)


def _objects(tile: Image.Image, disc: _Disc, course: _Course) -> None:
    """Each buoy a dot; the container at its footprint, with the shadow it casts."""
    draw = ImageDraw.Draw(tile)
    shadow = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    hidden = ImageDraw.Draw(shadow)
    half_m = asset(CONTAINER, Debris).length_m / 2
    for o in course.objects:
        range_m, off_deg = course.sight(o)[0]
        bearing_deg = course.axis_deg + off_deg
        if o.asset == BUOY:
            _dot(draw, *disc.px(range_m, bearing_deg), o)
            continue
        half = half_deg(2 * half_m, range_m)
        start, end = bearing_deg - half - 90.0, bearing_deg + half - 90.0
        hidden.pieslice(disc.box(disc.reach_m), start, end, fill=SHADOW)
        hidden.pieslice(disc.box(range_m), start, end, fill=(0, 0, 0, 0))
        # Round the footprint: the near face left to right, the far one back.
        corners = []
        for face_m, sides in (
            (range_m - CONTAINER_WIDE_M / 2, (-1, 1)),
            (range_m + CONTAINER_WIDE_M / 2, (1, -1)),
        ):
            out_m = math.hypot(half_m, face_m)
            across_deg = math.degrees(math.atan(half_m / face_m))
            for side in sides:
                corners.append(disc.px(out_m, bearing_deg + side * across_deg))
        draw.polygon(corners, fill=INK)
    tile.paste(shadow, mask=shadow)


def _from_above(course: _Course) -> Image.Image:
    """The rig, the bow up: a half disc with the rig at the bottom while every video
    and every object at the start lies forward, else a whole disc around it. Each
    cone ends where a buoy drops below `BUOY_PX`; grey is what both videos of a pair
    see. Range is on a square-root scale, so a near ring and a far one both show.
    """
    axis, videos = course.axis_deg, course.videos
    edges = [axis + d for v in videos for d in (v.left_deg, v.right_deg)]
    # A course turned past the beam places objects behind the starting bow.
    edges += [axis + course.sight(o)[0][1] for o in course.objects]
    aft = any(abs(wrapped_deg(e)) > 90.0 for e in edges)
    tile = _tile((ABOVE_W_PX, TOP_PX), "1  Top view", "grey: seen by two videos")
    # The forward half disc's ring labels go under its base, in two rows.
    foot = 0 if aft else 2 * FONT_PX + 2 * PAD_PX
    room = TOP_PX - HEAD_PX - foot
    if aft:
        radius = min(ABOVE_W_PX, room) / 2
        centre = (ABOVE_W_PX // 2, HEAD_PX + room // 2)
    else:
        radius = min(ABOVE_W_PX / 2 - DOT_PX, room)
        centre = (ABOVE_W_PX // 2, TOP_PX - foot)
    disc = _Disc(centre, radius, max(reach_m(v) for v in videos) * REACH)

    def bearings(span: tuple[float, float]) -> tuple[float, float]:
        return axis + span[0], axis + span[1]

    # Farthest first, so every shorter cone shows on top.
    for view in sorted(videos, key=reach_m, reverse=True):
        colour = course.colours[view.name]
        span = bearings((view.left_deg, view.right_deg))
        disc.wedge(tile, reach_m(view), span, (*colour, CONE_ALPHA), colour)
    for a, b in itertools.combinations(videos, 2):
        if both := overlap_deg(a, b):
            disc.wedge(tile, min(reach_m(a), reach_m(b)), bearings(both), PAIR)
    draw = ImageDraw.Draw(tile)
    clear, row = math.inf, 0
    for ring in course.rings:
        draw.arc(
            disc.box(ring.range_m),
            *((0, 360) if aft else (-180, 0)),
            fill=MUTED,
            width=K,
        )
        label = _metres(ring.range_m)
        if aft:
            x, y = disc.px(ring.range_m, 0.0)
            draw.text((x + PAD_PX, y), label, font=_font(), fill=MUTED, anchor="lb")
            continue
        # Under the arc's port end, a row down where it would touch the last.
        x, y = disc.px(ring.range_m, -90.0)
        half = draw.textlength(label, font=_font()) / 2
        row = 1 - row if x + half + PAD_PX > clear else 0
        clear = x - half
        top = y + PAD_PX + row * (FONT_PX + PAD_PX // 2)
        draw.text((x, top), label, font=_font(), fill=MUTED, anchor="ma")
    _objects(tile, disc, course)
    x, y = centre
    bow = [(x, y - 12 * K), (x - PAD_PX, y + 5 * K), (x + PAD_PX, y + 5 * K)]
    draw.polygon(bow, fill=INK)
    return tile


def _videos(rig: Rig) -> dict[str, tuple[tuple[str, ...], Video | None]]:
    """Each video's cameras and the video itself; each camera on its own, as None,
    where the rig records them whole.
    """
    return {
        name: (sources(video) if video else (name,), video)
        for name, video in (rig.videos or dict.fromkeys(rig.rendered)).items()
    }


def _margins_px(video: Video | None) -> tuple[int, int, int, int]:
    """The pixels `video` cuts off its camera's left, right, top and bottom."""
    if isinstance(video, Crop):
        return video.left_px, video.right_px, video.top_px, video.bottom_px
    return 0, 0, 0, 0


def frame_edges(
    rig: Rig, cameras: dict[str, np.ndarray]
) -> dict[str, list[np.ndarray]]:
    """Each video's kept frame edge as unit rays in the world (x east, y north, z
    up), one (4 * `EDGE_POINTS`, 3) array per camera it records, rounding the frame
    clockwise from its top left: the top edge, the right, the bottom, the left.

    Args:
        rig: Its videos and cameras.
        cameras: Each camera's rotation from camera to world, OpenCV's frame.
    """
    frames: dict[str, list[np.ndarray]] = {}
    for video, (names, cut) in _videos(rig).items():
        left, right, top, bottom = _margins_px(cut)
        for name in names:
            camera = rig.rendered[name]
            w, h, n = camera.width_px / 2, camera.height_px / 2, EDGE_POINTS
            xs = np.linspace(-w + left, w - right, n)
            ys = np.linspace(-h + top, h - bottom, n)  # rows count down
            x = np.concatenate([xs, np.full(n, xs[-1]), xs[::-1], np.full(n, xs[0])])
            y = np.concatenate([np.full(n, ys[0]), ys, np.full(n, ys[-1]), ys[::-1]])
            f = camera.focal_px
            rays = np.column_stack([x / f, y / f, np.ones_like(x)]) @ cameras[name].T
            unit = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            frames.setdefault(video, []).append(unit)
    return frames


def _bearing_deg(rays: np.ndarray, near_deg: float) -> np.ndarray:
    """Each of (n, 3) unit rays' bearing, unwrapped along the edge it rounds from
    the turn of `near_deg`, so the edges of a stern pair stay together.
    """
    b = np.degrees(np.unwrap(np.arctan2(rays[:, 0], rays[:, 1])))
    return b - 360.0 * round((b[0] - near_deg) / 360.0)


def _elevation_deg(rays: np.ndarray) -> np.ndarray:
    """Each of (n, 3) unit rays' elevation."""
    return np.degrees(np.arcsin(rays[:, 2]))


@dataclass(frozen=True, slots=True)
class _Unrolled:
    """The panorama's scale: bearing and elevation, linear in both, onto `plot`."""

    plot: tuple[int, int, int, int]
    bearing_deg: tuple[float, float]
    elevation_deg: tuple[float, float]
    # The rig's heading at the start, which sets the branch every edge unwraps on.
    near_deg: float

    def _px(self, b: np.ndarray, e: np.ndarray) -> np.ndarray:
        """Bearings `b` and elevations `e` as pixels, one row each."""
        x0, y0, x1, y1 = self.plot
        (low_b, high_b), (low_e, high_e) = self.bearing_deg, self.elevation_deg
        return np.column_stack(
            [
                x0 + (b - low_b) / (high_b - low_b) * (x1 - x0),
                y0 + (high_e - e) / (high_e - low_e) * (y1 - y0),
            ]
        )

    def xy(self, b: float, e: float) -> tuple[float, float]:
        ((x, y),) = self._px(np.array([b]), np.array([e]))
        return float(x), float(y)

    def polygon(self, b: np.ndarray, e: np.ndarray) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self._px(b, e)]


def _frames_layer(
    size: tuple[int, int],
    unrolled: _Unrolled,
    frames: dict[str, list[np.ndarray]],
    colours: dict[str, Colour],
    sweep_deg: list[float],
) -> Image.Image:
    """Each frame shaded over every bearing the turn sweeps it to, then outlined
    where it starts. A frame past +-180 shows at both ends.
    """
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    outlines = []
    for name, edges in frames.items():
        swept = Image.new("L", size, 0)
        sweeping = ImageDraw.Draw(swept)
        for rays in edges:
            b, e = _bearing_deg(rays, unrolled.near_deg), _elevation_deg(rays)
            for wrap in (-360.0, 0.0, 360.0):
                for d in sweep_deg:
                    sweeping.polygon(unrolled.polygon(b + d + wrap, e), fill=255)
                start = unrolled.polygon(b + wrap, e)
                outlines.append(([*start, start[0]], colours[name]))
        layer.paste((*colours[name], FRAME_ALPHA), mask=swept)
    pen = ImageDraw.Draw(layer)
    for points, colour in outlines:
        pen.line(points, fill=colour, width=LINE_PX, joint="curve")
    return layer


def _axes(draw: ImageDraw.ImageDraw, unrolled: _Unrolled) -> None:
    """A gridline at every 30 deg of bearing, 45 all round, and the horizon."""
    x0, y0, x1, y1 = unrolled.plot
    low_b, high_b = unrolled.bearing_deg
    step = 45 if high_b - low_b >= ALL_ROUND_DEG else 30
    for b in range(math.ceil(low_b / step) * step, math.floor(high_b) + 1, step):
        label = round(wrapped_deg(b)) if abs(b) > 180 else b
        _tick(draw, unrolled.xy(b, 0.0)[0], y0, y1, f"{label:+d}°")
    _, horizon = unrolled.xy(0.0, 0.0)
    draw.line([(x0, horizon), (x1, horizon)], fill=INK, width=K)
    draw.text((x1, horizon - PAD_PX), "horizon", font=_font(), fill=INK, anchor="rd")


def _panorama(course: _Course, cameras: dict[str, np.ndarray]) -> Image.Image:
    """What the rig sees, unrolled: bearing across, elevation up (equirectangular),
    over every bearing the turn sweeps. Each frame is outlined where it starts and
    shaded over the turn; each buoy and the container stand where they are.
    """
    size = (INNER_PX - ABOVE_W_PX - GAP_PX, TOP_PX)
    tile = _tile(
        size,
        "2  Panorama",
        "outline: at the start  ·  shade: the bearings the turn sweeps",
    )
    rig_at = rig_track(course.scenario, course.name)
    heading_deg = math.degrees(rig_at(0.0).heading_rad)
    times = course.scenario.outputs.times_s
    turned = [math.degrees(rig_at(t).heading_rad) - heading_deg for t in times]
    # The turn is continuous, so it passes every whole degree between its ends,
    # whatever it covers between two frames: one per degree, as a frame is wider than
    # a degree, leaves the shade no gaps, and a long hold draws once.
    sweep = [float(d) for d in range(round(min(turned)), round(max(turned)) + 1)]
    frames = frame_edges(course.rig, cameras)
    edges = [rays for rays_of in frames.values() for rays in rays_of]
    elevations = np.concatenate([_elevation_deg(rays) for rays in edges])
    # The posed cameras ride `rise_m` above the rig's height, and so do these.
    rise_m = course.scenario.ownship.rise_m

    def up_deg(range_m: float, above_m: float) -> float:
        return elevation_deg(course.rig, range_m, above_m - rise_m)

    lowest = min(up_deg(r.range_m, 0.0) for r in course.rings)
    low_e = min(float(elevations.min()), lowest) - UP_PAD_DEG
    high_e = float(elevations.max()) + UP_PAD_DEG
    across = [_bearing_deg(rays, heading_deg) for rays in edges]
    low_b = min(float(b.min()) for b in across) + min(turned) - ACROSS_PAD_DEG
    high_b = max(float(b.max()) for b in across) + max(turned) + ACROSS_PAD_DEG
    if high_b - low_b >= ALL_ROUND_DEG:
        low_b, high_b = -180.0, 180.0
    plot = (RING_LABEL_PX, HEAD_PX, size[0], size[1] - FOOT_PX)
    unrolled = _Unrolled(plot, (low_b, high_b), (low_e, high_e), heading_deg)
    layer = _frames_layer(size, unrolled, frames, course.colours, sweep).crop(plot)
    tile.paste(layer, plot[:2], mask=layer)

    draw = ImageDraw.Draw(tile)
    _axes(draw, unrolled)
    above_m = buoy_above_m()
    box = asset(CONTAINER, Debris)
    for o in course.objects:
        range_m, off_deg = course.sight(o)[0]
        # The object's copy, a turn either way, that falls in the plot.
        b = next(
            (
                heading_deg + off_deg + w
                for w in (0.0, -360.0, 360.0)
                if low_b <= heading_deg + off_deg + w <= high_b
            ),
            None,
        )
        if b is None:
            continue
        if o.asset == BUOY:
            _dot(draw, *unrolled.xy(b, up_deg(range_m, above_m)), o)
            continue
        half = half_deg(box.length_m, range_m)
        tall_m = CONTAINER_TALL_M - box.draught_m
        top = unrolled.xy(b - half, up_deg(range_m, tall_m))
        right, waterline = unrolled.xy(b + half, up_deg(range_m, 0.0))
        draw.rectangle((*top, right, min(waterline, plot[3])), fill=INK)
    # Far rings crowd the horizon: each label only where it clears the last.
    last = -math.inf
    for ring in course.rings:
        _, y = unrolled.xy(0.0, up_deg(ring.range_m, above_m))
        if abs(y - last) >= FONT_PX:
            draw.text(
                (0, y), _metres(ring.range_m), font=_font(), fill=MUTED, anchor="lm"
            )
            last = y
    return tile


def _signals(course: _Course, posed: Posed) -> dict[str, np.ndarray]:
    """The ownship's heading, pitch and roll, and a pan-tilt unit's pan and tilt, at
    each frame.
    """
    times = course.scenario.outputs.times_s
    heading = np.unwrap([posed.path(t).heading_rad for t in times])
    attitude = np.array([posed.attitude_deg(t) for t in times])
    pitch, roll = attitude[:, 0], attitude[:, 1]
    signals = {"heading": np.degrees(heading), "pitch": pitch, "roll": roll}
    if course.rig.ptu is not None:
        panned = panned_deg(course.scenario.turn, course.name)
        signals["pan"] = np.array([panned(t) for t in times])
        signals["tilt"] = held_tilt_deg(
            course.scenario, course.name, posed.attitude_deg
        )
    return signals


def _over_time(course: _Course, signals: dict[str, np.ndarray]) -> Image.Image:
    """The turn: a row for each ring and video it serves, lit while the video's
    bearings hold a buoy of the ring whole, then a strip for each attitude.

    The bar reads bearing alone, as `preview.held` does: a deck pitched past a
    fixed video's height can drop the buoy out of it, which only the preview's
    labels show.
    """
    placed = course.scenario.model_copy(update={"objects": course.objects})
    held = preview.held(placed)
    times = course.scenario.outputs.times_s
    rows_px = ROW_PX * len(held)
    height = HEAD_PX + rows_px + GAP_PX + SIGNAL_PX * len(signals) + FOOT_PX
    tile = _tile(
        (INNER_PX, height),
        "3  Timeline",
        "bar: holds a buoy of the ring whole, by bearing",
    )
    draw = ImageDraw.Draw(tile)

    # Out to the end of the last frame, which a bar on it reaches.
    span_s = course.scenario.outputs.span_s

    def x(t_s: float) -> float:
        return LEFT_PX + t_s / span_s * (INNER_PX - LEFT_PX)

    for t in range(0, int(times[-1]) + 1, 10):
        _tick(draw, x(t), HEAD_PX, height - FOOT_PX, f"{t} s")
    for k, ((ring, view), lit) in enumerate(held.items()):
        y = HEAD_PX + ROW_PX * k
        colour = course.colours[view.name]
        label = f"{_metres(ring.range_m)}  {view.name}"
        draw.text((0, y), label, font=_font(), fill=colour)
        frame = 0
        for on, run in itertools.groupby(lit):
            n = len(list(run))
            if on:
                end = times[frame + n] if frame + n < len(times) else span_s
                bar = (x(times[frame]), y + 4 * K, x(end), y + ROW_PX - 6 * K)
                draw.rounded_rectangle(bar, 6 * K, fill=colour)
            frame += n
    y = HEAD_PX + rows_px + GAP_PX
    for name, values in signals.items():
        lo, hi = float(values.min()), float(values.max())
        mid, half = (lo + hi) / 2, max((hi - lo) / 2, 0.5) * 1.15
        top, low = y + 10 * K, y + SIGNAL_PX - 10 * K
        draw.line([(LEFT_PX, y), (INNER_PX, y)], fill=GRID, width=K)
        draw.text((0, top), name, font=_font("Medium"), fill=INK)
        span = f"{lo:+.1f}° to {hi:+.1f}°"
        draw.text((0, top + FONT_PX + 4 * K), span, font=_font(), fill=MUTED)
        trace = [
            (x(t), low - (v - mid + half) / (2 * half) * (low - top))
            for t, v in zip(times, values, strict=True)
        ]
        draw.line(trace, fill=INK, width=LINE_PX, joint="curve")
        y += SIGNAL_PX
    return tile


def draw(scenario: Scenario, out: Path) -> Path:
    """Draw `scenario`'s course as `course.course` places it, not the objects the
    scenario stores.
    """
    course = _course(scenario)
    posed = posed_at_start(scenario, course.name)
    sheet = canvas.above(
        [
            _header(course),
            canvas.beside(
                [_from_above(course), _panorama(course, posed.cameras)],
                gap_px=GAP_PX,
                matte=MATTE,
            ),
            _over_time(course, _signals(course, posed)),
        ],
        gap_px=GAP_PX,
        matte=MATTE,
    )
    framed = ImageOps.expand(sheet, MARGIN_PX, MATTE)
    framed.resize(
        (framed.width // K, framed.height // K), Image.Resampling.LANCZOS
    ).save(out)
    return out
