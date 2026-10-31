"""The sea's material, assembled from its parts."""

import math
from functools import cache
from typing import NamedTuple

import bpy
import numpy as np

from seascape.blender.blend import (
    animate,
    curve_image,
    scalar,
    vector,
)
from seascape.blender.sea.daylight import (
    _daylight,
    _glinting,
    _glitter,
    _stretchable,
    _widest,
)
from seascape.blender.sea.foam import (
    FOAM_CELLS,
    FOAM_LENGTH_M,
    FOAM_SEAM_CELLS,
    FOAM_SPACING_M,
    _bare,
    _foam_texture,
    _remembered,
    _unit,
)
from seascape.blender.sea.geometry import (
    _displace,
    _displacement,
    _flat,
    _normal,
    displaced,
    sea_reach_m,
    water,
)
from seascape.blender.sea.memory import (
    foam_frames,
    foam_grids,
    forget_foam,
    keep_foam,
    remember_foam,
)
from seascape.blender.sea.pixel import FOOTPRINT_RANGE_M, _pixel
from seascape.blender.sea.radiance import (
    Clouds,
    Light,
    SkyCurve,
    _skin_radiance,
    _thermal,
    _traced,
)
from seascape.blender.sea.roughness import _grouped, _unresolved
from seascape.blender.sea.wakes import _smoothed, _wakes
from seascape.blender.sea.wind import (
    GUST_CELLS,
    GUST_SPACING_M,
    _gathered,
    _gust,
    _langmuir_rows,
    _slick,
    surface_drift,
)
from seascape.config import Outputs, Sea
from seascape.physics.wakes import (
    Wake,
)
from seascape.physics.waves import (
    GUST_LENGTH_M,
    FoamGrid,
    Wave,
    cox_munk_slope,
    gust_slope_variance,
    resolvable,
    slick_survivors,
    turbulence_intensity,
    unresolved_slope_variances,
    von_karman_field,
)


class Seen(NamedTuple):
    """What the cameras see of the sea: their finest footprint, and the grids their
    whitecaps are remembered on."""

    finest_footprint_m: float = 0.0
    foam_grids: tuple[FoamGrid, ...] = ()


UNSEEN = Seen()


def _sea_time(tree: bpy.types.NodeTree, outputs: Outputs) -> bpy.types.NodeSocket:
    """Seconds since the first frame, modulo the span in a loop."""
    times_s = outputs.times_s
    if not outputs.loop:
        clock = tree.nodes.new("ShaderNodeValue")
        clock.name = "sea_time"
        animate(clock.outputs["Value"], "default_value", times_s, lambda t: t)
        return clock.outputs["Value"]
    # A keyed t jumps at the wrap; every omega is a multiple of 2 pi / span, so t modulo
    # the span is exact.
    span_s = outputs.span_s
    turn = [tree.nodes.new("ShaderNodeValue") for _ in range(2)]
    names, trigs = ("sea_cos", "sea_sin"), (math.cos, math.sin)
    for node, name, trig in zip(turn, names, trigs, strict=True):
        node.name = name
        animate(
            node.outputs["Value"],
            "default_value",
            times_s,
            lambda t, trig=trig: trig(2 * math.pi * t / span_s),
        )
    angle = tree.nodes.new("ShaderNodeMath")
    angle.operation = "ARCTAN2"
    clock = tree.nodes.new("ShaderNodeMath")
    clock.operation = "MULTIPLY"
    clock.name = "sea_time"
    clock.inputs["Value_001"].default_value = span_s / (2 * math.pi)
    link = tree.links.new
    link(turn[1].outputs["Value"], angle.inputs["Value"])
    link(turn[0].outputs["Value"], angle.inputs["Value_001"])
    link(angle.outputs["Value"], clock.inputs["Value"])
    return clock.outputs["Value"]


def material(
    sea: Sea,
    wind: tuple[Wave, ...],
    swell: tuple[Wave, ...],
    outputs: Outputs,
    rngs: tuple[np.random.Generator, np.random.Generator, np.random.Generator],
    wakes: tuple[Wake, ...],
    light: Light,
    seen: Seen = UNSEEN,
) -> bpy.types.Material:
    """Each pixel draws the waves it resolves and takes the rest as roughness;
    `rngs` draw its gusts, its windrows and its foam, whose whitecaps are remembered on
    the grids its cameras see. With a sky the sea is ir and reflects it; without, eo,
    glittering towards the light's sun if it has one.

    No node draws a wave a pixel of the cameras' finest footprint cannot resolve; the
    tables and statistics still describe the whole sea."""
    gust_rng, slick_rng, foam_rng = rngs
    material = bpy.data.materials.new("sea")
    tree = material.node_tree
    tree.nodes.clear()
    pixel = _pixel(tree)
    time_s = _sea_time(tree, outputs)
    speed = sea.wind_speed_mps
    drawn = (
        resolvable(wind, seen.finest_footprint_m),
        resolvable(swell, seen.finest_footprint_m),
    )
    tall, shaded = displaced(drawn)
    position = _flat(tree, time_s)
    rows = _langmuir_rows(tree, sea, position, outputs, slick_rng)
    gust_max, gust, slick, tile = 0.0, None, None, None
    if speed > 0.0:
        tile = von_karman_field(gust_rng, GUST_CELLS, GUST_SPACING_M, GUST_LENGTH_M)
        # A crossfade reaches sqrt(2) of the tile's peak at most.
        gust_max = math.sqrt(2) * np.abs(tile).max() * turbulence_intensity(speed)
        gust = _gust(tree, sea, position, outputs, curve_image("sea_gust", tile))
        slick = _slick(tree, sea, rows)
    # Made where foam shows: an image nothing reads is not saved.
    foam = cache(
        lambda: curve_image(
            "sea_foam",
            _unit(
                von_karman_field(foam_rng, FOAM_CELLS, FOAM_SPACING_M, FOAM_LENGTH_M)
            ),
        )
    )
    survivors = (
        wind
        if slick is None
        else slick_survivors(speed, wind, math.radians(sea.wind_from_deg))
    )
    damped = frozenset(wind) - frozenset(survivors)
    calm = (
        scalar(tree, "SUBTRACT", 1.0, slick) if slick is not None and damped else None
    )
    normal = _normal(tree, *shaded, position, pixel, calm, damped)
    wake = None
    if wakes:
        sea_variance = cox_munk_slope(speed) ** 2
        wake = _wakes(tree, wakes, position, pixel.along_m, foam(), sea_variance)
    smooth = _smoothed(tree, slick, wake)
    if wake is not None and wake.tilt is not None:
        normal = vector(
            tree,
            "NORMALIZE",
            vector(tree, "ADD", normal, wake.tilt),
            name="wake_normal",
        )
    unresolved = _unresolved(tree, sea, wind, swell, survivors, pixel, gust, smooth)
    whitecaps = (
        _remembered(
            tree,
            sea,
            outputs,
            position,
            pixel.across_m,
            _foam_texture(
                tree,
                sea,
                position,
                outputs,
                foam(),
                # Its own stream, so the lace's draws stay as they were.
                foam_rng.spawn(1)[0],
                pixel.along_m,
            ),
            foam_frames(sea, wind, outputs, seen.foam_grids),
            _gathered(tree, sea, pixel.along_m, rows),
        )
        if wind and seen.foam_grids
        else _bare(tree)
    )

    if light.sky is None:
        wide, narrow, tangent = _widest(tree, pixel, unresolved)
        carried: float | bpy.types.NodeSocket = 0.0
        if wind and light.sun is not None:
            # The wider spread reaches the most facets. A judgement.
            glinting = _glinting(tree, normal, light.sun, wide)
            tilt, carried = _glitter(tree, wind, position, pixel, narrow, glinting)
            normal = vector(
                tree,
                "NORMALIZE",
                vector(tree, "ADD", normal, tilt),
                name="glitter_normal",
            )
        wide = scalar(tree, "MAXIMUM", scalar(tree, "SUBTRACT", wide, carried), 0.0)
        unresolved = (
            wide,
            _stretchable(tree, wide, scalar(tree, "SUBTRACT", narrow, carried)),
        )
        surface = _daylight(tree, sea, normal, tangent, unresolved, whitecaps, wake)
    else:
        # Past the tables' widest lobe, a rough cell reads their edge.
        unresolved = _grouped(tree, sea, position, pixel, unresolved)
        # The coarsest footprint leaves the most, Cox & Munk's and all the swell's;
        # along the view a pixel takes twice the wider axis's, and the strongest gust
        # adds the most.
        axes = unresolved_slope_variances(
            speed, math.radians(sea.wind_from_deg), wind, swell, FOOTPRINT_RANGE_M[1]
        )
        slope_max = math.sqrt(
            2 * max(axes) + gust_max * gust_slope_variance(sea.wind_speed_mps)
        )
        surface = _traced(
            tree,
            *_thermal(
                tree,
                sea,
                normal,
                unresolved,
                slope_max,
                light,
                _skin_radiance(tree, sea, light.sun_up, wake),
                whitecaps[0]
                if wake is None
                else scalar(tree, "MAXIMUM", whitecaps[0], wake.foam),
            ),
            normal,
            pixel,
            unresolved,
        )
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(surface, output.inputs["Surface"])
    _displace(material, output, _displacement(tree, tall, position))
    return material


__all__ = [
    "FOAM_SEAM_CELLS",
    "UNSEEN",
    "Clouds",
    "Light",
    "Seen",
    "SkyCurve",
    "displaced",
    "foam_grids",
    "forget_foam",
    "keep_foam",
    "material",
    "remember_foam",
    "sea_reach_m",
    "surface_drift",
    "water",
]
