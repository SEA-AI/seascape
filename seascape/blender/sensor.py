"""A colour sensor and its ISP, as compositor nodes.

One tree serves every camera of a band, so each camera's settings are named value
nodes that `respond` sets before its render.
"""

import math

import bpy
import numpy as np

from seascape.blender.blend import (
    Num,
    combine,
    feed,
    node,
    scalar,
    set_value,
    switch,
    toggle,
    value,
    vector,
)
from seascape.config import Camera, Processing
from seascape.physics import colour

type Socket = bpy.types.NodeSocket


def _shift(tree: bpy.types.NodeTree, image: Socket, x: int, y: int) -> Socket:
    node = tree.nodes.new("CompositorNodeTranslate")
    node.inputs["X"].default_value, node.inputs["Y"].default_value = x, y
    node.inputs["Interpolation"].default_value = "Nearest"
    # Repeat would move the image's domain, and the frame with it, by the shift.
    node.inputs["Extension X"].default_value = "Extend"
    node.inputs["Extension Y"].default_value = "Extend"
    tree.links.new(image, node.inputs["Image"])
    # An operation takes its first input's domain: the image's own, not the shifted
    # one, whose first row or column would otherwise copy its neighbour.
    return vector(tree, "ADD", vector(tree, "SCALE", image, 0.0), node.outputs["Image"])


def _rgb(tree: bpy.types.NodeTree, r: Socket, g: Socket, b: Socket) -> Socket:
    combined = node(
        tree, "CompositorNodeCombineColor", {"Red": r, "Green": g, "Blue": b}
    )
    return combined.outputs["Image"]


def _channels(tree: bpy.types.NodeTree, image: Socket) -> tuple[Socket, ...]:
    split = node(tree, "CompositorNodeSeparateColor", {"Image": image})
    return split.outputs["Red"], split.outputs["Green"], split.outputs["Blue"]


def _bilinear(tree: bpy.types.NodeTree, image: Socket) -> Socket:
    """Bilinear interpolation's kernels with the centre at 1: red and blue through
    [1 2 1] / 2 across and down, green through its four neighbours at a quarter."""
    across = vector(tree, "ADD", _shift(tree, image, 1, 0), _shift(tree, image, -1, 0))
    rows = vector(tree, "MULTIPLY_ADD", across, (0.5, 0.5, 0.5), image)
    filled = vector(
        tree,
        "MULTIPLY_ADD",
        vector(tree, "ADD", _shift(tree, rows, 0, 1), _shift(tree, rows, 0, -1)),
        (0.5, 0.5, 0.5),
        rows,
    )
    cross = vector(
        tree,
        "ADD",
        across,
        vector(tree, "ADD", _shift(tree, image, 0, 1), _shift(tree, image, 0, -1)),
    )
    _, centre_g, _ = _channels(tree, image)
    _, cross_g, _ = _channels(tree, cross)
    fill_r, _, fill_b = _channels(tree, filled)
    return _rgb(
        tree, fill_r, scalar(tree, "MULTIPLY_ADD", cross_g, 0.25, centre_g), fill_b
    )


def _electrons(x: Num, shot: Num, well: Num, noise: Num) -> Num:
    """What a well collects of a mean `x`, with shot noise, up to `well`."""
    lit = x.op("MAXIMUM", 0.0).op("MINIMUM", well)
    spread = (lit * shot).op("SQRT")
    return spread.madd(noise, x).op("MINIMUM", well)


def _read(
    tree: bpy.types.NodeTree,
    x: Num,
    w: Num,
    seeded: Socket,
    shot: Num,
    well: Num,
    dark: Num,
    levels: Num,
) -> Num:
    """One colour's read of a mean `x`, in units of white. Poissonian-Gaussian (Foi
    et al. 2008): shot noise of variance shot * x, then the well fills, then dark
    noise of standard deviation `dark`."""
    shot_noise, dark_noise = _gaussian(tree, seeded, w)
    noisy = dark.madd(dark_noise, _electrons(x, shot, well, shot_noise))
    counts = (noisy * levels).op("ROUND")
    return (counts / levels).op("MAXIMUM", 0.0).op("MINIMUM", 1.0)


def _unsaturated(read: Num, dark_variance: Num) -> Num:
    """1 where `read` did not saturate, else 0: a read within three times a saturated
    read's spread of white, here at most the dark noise, counts as saturated."""
    return read.op("LESS_THAN", dark_variance.op("SQRT").madd(-3.0, 1.0))


def _merge(
    long: Num,
    short: Num,
    unsaturated: Num,
    ratio: Num,
    shot: Num,
    dark_variance: Num,
    levels: Num,
) -> Num:
    """Granados et al. 2010: each exposure's estimate z / t weighted by the inverse of
    its variance, which the noise model gives at the read, with t = 1 and 1 / ratio;
    a long read that saturated is left out. The ADC's rounding,
    (1 / levels)^2 / 12, keeps a noiseless read finite."""
    floor = dark_variance + (1.0 / 12) / (levels * levels)
    long_weight = unsaturated / long.madd(shot, floor)
    short_weight = 1.0 / (short.madd(shot, floor) * (ratio * ratio))
    return long_weight.madd(long, short_weight * (short * ratio)) / (
        long_weight + short_weight
    )


def _gaussian(tree: bpy.types.NodeTree, pixel: Socket, w: Num) -> tuple[Num, Num]:
    """Two independent standard normals per pixel and `w`: Box & Muller 1958 from
    two uniforms."""
    uniforms = []
    for offset in (0.0, 0.5):
        noise = tree.nodes.new("ShaderNodeTexWhiteNoise")
        noise.noise_dimensions = "4D"
        tree.links.new(pixel, noise.inputs["Vector"])
        feed(tree, noise.inputs["W"], w + offset)
        uniforms.append(Num(tree, noise.outputs["Value"]))
    # 1 - u is in (0, 1], and Blender's log of 0 is 0.
    radius = ((1.0 - uniforms[0]).op("LOGARITHM", math.e) * -2.0).op("SQRT")
    angle = uniforms[1] * (2 * math.pi)
    return radius * angle.op("COSINE"), radius * angle.op("SINE")


def chain(
    tree: bpy.types.NodeTree,
    image: Socket,
    pixel: Socket,
    grey: Socket,
) -> Socket:
    """`image` as the sensor reads it out, in units of its white; `pixel` is the
    image's integer pixel coordinates, and `grey` where auto-exposure put the
    log-average, which HDR's tone map keeps."""
    xy = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(pixel, xy.inputs["Vector"])
    # RGGB from the bottom-left pixel; no phase changes a statistic.
    odd_x, odd_y = (
        Num(tree, xy.outputs[axis]).op("FLOORED_MODULO", 2.0) for axis in "XY"
    )
    blue = odd_x * odd_y
    red = (1.0 - odd_x) * (1.0 - odd_y)
    green = 1.0 - (red + blue)
    seeded = vector(
        tree, "ADD", pixel, combine(tree, 0.0, 0.0, value(tree, "sensor.seed"))
    )
    frame = Num(tree, tree.nodes.new("CompositorNodeSceneTime").outputs["Frame"])
    shot, well = (
        Num(tree, value(tree, "sensor.shot")),
        Num(tree, value(tree, "sensor.well", 1.0)),
    )
    dark, levels = (
        Num(tree, value(tree, "sensor.dark")),
        Num(tree, value(tree, "sensor.levels", 4095.0)),
    )
    ratio = Num(tree, value(tree, "sensor.ratio", 1.0))
    fixed = combine(
        tree,
        value(tree, "sensor.wb_r", 1.0),
        value(tree, "sensor.wb_g", 1.0),
        value(tree, "sensor.wb_b", 1.0),
    )
    gains = switch(
        tree, "sensor.grey_world", fixed, _grey_world(tree, image, ratio.socket)
    )
    # Auto-exposure metered the frame before the gains. Divide by the luminance the
    # gains give a grey, as a camera metering its output would. Without HDR, gains of
    # at least 1 keep clipped white white.
    luminance_gain = vector(tree, "DOT_PRODUCT", gains, colour.LUMINANCE)
    image = vector(tree, "SCALE", image, scalar(tree, "DIVIDE", 1.0, luminance_gain))
    longs, merged = [], []
    for index, x in enumerate(Num(tree, c) for c in _channels(tree, image)):
        # An average of n sites divides the variances by n.
        averaged = Num(tree, value(tree, f"sensor.sites_{'rgb'[index]}", 1.0))
        shot_c, dark_c = shot / averaged, dark / averaged
        noise = (seeded, shot_c, well, dark_c.op("SQRT"), levels)
        # Two normals per colour, exposure and frame: W at 8f + 2i + e and + 0.5.
        w = frame.madd(8.0, 2.0 * index)
        long = _read(tree, x, w, *noise)
        short = _read(tree, x / ratio, w + 1.0, *noise)
        longs.append(long.socket)
        merged.append(
            _merge(
                long=long,
                short=short,
                unsaturated=_unsaturated(long, dark_c),
                ratio=ratio,
                shot=shot_c,
                dark_variance=dark_c,
                levels=levels,
            ).socket
        )
    raw = switch(tree, "sensor.hdr", combine(tree, *longs), combine(tree, *merged))
    # A sensor at the output's pitch reads one colour a pixel: its sites, then a
    # bilinear demosaic, normalized (Knutsson & Westin 1993) so a border site
    # averages only the neighbours it has. A finer one, demosaiced and scaled down
    # by the ISP, reads each colour averaged over its sites.
    masks = _rgb(tree, red.socket, green.socket, blue.socket)
    sites = vector(tree, "MULTIPLY", masks, raw)
    demosaiced = vector(tree, "DIVIDE", _bilinear(tree, sites), _bilinear(tree, masks))
    demosaiced = switch(tree, "sensor.mosaic", raw, demosaiced)
    balanced = vector(tree, "MULTIPLY", demosaiced, gains)
    balanced = switch(
        tree, "sensor.tone", _clip(tree, balanced), _reinhard(tree, balanced, grey)
    )
    # Unsharp mask over a pixel's Gaussian.
    blur = tree.nodes.new("CompositorNodeBlur")
    blur.inputs["Type"].default_value = "Gaussian"
    blur.inputs["Size"].default_value = (1.0, 1.0)
    tree.links.new(balanced, blur.inputs["Image"])
    detail = vector(tree, "SUBTRACT", balanced, blur.outputs["Image"])
    sharpened = vector(
        tree,
        "ADD",
        balanced,
        vector(tree, "SCALE", detail, value(tree, "sensor.sharpen")),
    )
    return _clip(tree, sharpened)


def _fixed_gains(error_k: float) -> tuple[float, ...]:
    """The fixed balance's gains for `error_k`. dcraw's: the smallest gain is 1, so
    what clipped white stays white."""
    gains = colour.white_balance_gains(error_k)
    return tuple(gain / min(gains) for gain in gains)


def _largest_fixed_gain() -> float:
    """The largest gain the fixed balance gives over `white_balance_k`'s range."""
    bounds = {
        type(m).__name__: m for m in Processing.model_fields["white_balance_k"].metadata
    }
    span = np.linspace(bounds["Ge"].ge, bounds["Le"].le, 221)
    return max(max(_fixed_gains(float(error_k))) for error_k in span)


def _grey_world(tree: bpy.types.NodeTree, image: Socket, ratio: Socket) -> Socket:
    """Gains that make the mean of `image` grey (Buchsbaum 1980). It counts only
    pixels below `ratio` whites in every colour, where the last read saturates. The
    smallest gain is 1, as in the fixed balance. Each gain is at most the largest the
    fixed balance gives.

    Limitation: it judges a pixel by the light before noise and before the frame is
    dimmed by the gains' luminance. So noise never takes a pixel across white, and
    one the dimming takes below white is still left out.
    """
    r, g, b = (
        Num(tree, channel).op("LESS_THAN", ratio) for channel in _channels(tree, image)
    )
    sampled = vector(tree, "SCALE", image, (r * g * b))
    means = []
    # Levels counts a left-out pixel as 0. That scales all three means alike, so it
    # cancels in the gains.
    for channel in ("Red", "Green", "Blue"):
        levels = tree.nodes.new("CompositorNodeLevels")
        levels.inputs["Channel"].default_value = channel
        tree.links.new(sampled, levels.inputs["Image"])
        # Blender's x / 0 is 0. With no unsaturated pixel every mean is this floor,
        # so every gain is 1.
        means.append(Num(tree, levels.outputs["Mean"]).op("MAXIMUM", 1e-6))
    mean_r, mean_g, mean_b = means
    brightest = mean_r.op("MAXIMUM", mean_g).op("MAXIMUM", mean_b)
    largest = value(tree, "sensor.wb_largest", _largest_fixed_gain())
    return combine(
        tree,
        *((brightest / mean).op("MINIMUM", largest).socket for mean in means),
    )


def log_average(tree: bpy.types.NodeTree, image: Socket) -> tuple[Socket, Socket]:
    """`image`'s luminance, kept off 0, and its log-average (Reinhard et al. 2002)."""
    luminance = node(tree, "CompositorNodeRGBToBW", {"Image": image})
    # Blender's log of 0 is 0, which would count black as a luminance of 1.
    lit = Num(tree, luminance.outputs["Val"]).op("MAXIMUM", 1e-6)
    levels = tree.nodes.new("CompositorNodeLevels")
    levels.inputs["Channel"].default_value = "Luminance"
    feed(tree, levels.inputs["Image"], lit.op("LOGARITHM", math.e))
    return lit.socket, scalar(tree, "EXPONENT", levels.outputs["Mean"])


def _reinhard(tree: bpy.types.NodeTree, image: Socket, grey: Socket) -> Socket:
    """Reinhard et al. 2002's global operator on luminance, L / (1 + L), with the
    log-average at `grey`: Blender's Tonemap compresses each channel, which bleaches
    a bright colour."""
    lit, average = (Num(tree, s) for s in log_average(tree, image))
    # The key a, with a / (1 + a) = grey, is infinite at a grey of 1, so cap it.
    capped = Num(tree, grey).op("MINIMUM", 0.99)
    key = capped / (1.0 - capped)
    scaled = lit * (key / average)
    mapped = scaled / (scaled + 1.0)
    return vector(tree, "SCALE", image, (mapped / lit))


def _clip(tree: bpy.types.NodeTree, image: Socket) -> Socket:
    return vector(
        tree,
        "MINIMUM",
        vector(tree, "MAXIMUM", image, (0.0, 0.0, 0.0)),
        (1.0, 1.0, 1.0),
    )


def respond(
    tree: bpy.types.NodeTree, camera: Camera, noise: np.random.Generator
) -> None:
    """Set the chain for `camera`, its noise seeded from `noise`."""
    sensor = camera.sensor
    full_well_e = sensor.full_well_e or math.inf
    gain = 10 ** (sensor.gain_db / 20)
    # At gain g, full_well_e / g electrons fill the ADC.
    set_value(tree, "sensor.shot", gain / full_well_e)
    # The well holds full_well_e: the ADC's white times the gain.
    set_value(tree, "sensor.well", gain)
    set_value(tree, "sensor.dark", (gain * sensor.dark_noise_e / full_well_e) ** 2)
    set_value(tree, "sensor.levels", 2**sensor.adc_bits - 1)
    set_value(tree, "sensor.ratio", sensor.hdr_ratio)
    for name in ("sensor.hdr", "sensor.tone"):
        toggle(tree, name, sensor.hdr_ratio > 1.0)
    # Float32 holds integers exactly to 2^24.
    set_value(tree, "sensor.seed", float(noise.integers(2**24)))
    toggle(tree, "sensor.grey_world", camera.processing.white_balance == "grey_world")
    gains = _fixed_gains(camera.processing.white_balance_k)
    for channel, gain in zip("rgb", gains, strict=True):
        set_value(tree, f"sensor.wb_{channel}", gain)
    set_value(tree, "sensor.sharpen", camera.processing.sharpen)
    # Sites of the sensor across an output pixel. Limitation: below 2 the sensor
    # mosaics at the output's pitch; a scale between would need the mosaic resampled.
    k = (sensor.width_px or camera.width_px) / camera.width_px
    toggle(tree, "sensor.mosaic", k < 2)
    for channel, share in zip("rgb", (0.25, 0.5, 0.25), strict=True):
        set_value(tree, f"sensor.sites_{channel}", 1.0 if k < 2 else share * k * k)
