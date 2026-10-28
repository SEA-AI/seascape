"""A colour sensor and its ISP, as compositor nodes.

One tree serves every camera of a band, so each camera's settings are named value
nodes that `respond` sets before its render.
"""

import math

import bpy
import numpy as np

from seascape.blender.blend import _math, _vector
from seascape.config import Camera
from seascape.physics import colour

type Socket = bpy.types.NodeSocket


def _value(tree: bpy.types.NodeTree, name: str, value: float = 0.0) -> Socket:
    node = tree.nodes.new("ShaderNodeValue")
    node.name = name
    node.outputs["Value"].default_value = value
    return node.outputs["Value"]


def _set(tree: bpy.types.NodeTree, name: str, value: float) -> None:
    tree.nodes[name].outputs["Value"].default_value = value


def _switch(tree: bpy.types.NodeTree, name: str, off: Socket, on: Socket) -> Socket:
    """`on` or `off` for the whole frame, as `_toggle` sets it: a pick, where a blend
    by 0 would turn an inf in the unpicked image into a nan."""
    node = tree.nodes.new("CompositorNodeSwitch")
    node.name = name
    tree.links.new(off, node.inputs["Off"])
    tree.links.new(on, node.inputs["On"])
    return node.outputs["Image"]


def _toggle(tree: bpy.types.NodeTree, name: str, on: bool) -> None:
    tree.nodes[name].inputs["Switch"].default_value = on


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
    return _vector(
        tree, "ADD", _vector(tree, "SCALE", image, 0.0), node.outputs["Image"]
    )


def _rgb(tree: bpy.types.NodeTree, r: Socket, g: Socket, b: Socket) -> Socket:
    node = tree.nodes.new("CompositorNodeCombineColor")
    for socket, channel in zip(("Red", "Green", "Blue"), (r, g, b), strict=True):
        tree.links.new(channel, node.inputs[socket])
    return node.outputs["Image"]


def _channels(tree: bpy.types.NodeTree, image: Socket) -> tuple[Socket, ...]:
    node = tree.nodes.new("CompositorNodeSeparateColor")
    tree.links.new(image, node.inputs["Image"])
    return node.outputs["Red"], node.outputs["Green"], node.outputs["Blue"]


def _bilinear(tree: bpy.types.NodeTree, image: Socket) -> Socket:
    """Bilinear interpolation's kernels with the centre at 1: red and blue through
    [1 2 1] / 2 across and down, green through its four neighbours at a quarter."""
    across = _vector(tree, "ADD", _shift(tree, image, 1, 0), _shift(tree, image, -1, 0))
    rows = _vector(tree, "MULTIPLY_ADD", across, (0.5, 0.5, 0.5), image)
    filled = _vector(
        tree,
        "MULTIPLY_ADD",
        _vector(tree, "ADD", _shift(tree, rows, 0, 1), _shift(tree, rows, 0, -1)),
        (0.5, 0.5, 0.5),
        rows,
    )
    cross = _vector(
        tree,
        "ADD",
        across,
        _vector(tree, "ADD", _shift(tree, image, 0, 1), _shift(tree, image, 0, -1)),
    )
    _, centre_g, _ = _channels(tree, image)
    _, cross_g, _ = _channels(tree, cross)
    fill_r, _, fill_b = _channels(tree, filled)
    return _rgb(
        tree, fill_r, _math(tree, "MULTIPLY_ADD", cross_g, 0.25, centre_g), fill_b
    )


def _electrons(
    tree: bpy.types.NodeTree, x: Socket, shot: Socket, well: Socket, noise: Socket
) -> Socket:
    """What a well collects of a mean `x`, with shot noise, up to `well`."""
    lit = _math(tree, "MINIMUM", _math(tree, "MAXIMUM", x, 0.0), well)
    spread = _math(tree, "SQRT", _math(tree, "MULTIPLY", lit, shot))
    return _math(tree, "MINIMUM", _math(tree, "MULTIPLY_ADD", spread, noise, x), well)


def _read(
    tree: bpy.types.NodeTree,
    x: Socket,
    w: Socket,
    seeded: Socket,
    shot: Socket,
    well: Socket,
    dark: Socket,
    levels: Socket,
) -> Socket:
    """One colour's read of a mean `x`, in units of white. Poissonian-Gaussian (Foi
    et al. 2008): shot noise of variance shot * x, then the well fills, then dark
    noise of standard deviation `dark`."""
    shot_noise, dark_noise = _gaussian(tree, seeded, w)
    collected = _electrons(tree, x, shot, well, shot_noise)
    noisy = _math(tree, "MULTIPLY_ADD", dark, dark_noise, collected)
    counts = _math(tree, "ROUND", _math(tree, "MULTIPLY", noisy, levels))
    return _math(
        tree,
        "MINIMUM",
        _math(tree, "MAXIMUM", _math(tree, "DIVIDE", counts, levels), 0.0),
        1.0,
    )


def _merge(
    tree: bpy.types.NodeTree,
    long: Socket,
    short: Socket,
    ratio: Socket,
    shot: Socket,
    dark_variance: Socket,
    levels: Socket,
) -> Socket:
    """Granados et al. 2010: each exposure's estimate z / t weighted by the inverse of
    its variance, which the noise model gives at the read, with t = 1 and 1 / ratio;
    a long read within three times a saturated read's spread of white, here at most
    the dark noise, counts as saturated and is left out. The ADC's rounding,
    (1 / levels)^2 / 12, keeps a noiseless read finite."""
    rounding = _math(tree, "DIVIDE", 1.0 / 12, _math(tree, "MULTIPLY", levels, levels))
    floor = _math(tree, "ADD", dark_variance, rounding)
    unsaturated = _math(
        tree,
        "LESS_THAN",
        long,
        _math(tree, "MULTIPLY_ADD", _math(tree, "SQRT", dark_variance), -3.0, 1.0),
    )
    long_weight = _math(
        tree,
        "DIVIDE",
        unsaturated,
        _math(tree, "MULTIPLY_ADD", long, shot, floor),
    )
    short_weight = _math(
        tree,
        "DIVIDE",
        1.0,
        _math(
            tree,
            "MULTIPLY",
            _math(tree, "MULTIPLY_ADD", short, shot, floor),
            _math(tree, "MULTIPLY", ratio, ratio),
        ),
    )
    return _math(
        tree,
        "DIVIDE",
        _math(
            tree,
            "MULTIPLY_ADD",
            long_weight,
            long,
            _math(
                tree,
                "MULTIPLY",
                short_weight,
                _math(tree, "MULTIPLY", short, ratio),
            ),
        ),
        _math(tree, "ADD", long_weight, short_weight),
    )


def _gaussian(
    tree: bpy.types.NodeTree, pixel: Socket, w: Socket
) -> tuple[Socket, Socket]:
    """Two independent standard normals per pixel and `w`: Box & Muller 1958 from
    two uniforms."""
    uniforms = []
    for offset in (0.0, 0.5):
        noise = tree.nodes.new("ShaderNodeTexWhiteNoise")
        noise.noise_dimensions = "4D"
        tree.links.new(pixel, noise.inputs["Vector"])
        tree.links.new(_math(tree, "ADD", w, offset), noise.inputs["W"])
        uniforms.append(noise.outputs["Value"])
    # 1 - u is in (0, 1], and Blender's log of 0 is 0.
    radius = _math(
        tree,
        "SQRT",
        _math(
            tree,
            "MULTIPLY",
            _math(tree, "LOGARITHM", _math(tree, "SUBTRACT", 1.0, uniforms[0]), math.e),
            -2.0,
        ),
    )
    angle = _math(tree, "MULTIPLY", uniforms[1], 2 * math.pi)
    return (
        _math(tree, "MULTIPLY", radius, _math(tree, "COSINE", angle)),
        _math(tree, "MULTIPLY", radius, _math(tree, "SINE", angle)),
    )


def chain(
    tree: bpy.types.NodeTree, image: Socket, pixel: Socket, grey: Socket
) -> Socket:
    """`image` as the sensor reads it out, in units of its white; `pixel` is the
    image's integer pixel coordinates, and `grey` where auto-exposure put the
    log-average, which HDR's tone map keeps."""
    xy = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(pixel, xy.inputs["Vector"])
    # RGGB from the bottom-left pixel; no phase changes a statistic.
    odd_x = _math(tree, "FLOORED_MODULO", xy.outputs["X"], 2.0)
    odd_y = _math(tree, "FLOORED_MODULO", xy.outputs["Y"], 2.0)
    blue = _math(tree, "MULTIPLY", odd_x, odd_y)
    red = _math(
        tree,
        "MULTIPLY",
        _math(tree, "SUBTRACT", 1.0, odd_x),
        _math(tree, "SUBTRACT", 1.0, odd_y),
    )
    green = _math(tree, "SUBTRACT", 1.0, _math(tree, "ADD", red, blue))
    seeded = _vector(
        tree, "ADD", pixel, _combine(tree, 0.0, 0.0, _value(tree, "sensor.seed"))
    )
    frame = tree.nodes.new("CompositorNodeSceneTime").outputs["Frame"]
    shot, well = _value(tree, "sensor.shot"), _value(tree, "sensor.well", 1.0)
    dark, levels = _value(tree, "sensor.dark"), _value(tree, "sensor.levels", 4095.0)
    ratio = _value(tree, "sensor.ratio", 1.0)
    longs, merged = [], []
    for index, x in enumerate(_channels(tree, image)):
        # An average of n sites divides the variances by n.
        sites = _value(tree, f"sensor.sites_{'rgb'[index]}", 1.0)
        shot_c = _math(tree, "DIVIDE", shot, sites)
        dark_c = _math(tree, "DIVIDE", dark, sites)
        noise = (seeded, shot_c, well, _math(tree, "SQRT", dark_c), levels)
        # Two normals per colour, exposure and frame: W at 8f + 2i + e and + 0.5.
        w = _math(tree, "MULTIPLY_ADD", frame, 8.0, 2.0 * index)
        long = _read(tree, x, w, *noise)
        short = _read(
            tree,
            _math(tree, "DIVIDE", x, ratio),
            _math(tree, "ADD", w, 1.0),
            *noise,
        )
        longs.append(long)
        merged.append(_merge(tree, long, short, ratio, shot_c, dark_c, levels))
    raw = _switch(tree, "sensor.hdr", _combine(tree, *longs), _combine(tree, *merged))
    # A sensor at the output's pitch reads one colour a pixel: its sites, then a
    # bilinear demosaic, normalized (Knutsson & Westin 1993) so a border site
    # averages only the neighbours it has. A finer one, demosaiced and scaled down
    # by the ISP, reads each colour averaged over its sites.
    masks = _rgb(tree, red, green, blue)
    sites = _vector(tree, "MULTIPLY", masks, raw)
    demosaiced = _vector(tree, "DIVIDE", _bilinear(tree, sites), _bilinear(tree, masks))
    demosaiced = _switch(tree, "sensor.mosaic", raw, demosaiced)
    gains = _combine(
        tree,
        _value(tree, "sensor.wb_r", 1.0),
        _value(tree, "sensor.wb_g", 1.0),
        _value(tree, "sensor.wb_b", 1.0),
    )
    balanced = _vector(tree, "MULTIPLY", demosaiced, gains)
    balanced = _switch(
        tree, "sensor.tone", _clip(tree, balanced), _reinhard(tree, balanced, grey)
    )
    # Unsharp mask over a pixel's Gaussian.
    blur = tree.nodes.new("CompositorNodeBlur")
    blur.inputs["Type"].default_value = "Gaussian"
    blur.inputs["Size"].default_value = (1.0, 1.0)
    tree.links.new(balanced, blur.inputs["Image"])
    detail = _vector(tree, "SUBTRACT", balanced, blur.outputs["Image"])
    sharpened = _vector(
        tree,
        "ADD",
        balanced,
        _vector(tree, "SCALE", detail, _value(tree, "sensor.sharpen")),
    )
    return _clip(tree, sharpened)


def _reinhard(tree: bpy.types.NodeTree, image: Socket, grey: Socket) -> Socket:
    """Reinhard et al. 2002's global operator on luminance, L / (1 + L), with the
    log-average at `grey`: Blender's Tonemap compresses each channel, which bleaches
    a bright colour."""
    # IEC 61966-2-1: the luminance of linear sRGB.
    luminance = _vector(tree, "DOT_PRODUCT", image, (0.2126, 0.7152, 0.0722))
    # Blender's log of 0 is 0, which would count black as a luminance of 1.
    lit = _math(tree, "MAXIMUM", luminance, 1e-6)
    levels = tree.nodes.new("CompositorNodeLevels")
    levels.inputs["Channel"].default_value = "Luminance"
    tree.links.new(_math(tree, "LOGARITHM", lit, math.e), levels.inputs["Image"])
    log_average = _math(tree, "EXPONENT", levels.outputs["Mean"])
    # The key a, with a / (1 + a) = grey, is infinite at a grey of 1, so cap it.
    grey = _math(tree, "MINIMUM", grey, 0.99)
    key = _math(tree, "DIVIDE", grey, _math(tree, "SUBTRACT", 1.0, grey))
    scaled = _math(tree, "MULTIPLY", lit, _math(tree, "DIVIDE", key, log_average))
    mapped = _math(tree, "DIVIDE", scaled, _math(tree, "ADD", scaled, 1.0))
    return _vector(tree, "SCALE", image, _math(tree, "DIVIDE", mapped, lit))


def _combine(tree: bpy.types.NodeTree, *xyz: Socket | float) -> Socket:
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    for socket, value in zip(node.inputs, xyz, strict=True):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value
    return node.outputs["Vector"]


def _clip(tree: bpy.types.NodeTree, image: Socket) -> Socket:
    return _vector(
        tree,
        "MINIMUM",
        _vector(tree, "MAXIMUM", image, (0.0, 0.0, 0.0)),
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
    _set(tree, "sensor.shot", gain / full_well_e)
    # The well holds full_well_e: the ADC's white times the gain.
    _set(tree, "sensor.well", gain)
    _set(tree, "sensor.dark", (gain * sensor.dark_noise_e / full_well_e) ** 2)
    _set(tree, "sensor.levels", 2**sensor.adc_bits - 1)
    _set(tree, "sensor.ratio", sensor.hdr_ratio)
    for switch in ("sensor.hdr", "sensor.tone"):
        _toggle(tree, switch, sensor.hdr_ratio > 1.0)
    # Float32 holds integers exactly to 2^24.
    _set(tree, "sensor.seed", float(noise.integers(2**24)))
    gains = colour.white_balance_gains(camera.processing.white_balance_k)
    # dcraw's: the smallest gain is 1, so what clipped white stays white.
    for channel, gain in zip("rgb", gains, strict=True):
        _set(tree, f"sensor.wb_{channel}", gain / min(gains))
    _set(tree, "sensor.sharpen", camera.processing.sharpen)
    # Sites of the sensor across an output pixel. ponytail: below 2 the sensor
    # mosaics at the output's pitch; a scale between would need the mosaic resampled.
    k = (sensor.width_px or camera.width_px) / camera.width_px
    _toggle(tree, "sensor.mosaic", k < 2)
    for channel, share in zip("rgb", (0.25, 0.5, 0.25), strict=True):
        _set(tree, f"sensor.sites_{channel}", 1.0 if k < 2 else share * k * k)
