"""A colour sensor and its ISP, as compositor nodes.

One tree serves every camera of a band, so each camera's settings are named value
nodes that `respond` sets before its render.
"""

import math

import bpy

from seascape import colour, sea
from seascape.config import Camera, substream

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
    return sea._vector(
        tree, "ADD", sea._vector(tree, "SCALE", image, 0.0), node.outputs["Image"]
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
    across = sea._vector(
        tree, "ADD", _shift(tree, image, 1, 0), _shift(tree, image, -1, 0)
    )
    rows = sea._vector(tree, "MULTIPLY_ADD", across, (0.5, 0.5, 0.5), image)
    filled = sea._vector(
        tree,
        "MULTIPLY_ADD",
        sea._vector(tree, "ADD", _shift(tree, rows, 0, 1), _shift(tree, rows, 0, -1)),
        (0.5, 0.5, 0.5),
        rows,
    )
    cross = sea._vector(
        tree,
        "ADD",
        across,
        sea._vector(tree, "ADD", _shift(tree, image, 0, 1), _shift(tree, image, 0, -1)),
    )
    _, centre_g, _ = _channels(tree, image)
    _, cross_g, _ = _channels(tree, cross)
    fill_r, _, fill_b = _channels(tree, filled)
    return _rgb(
        tree, fill_r, sea._math(tree, "MULTIPLY_ADD", cross_g, 0.25, centre_g), fill_b
    )


def _electrons(
    tree: bpy.types.NodeTree, x: Socket, shot: Socket, well: Socket, noise: Socket
) -> Socket:
    """What a well collects of a mean `x`, with shot noise, up to `well`."""
    lit = sea._math(tree, "MINIMUM", sea._math(tree, "MAXIMUM", x, 0.0), well)
    spread = sea._math(tree, "SQRT", sea._math(tree, "MULTIPLY", lit, shot))
    return sea._math(
        tree, "MINIMUM", sea._math(tree, "MULTIPLY_ADD", spread, noise, x), well
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
        tree.links.new(sea._math(tree, "ADD", w, offset), noise.inputs["W"])
        uniforms.append(noise.outputs["Value"])
    # 1 - u is in (0, 1], and Blender's log of 0 is 0.
    radius = sea._math(
        tree,
        "SQRT",
        sea._math(
            tree,
            "MULTIPLY",
            sea._math(
                tree, "LOGARITHM", sea._math(tree, "SUBTRACT", 1.0, uniforms[0]), math.e
            ),
            -2.0,
        ),
    )
    angle = sea._math(tree, "MULTIPLY", uniforms[1], 2 * math.pi)
    return (
        sea._math(tree, "MULTIPLY", radius, sea._math(tree, "COSINE", angle)),
        sea._math(tree, "MULTIPLY", radius, sea._math(tree, "SINE", angle)),
    )


def chain(tree: bpy.types.NodeTree, image: Socket, pixel: Socket) -> Socket:
    """`image` as the sensor reads it out, in units of its white; `pixel` is the
    image's integer pixel coordinates."""
    xy = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(pixel, xy.inputs["Vector"])
    # RGGB from the bottom-left pixel; no phase changes a statistic.
    odd_x = sea._math(tree, "FLOORED_MODULO", xy.outputs["X"], 2.0)
    odd_y = sea._math(tree, "FLOORED_MODULO", xy.outputs["Y"], 2.0)
    blue = sea._math(tree, "MULTIPLY", odd_x, odd_y)
    red = sea._math(
        tree,
        "MULTIPLY",
        sea._math(tree, "SUBTRACT", 1.0, odd_x),
        sea._math(tree, "SUBTRACT", 1.0, odd_y),
    )
    green = sea._math(tree, "SUBTRACT", 1.0, sea._math(tree, "ADD", red, blue))
    seeded = sea._vector(
        tree, "ADD", pixel, _combine(tree, 0.0, 0.0, _value(tree, "sensor.seed"))
    )
    frame = tree.nodes.new("CompositorNodeSceneTime").outputs["Frame"]
    shot_noise, dark_noise = _gaussian(
        tree, seeded, sea._math(tree, "MULTIPLY", frame, 2.0)
    )
    # Poissonian-Gaussian (Foi et al. 2008), in units of white: shot noise of
    # variance shot * x, then the well fills, then dark noise of variance dark. A
    # pixel reads one channel, so the channels share its draws.
    shot, well = _value(tree, "sensor.shot"), _value(tree, "sensor.well", 1.0)
    collected = [
        _electrons(tree, channel, shot, well, shot_noise)
        for channel in _channels(tree, image)
    ]
    mosaic = sea._math(
        tree,
        "ADD",
        sea._math(tree, "MULTIPLY", collected[0], red),
        sea._math(
            tree,
            "ADD",
            sea._math(tree, "MULTIPLY", collected[1], green),
            sea._math(tree, "MULTIPLY", collected[2], blue),
        ),
    )
    dark = sea._math(tree, "SQRT", _value(tree, "sensor.dark"))
    noisy = sea._math(tree, "MULTIPLY_ADD", dark, dark_noise, mosaic)
    levels = _value(tree, "sensor.levels", 4095.0)
    counts = sea._math(tree, "ROUND", sea._math(tree, "MULTIPLY", noisy, levels))
    raw = sea._math(
        tree,
        "MINIMUM",
        sea._math(tree, "MAXIMUM", sea._math(tree, "DIVIDE", counts, levels), 0.0),
        1.0,
    )
    # Normalized convolution (Knutsson & Westin 1993): the samples' bilinear over the
    # site masks', so a border site averages only the neighbours it has.
    masks = _rgb(tree, red, green, blue)
    sites = sea._vector(tree, "SCALE", masks, raw)
    demosaiced = sea._vector(
        tree, "DIVIDE", _bilinear(tree, sites), _bilinear(tree, masks)
    )
    gains = _combine(
        tree,
        _value(tree, "sensor.wb_r", 1.0),
        _value(tree, "sensor.wb_g", 1.0),
        _value(tree, "sensor.wb_b", 1.0),
    )
    balanced = _clip(tree, sea._vector(tree, "MULTIPLY", demosaiced, gains))
    # Unsharp mask over a pixel's Gaussian.
    blur = tree.nodes.new("CompositorNodeBlur")
    blur.inputs["Type"].default_value = "Gaussian"
    blur.inputs["Size"].default_value = (1.0, 1.0)
    tree.links.new(balanced, blur.inputs["Image"])
    detail = sea._vector(tree, "SUBTRACT", balanced, blur.outputs["Image"])
    sharpened = sea._vector(
        tree,
        "ADD",
        balanced,
        sea._vector(tree, "SCALE", detail, _value(tree, "sensor.sharpen")),
    )
    return _clip(tree, sharpened)


def _combine(tree: bpy.types.NodeTree, *xyz: Socket | float) -> Socket:
    node = tree.nodes.new("ShaderNodeCombineXYZ")
    for socket, value in zip(node.inputs, xyz, strict=True):
        if isinstance(value, bpy.types.NodeSocket):
            tree.links.new(value, socket)
        else:
            socket.default_value = value
    return node.outputs["Vector"]


def _clip(tree: bpy.types.NodeTree, image: Socket) -> Socket:
    return sea._vector(
        tree,
        "MINIMUM",
        sea._vector(tree, "MAXIMUM", image, (0.0, 0.0, 0.0)),
        (1.0, 1.0, 1.0),
    )


def respond(tree: bpy.types.NodeTree, camera: Camera, name: str, seed: int) -> None:
    """Set the chain for `camera`, its noise on the scenario `seed`'s substream for
    `name`."""
    sensor = camera.sensor
    full_well_e = sensor.full_well_e or math.inf
    gain = 10 ** (sensor.gain_db / 20)
    # At gain g, full_well_e / g electrons fill the ADC.
    _set(tree, "sensor.shot", gain / full_well_e)
    # The well holds full_well_e: the ADC's white times the gain.
    _set(tree, "sensor.well", gain)
    _set(tree, "sensor.dark", (gain * sensor.dark_noise_e / full_well_e) ** 2)
    _set(tree, "sensor.levels", 2**sensor.adc_bits - 1)
    # Float32 holds integers exactly to 2^24.
    _set(tree, "sensor.seed", float(substream(seed, f"sensor/{name}").integers(2**24)))
    gains = colour.white_balance_gains(camera.processing.white_balance_k)
    # dcraw's: the smallest gain is 1, so what clipped white stays white.
    for channel, gain in zip("rgb", gains, strict=True):
        _set(tree, f"sensor.wb_{channel}", gain / min(gains))
    _set(tree, "sensor.sharpen", camera.processing.sharpen)
