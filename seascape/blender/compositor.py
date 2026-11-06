"""The compositor each camera writes through, its output formats, and the GPU."""

import math

import bpy
import numpy as np

from seascape.blender import sensor
from seascape.blender.blend import (
    Num,
    combine,
    curve_image,
    node,
    scalar,
    separate,
    set_value,
    switch,
    toggle,
    value,
    vector,
)
from seascape.config import (
    Band,
    Camera,
    ImageFormat,
    Mount,
    Outputs,
)
from seascape.physics import colour, optics

JPEG_QUALITY = 95
# The Switches that turn a fisheye's image and object-index pass back.
ROLL = "lens.roll"
INDEX_ROLL = "index.roll"


# ISO 12232's saturation-based speed puts mid-grey at 10/78 of saturation; the
# log-average luminance stands for a scene's mid-grey (Reinhard et al. 2002).
MID_GREY = 10 / 78


# A lens scatters in Harvey's form, (1 + (r / r0)^2)^(-g / 2) off the direct light
# (Harvey 1976, the ABg model). Both r0, in half the image's longer side, and the
# slope g are judgements for a generic lens.
GLARE_SHOULDER, GLARE_SLOPE = 0.01, 3.0


# Judgement: the Blur node's size, in the image's longer side; a pixel of a 4K frame.
BLUR_OF_SIDE = 1 / 3840


# Blender's identifier and bit depth.
FORMATS: dict[ImageFormat, tuple[str, str]] = {
    "exr": ("OPEN_EXR", "32"),
    "png": ("PNG", "8"),
    "jpg": ("JPEG", "8"),
}


def enable_gpu() -> str | None:
    """Point Cycles at a GPU and name its backend; None leaves it on the CPU.

    Without `refresh_devices()` it stays on the CPU.
    """
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            preferences.compute_device_type = backend
        except TypeError:
            continue  # not compiled into this build
        preferences.refresh_devices()
        # `devices` lists a card once per backend; CPU alongside the GPU wins nothing.
        if any(device.type == backend for device in preferences.devices):
            for device in preferences.devices:
                device.use = device.type == backend
            return backend
    return None


def set_output(
    outputs: Outputs, band: Band, largest_px: int, graded: bool = False
) -> None:
    """Render and display settings, in the .blend, so F12 renders what `render` does.

    Args:
        outputs: The samples, format and exposure compensation to render with.
        band: The band rendered; ir always writes exr.
        largest_px: The longest side of any of the band's cameras, in pixels.
        graded: Whether any of the band's cameras has a sensor.
    """
    sc = bpy.context.scene
    # Not EEVEE: it darkens rough reflections at grazing view.
    sc.render.engine = "CYCLES"
    # Caps reflected light at 10, and LWIR radiance is tens of W m^-2 sr^-1.
    sc.eevee.clamp_surface_indirect = 0.0
    sc.cycles.device = "GPU" if enable_gpu() else "CPU"
    sc.cycles.samples = getattr(outputs.samples, band)
    # Adaptive sampling stops a pixel at a noise threshold set for a picture, not for
    # kelvin; ir's samples are what holds its grain below the NETD.
    sc.cycles.use_adaptive_sampling = band != "ir"
    # On by default. OIDN breaks the ir frame's R=G=B and blurs the waves.
    sc.cycles.use_denoising = False
    # Cycles merges tiles through a file on disk; a frame in one tile renders the same
    # pixels sooner. RNA clamps a side past its maximum, which then tiles.
    sc.cycles.tile_size = largest_px
    # The factory AgX is a film curve; a camera clips, and LWIR radiance is no picture.
    view = sc.view_settings
    view.view_transform, view.look = "Standard", "None"
    view.exposure, view.gamma = 0.0, 1.0
    # 8-bit radiance is not radiance; `render` maps 8-bit ir from the exr.
    file_format, depth = FORMATS[outputs.format if band == "eo" else "exr"]
    _compositor(outputs.exposure_compensation_ev if depth == "8" else None, graded)
    sc.render.image_settings.file_format = file_format
    sc.render.image_settings.color_depth = depth
    sc.render.image_settings.quality = JPEG_QUALITY
    sc.render.use_persistent_data = True
    # The GPU compositor, the factory's, keeps memory from every render, and a clip
    # runs out of it.
    sc.render.compositor_device = "CPU"
    # A fixed seed would hold the sample noise still while the scene moves under it.
    sc.cycles.use_animated_seed = True


def unroll(
    tree: bpy.types.NodeTree, image: bpy.types.NodeSocket, name: str
) -> bpy.types.NodeSocket:
    """An image as Cycles' polynomial fisheye draws it, turned back: the projection puts
    the scene half a turn about the optical axis.

    Args:
        tree: The compositor tree to add the turn to.
        image: The rendered pass, such as the image or the object index.
        name: The Switch's name, which `respond` turns on for a fisheye camera.
    """
    flip = node(tree, "CompositorNodeFlip", {"Image": image})
    flip.inputs["Flip X"].default_value = flip.inputs["Flip Y"].default_value = True
    return switch(tree, name, image, flip.outputs["Image"])


def _compositor(compensation_ev: float | None, graded: bool = False) -> None:
    """The render as a camera takes it.

    Args:
        compensation_ev: Stops over auto-exposure, each doubling the light; None
            passes the render as rendered.
        graded: Pass the frame through a sensor.
    """
    tree = bpy.data.node_groups.new("compositor", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    bpy.context.scene.compositing_node_group = tree
    image = unroll(tree, tree.nodes.new("CompositorNodeRLayers").outputs["Image"], ROLL)
    if compensation_ev is not None:
        image = _camera(tree, image, compensation_ev, graded)
    node(tree, "NodeGroupOutput", {"Image": image})


def _camera(
    tree: bpy.types.NodeTree,
    image: bpy.types.NodeSocket,
    compensation_ev: float,
    with_sensor: bool = False,
) -> bpy.types.NodeSocket:
    """A lens's falloff, diffraction, glare and blur, then auto-exposure, then the
    ISP's tone curve. Without a sensor the Standard view clips at white, as a sensor
    does. `respond` sets the camera.

    Limitation: each frame meters itself; a real camera's exposure lags, so add a time
    constant if a sequence flickers.

    Args:
        tree: The compositor's tree.
        image: The render's linear radiance, as it reaches the lens.
        compensation_ev: Stops over auto-exposure, which puts the frame's log-average
            at `MID_GREY`.
        with_sensor: Add the sensor and ISP after auto-exposure.
    """
    coords = node(tree, "CompositorNodeImageCoordinates", {"Image": image})
    info = node(tree, "CompositorNodeImageInfo", {"Image": image})
    # Uniform spans -1 to 1 along the longer side, so the kernel is a fixed share of
    # the frame at any resolution. Fog Glow and Bloom change shape with it.
    radius = vector(tree, "LENGTH", coords.outputs["Uniform"])
    # Natural vignetting: irradiance falls as cos^4 of the field angle, 1 / (1 +
    # tan^2)^2 (Kerr 2007). `lens.vignette` is the tangent at Uniform's 1, or 0.
    tangent = Num(tree, radius) * value(tree, "lens.vignette")
    falloff = tangent.madd(tangent, 1.0).op("POWER", -2.0)
    image = vector(tree, "SCALE", image, falloff)
    image = _diffraction(tree, image)
    shoulder = Num(tree, radius) / GLARE_SHOULDER
    harvey = shoulder.madd(shoulder, 1.0).op("POWER", -GLARE_SLOPE / 2)
    # Half the resolution: Harvey's glow is smooth, so only the sun's rim shows it. A
    # lens scatters all the light, so the threshold is 0.
    glare = node(
        tree,
        "CompositorNodeGlare",
        {"Type": "Kernel", "Quality": "Medium", "Threshold": 0.0, "Image": image},
        name="lens.veiling_glare",
    )
    kernel = next(s for s in glare.inputs if s.name == "Kernel" and s.type == "VALUE")
    tree.links.new(harvey.socket, kernel)
    x_px, y_px, _ = separate(tree, info.outputs["Resolution"])
    longer_px = scalar(tree, "MAXIMUM", x_px, y_px)
    # A share of the frame, not a pixel count, so it blurs alike at any resolution.
    share = value(tree, "lens.blur_of_side", BLUR_OF_SIDE)
    size_px = scalar(tree, "MULTIPLY", longer_px, share)
    blur = node(tree, "CompositorNodeBlur", {"Type": "Gaussian"})
    tree.links.new(combine(tree, size_px, size_px), blur.inputs["Size"])
    tree.links.new(glare.outputs["Image"], blur.inputs["Image"])
    image = blur.outputs["Image"]
    _, average = sensor.log_average(tree, image)
    gain = scalar(
        tree, "DIVIDE", 2**compensation_ev * MID_GREY, average, name="exposure"
    )
    processing = value(tree, "processing.gain", 1.0)
    gain = scalar(tree, "MULTIPLY", gain, processing)
    exposed = vector(tree, "SCALE", image, gain)
    if with_sensor:
        # A camera without a full well shares the tree, and takes the ideal frame.
        grey = scalar(tree, "MULTIPLY", processing, 2**compensation_ev * MID_GREY)
        read = sensor.chain(tree, exposed, coords.outputs["Pixel"], grey)
        exposed = switch(tree, "sensor.on", exposed, read)
    return _tone(tree, exposed)


def _tone(
    tree: bpy.types.NodeTree, image: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """A frame through the curve `respond` gives "processing.tone", which takes and
    gives sRGB-encoded values, so the Standard view writes the curve's own.

    Args:
        tree: The compositor's tree.
        image: The exposed frame, linear, white at 1.
    """
    nodes, links = tree.nodes, tree.links
    encode, decode = (nodes.new("CompositorNodeConvertColorSpace") for _ in range(2))
    encode.to_color_space = decode.from_color_space = "sRGB"
    curve = nodes.new("CompositorNodeCurveRGB")
    curve.name = "processing.tone"
    links.new(image, encode.inputs["Image"])
    links.new(encode.outputs["Image"], curve.inputs["Image"])
    links.new(curve.outputs["Image"], decode.inputs["Image"])
    return switch(tree, "processing.toned", image, decode.outputs["Image"])


def _set_curve(node: bpy.types.Node, y: np.ndarray) -> None:
    """`node`'s combined curve through `y` at `colour.TONE_IN`."""
    points = node.mapping.curves[3].points
    while len(points) > 2:
        points.remove(points[1])
    points[0].location = (0.0, float(y[0]))
    points[-1].location = (1.0, float(y[-1]))
    for x, out in zip(colour.TONE_IN[1:-1], y[1:-1], strict=True):
        points.new(float(x), float(out))
    for point in points:
        # Clamped: a monotonic curve's spline does not overshoot its knots.
        point.handle_type = "AUTO_CLAMPED"
    # Past white the curve stays at white, as the sensor clips, not its slope.
    node.mapping.extend = "HORIZONTAL"
    node.mapping.update()


def _ones_like(
    tree: bpy.types.NodeTree, image: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """A frame of ones: an operation takes its first input's domain, and a colour
    alone has none.

    Args:
        tree: The compositor's tree.
        image: The frame whose size the ones take.
    """
    return vector(tree, "ADD", vector(tree, "SCALE", image, 0.0), (1.0, 1.0, 1.0))


def _convolve(
    tree: bpy.types.NodeTree,
    image: bpy.types.NodeSocket,
    kernel: bpy.types.NodeSocket,
) -> bpy.types.NodeSocket:
    """`image` convolved with a colour `kernel`, each channel by its own."""
    convolve = node(
        tree, "CompositorNodeConvolve", {"Kernel Data Type": "Color", "Image": image}
    )
    rgb = next(s for s in convolve.inputs if s.name == "Kernel" and s.type == "RGBA")
    tree.links.new(kernel, rgb)
    return convolve.outputs["Image"]


def _diffraction(
    tree: bpy.types.NodeTree, image: bpy.types.NodeSocket
) -> bpy.types.NodeSocket:
    """`image` through the kernel `respond` puts in `lens.psf`, or as it is."""
    psf = node(tree, "CompositorNodeImage", name="lens.psf")
    kernel = psf.outputs["Image"]
    # Convolve reads black past the frame. Dividing by the convolved ones (Knutsson
    # & Westin 1993) keeps a border pixel at its level.
    spread = vector(
        tree,
        "DIVIDE",
        _convolve(tree, image, kernel),
        _convolve(tree, _ones_like(tree, image), kernel),
    )
    return switch(tree, "lens.diffract", image, spread)


def _set_diffraction(tree: bpy.types.NodeTree, camera: Camera) -> None:
    """Turn diffraction on for `camera` or off, building its kernel when it changes."""
    f_number = camera.lens.f_number
    toggle(tree, "lens.diffract", f_number is not None)
    # The Airy pattern is the lens's softness; the judgement blur stands in for it.
    set_value(tree, "lens.blur_of_side", BLUR_OF_SIDE if f_number is None else 0.0)
    psf = tree.nodes["lens.psf"]
    if f_number is None:
        if psf.image is not None:
            bpy.data.images.remove(psf.image)
        return
    assert camera.sensor.pitch_um is not None, "f_number needs sensor.pitch_um"
    sites_per_px = (camera.sensor.width_px or camera.width_px) / camera.width_px
    pitch_m = camera.sensor.pitch_um * 1e-6 * sites_per_px
    built = (psf.get("f_number"), psf.get("pitch_m"))
    if psf.image is not None and built == (f_number, pitch_m):
        return
    if psf.image is not None:
        bpy.data.images.remove(psf.image)
    kernel = optics.airy_kernel(colour.SRGB_DOMINANT_M, f_number, pitch_m)
    psf.image = curve_image("lens.psf", kernel)
    psf["f_number"], psf["pitch_m"] = f_number, pitch_m


def respond(mount: Mount, noise: np.random.Generator) -> None:
    """Set Cycles' pixel filter and the compositor's camera to a mount's; a build
    without a compositor takes only the filter.

    Args:
        mount: The camera whose lens, sensor and processing the compositor takes.
        noise: The stream the sensor's noise is seeded from.
    """
    # A microbolometer integrates over its pixel: Cycles' box filter, a pixel wide.
    cycles = bpy.context.scene.cycles
    props = cycles.bl_rna.properties
    if mount.camera.microbolometer.pitch_um is not None:
        cycles.pixel_filter_type, cycles.filter_width = "BOX", 1.0
    else:
        cycles.pixel_filter_type = props["pixel_filter_type"].default
        cycles.filter_width = props["filter_width"].default
    tree = bpy.context.scene.compositing_node_group
    if tree is None:
        return
    camera = mount.camera
    for name in (ROLL, INDEX_ROLL):
        if name in tree.nodes:
            toggle(tree, name, camera.lens.fisheye is not None)
    if "lens.vignette" not in tree.nodes:
        return
    longer_of_width = max(camera.width_px, camera.height_px) / camera.width_px
    tangent = longer_of_width * math.tan(math.radians(camera.hfov_deg) / 2)
    set_value(tree, "lens.vignette", tangent if camera.lens.vignetting else 0.0)
    set_value(tree, "processing.gain", 2**camera.processing.exposure_ev)
    share = camera.lens.veiling_glare
    # Glare adds its glow to the image; normalized, the glow is this share of the sum.
    strength = tree.nodes["lens.veiling_glare"].inputs["Strength"]
    strength.default_value = share / (1 - share)
    _set_diffraction(tree, camera)
    tone = camera.processing.tone
    toggle(tree, "processing.toned", tone != "none")
    if tone != "none":
        _set_curve(tree.nodes["processing.tone"], colour.tone_curve(tone))
    if "sensor.on" not in tree.nodes:
        if camera.sensor.full_well_e is not None:
            raise ValueError("this build has no sensor: no camera of its band has one")
        return
    toggle(tree, "sensor.on", camera.sensor.full_well_e is not None)
    sensor.respond(tree, camera, noise)
