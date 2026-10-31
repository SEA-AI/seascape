"""The compositor each camera writes through, its output formats, and the GPU."""

import math

import bpy
import numpy as np

from seascape.blender import sensor
from seascape.blender.blend import (
    scalar,
    vector,
)
from seascape.config import (
    Band,
    ImageFormat,
    Mount,
    Outputs,
)

JPEG_QUALITY = 95


# ISO 12232's saturation-based speed puts mid-grey at 10/78 of saturation; the
# log-average luminance stands for a scene's mid-grey (Reinhard et al. 2002).
MID_GREY = 10 / 78


# A lens scatters in Harvey's form, (1 + (r / r0)^2)^(-g / 2) off the direct light
# (Harvey 1976, the ABg model). Judgements, for a generic lens: the share of the light
# it scatters, r0 in half the image's longer side, and g.
GLARE_SHARE, GLARE_SHOULDER, GLARE_SLOPE = 0.02, 0.01, 3.0


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
    `largest_px` is the longest side of any of the band's cameras, and `graded` whether
    any has a sensor."""
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


def _compositor(compensation_ev: float | None, graded: bool = False) -> None:
    """The render as a camera takes it, `compensation_ev` over auto-exposure, through
    a sensor if `graded`; as rendered for None."""
    tree = bpy.data.node_groups.new("compositor", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    bpy.context.scene.compositing_node_group = tree
    image = tree.nodes.new("CompositorNodeRLayers").outputs["Image"]
    if compensation_ev is not None:
        image = _camera(tree, image, compensation_ev, graded)
    tree.links.new(image, tree.nodes.new("NodeGroupOutput").inputs["Image"])


def _camera(
    tree: bpy.types.NodeTree,
    image: bpy.types.NodeSocket,
    compensation_ev: float,
    with_sensor: bool = False,
) -> bpy.types.NodeSocket:
    """A lens's falloff, glare and blur, then auto-exposure, and with `with_sensor`
    the sensor and ISP. Without one the Standard view clips at white, as a sensor
    does. `respond` sets the camera.

    Limitation: each frame meters itself; a real camera's exposure lags, so add a time
    constant if a sequence flickers.
    """
    nodes, links = tree.nodes, tree.links
    coords, info = (
        nodes.new(kind)
        for kind in ("CompositorNodeImageCoordinates", "CompositorNodeImageInfo")
    )
    links.new(image, coords.inputs["Image"])
    links.new(image, info.inputs["Image"])
    # Uniform spans -1 to 1 along the longer side, so the kernel is a fixed share of
    # the frame at any resolution. Fog Glow and Bloom change shape with it.
    radius = vector(tree, "LENGTH", coords.outputs["Uniform"])
    # Natural vignetting: irradiance falls as cos^4 of the field angle, 1 / (1 +
    # tan^2)^2 (Kerr 2007). `lens.vignette` is the tangent at Uniform's 1, or 0.
    tangent = scalar(tree, "MULTIPLY", radius, sensor._value(tree, "lens.vignette"))
    falloff = scalar(
        tree,
        "POWER",
        scalar(tree, "MULTIPLY_ADD", tangent, tangent, 1.0),
        -2.0,
    )
    image = vector(tree, "SCALE", image, falloff)
    shoulder = scalar(tree, "DIVIDE", radius, GLARE_SHOULDER)
    harvey = scalar(
        tree,
        "POWER",
        scalar(tree, "MULTIPLY_ADD", shoulder, shoulder, 1.0),
        -GLARE_SLOPE / 2,
    )
    glare = nodes.new("CompositorNodeGlare")
    glare.inputs["Type"].default_value = "Kernel"
    # Half the resolution: Harvey's glow is smooth, so only the sun's rim shows it.
    glare.inputs["Quality"].default_value = "Medium"
    # A lens scatters all the light, not only what clips.
    glare.inputs["Threshold"].default_value = 0.0
    # Glare adds its glow to the image; normalized, the glow is this share of the sum.
    glare.inputs["Strength"].default_value = GLARE_SHARE / (1 - GLARE_SHARE)
    kernel = next(s for s in glare.inputs if s.name == "Kernel" and s.type == "VALUE")
    links.new(harvey, kernel)
    links.new(image, glare.inputs["Image"])
    side = nodes.new("ShaderNodeSeparateXYZ")
    links.new(info.outputs["Resolution"], side.inputs["Vector"])
    longer_px = scalar(tree, "MAXIMUM", side.outputs["X"], side.outputs["Y"])
    # A share of the frame, not a pixel count, so it blurs alike at any resolution.
    size_px = scalar(tree, "MULTIPLY", longer_px, BLUR_OF_SIDE)
    size = nodes.new("ShaderNodeCombineXYZ")
    links.new(size_px, size.inputs["X"])
    links.new(size_px, size.inputs["Y"])
    blur = nodes.new("CompositorNodeBlur")
    blur.inputs["Type"].default_value = "Gaussian"
    links.new(size.outputs["Vector"], blur.inputs["Size"])
    links.new(glare.outputs["Image"], blur.inputs["Image"])
    image = blur.outputs["Image"]
    _, log_average = sensor._log_average(tree, image)
    gain = scalar(
        tree, "DIVIDE", 2**compensation_ev * MID_GREY, log_average, name="exposure"
    )
    processing = sensor._value(tree, "processing.gain", 1.0)
    gain = scalar(tree, "MULTIPLY", gain, processing)
    exposed = vector(tree, "SCALE", image, gain)
    if not with_sensor:
        return exposed
    # A camera without a full well shares the tree, and takes the ideal frame.
    grey = scalar(tree, "MULTIPLY", processing, 2**compensation_ev * MID_GREY)
    read = sensor.chain(tree, exposed, coords.outputs["Pixel"], grey)
    return sensor._switch(tree, "sensor.on", exposed, read)


def respond(mount: Mount, noise: np.random.Generator) -> None:
    """Set Cycles' pixel filter and the compositor's camera to `mount`'s, its sensor
    noise seeded from `noise`; a build without a compositor takes only the filter."""
    # A microbolometer integrates over its pixel: Cycles' box filter, a pixel wide.
    cycles = bpy.context.scene.cycles
    props = cycles.bl_rna.properties
    if mount.camera.microbolometer.pitch_um is not None:
        cycles.pixel_filter_type, cycles.filter_width = "BOX", 1.0
    else:
        cycles.pixel_filter_type = props["pixel_filter_type"].default
        cycles.filter_width = props["filter_width"].default
    tree = bpy.context.scene.compositing_node_group
    if tree is None or "lens.vignette" not in tree.nodes:
        return
    camera = mount.camera
    longer_of_width = max(camera.width_px, camera.height_px) / camera.width_px
    tangent = longer_of_width * math.tan(math.radians(camera.hfov_deg) / 2)
    sensor._set(tree, "lens.vignette", tangent if camera.lens.vignetting else 0.0)
    sensor._set(tree, "processing.gain", 2**camera.processing.exposure_ev)
    if "sensor.on" not in tree.nodes:
        return
    sensor._toggle(tree, "sensor.on", camera.sensor.full_well_e is not None)
    sensor.respond(tree, camera, noise)
