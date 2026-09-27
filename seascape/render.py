"""Render the cameras a scenario asks for: one image each, per band and frame.

EO reaches 8 bits through the exposure and Blender's film curve. LWIR cannot: its
pixels are radiance in W m^-2 sr^-1, which Blender would clip to white, so an ir frame
is rendered float and written here as a thermal camera writes one: a png as 16-bit
centikelvin, a jpg as 8-bit grey through `agc.Agc`.
"""

import subprocess
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import bpy
import cv2
import numpy as np

from seascape import agc, labels, lwir, scene
from seascape.calibration import Calibration, CameraCalibration
from seascape.config import ImageFormat, Scenario


def _pixels(path: Path) -> np.ndarray:
    """An image's RGBA, (height, width, 4), bottom row first as Blender stores it."""
    image = bpy.data.images.load(str(path))
    try:
        width, height = image.size
        buffer = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(buffer)
    finally:
        bpy.data.images.remove(image)
    return buffer.reshape(height, width, 4)


def _temperatures_k(exr: Path) -> np.ndarray:
    return lwir.brightness_temperature(_pixels(exr)[..., 0])


def _write_thermal(exr: Path, fmt: ImageFormat, tone: agc.Agc) -> None:
    """Rewrite a float LWIR render as `fmt` beside it, and delete the exr."""
    t_k = _temperatures_k(exr)[::-1]  # cv2 writes the top row first
    out = exr.with_suffix(f".{fmt}")
    image = agc.counts(t_k) if fmt == "png" else tone(t_k)
    if not cv2.imwrite(str(out), image, [cv2.IMWRITE_JPEG_QUALITY, scene.JPEG_QUALITY]):
        raise OSError(f"cannot write {out}")
    exr.unlink()


def _index_output(folder: Path) -> bpy.types.CompositorNodeOutputFile:
    """Write each render's object-index pass to `folder`, as float EXR named
    `<file_name>index.exr`.

    Cycles takes it on the ray through the pixel centre, where calibration.json puts
    the pixel, whatever the pixel filter.
    """
    bpy.context.view_layer.use_pass_object_index = True
    tree = bpy.data.node_groups.new("labels", "CompositorNodeTree")
    bpy.context.scene.compositing_node_group = tree
    layers = tree.nodes.new("CompositorNodeRLayers")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.directory = str(folder)
    output.format.media_type = "IMAGE"
    output.format.file_format = "OPEN_EXR"
    output.format.color_depth = "32"
    output.file_output_items.new("FLOAT", "index")
    tree.links.new(layers.outputs["Object Index"], output.inputs["index"])
    return output


def _targets(built: scene.Built) -> list[labels.Target]:
    return [
        labels.Target(
            pass_index=anchor.pass_index,
            name=anchor.name,
            category=asset,
            centre_m=tuple(anchor.matrix_world.translation.xy),
            waterline_m=scene.waterline_m(anchor),
        )
        for asset, anchors in built.targets.items()
        for anchor in anchors
    ]


def _write_truth(
    into: Path, cameras: list[CameraCalibration], truth: labels.Labels
) -> list[Path]:
    return [Calibration(cameras=cameras).write(into), truth.write(into)]


def _commit() -> str | None:
    """The source checkout's commit, or None for an installed package."""
    root = Path(__file__).parents[1]
    # A wheel installed into another project's .venv sits inside that project's work
    # tree, and git would answer with its commit.
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.run(
            # --exclude: a tag would otherwise replace the hash with its own name.
            ["git", "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _info(scenario: Scenario) -> dict[str, Any]:
    """COCO's info block, and what it takes to reproduce the render."""
    devices = bpy.context.preferences.addons["cycles"].preferences.devices
    gpu = bpy.context.scene.cycles.device == "GPU"
    return {
        "version": version("seascape"),
        "date_created": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "blender": bpy.app.version_string,
        "blender_build": bpy.app.build_hash.decode(),
        "devices": [d.name for d in devices if d.use] if gpu else ["CPU"],
        "scenario": scenario.model_dump(mode="json"),
    }


def render(scenario: Scenario, into: Path) -> list[Path]:
    """Write one image per camera and frame into `into`, their calibration and their
    labels. A sequence puts each camera's frames in a folder of its own."""
    # Blender resolves a relative render.filepath against the .blend, not the shell.
    into = into.resolve()
    into.mkdir(parents=True, exist_ok=True)
    outputs = scenario.outputs
    radius_m = scene.earth_radius_m(scenario.sea.refraction_k)
    sequence = len(outputs.times_s) > 1
    written: list[Path] = []
    cameras: list[CameraCalibration] = []
    truth = labels.Labels()
    with tempfile.TemporaryDirectory() as tmp:
        passes = Path(tmp)
        for band in outputs.bands:
            # The default bands can name one the rig has no camera for.
            mounts = [m for m in scenario.rig.mounts if m.camera.kind == band]
            if not mounts:
                continue
            thermal = band == "ir" and outputs.format != "exr"
            built = scene.build(scenario, band)
            # After a build: the first one picks the device.
            truth.info = truth.info or _info(scenario)
            index_output = _index_output(passes)
            sc = bpy.context.scene
            tones = {m.name: agc.Agc(1 / outputs.fps) for m in mounts}
            for frame, time_s in enumerate(outputs.times_s):
                sc.frame_set(frame)
                # Both read matrix_world, which moves with the frame.
                targets = _targets(built)
                for mount in mounts:
                    sc.camera = built.cameras[mount.name]
                    sc.render.resolution_x, sc.render.resolution_y = (
                        mount.camera.width_px,
                        mount.camera.height_px,
                    )
                    name = f"{mount.name}/{frame:04d}" if sequence else mount.name
                    sc.render.filepath = str(into / name)
                    index_output.file_name = f"{mount.name}."
                    bpy.ops.render.render(write_still=True)
                    if thermal:
                        _write_thermal(
                            into / f"{name}.exr", outputs.format, tones[mount.name]
                        )
                    file_name = f"{name}.{outputs.format}"
                    written.append(into / file_name)
                    camera = scene.calibrate(built, mount, file_name)
                    cameras.append(camera)
                    index = _pixels(passes / f"{mount.name}.index.exr")[::-1, :, 0]
                    truth.add(
                        camera, time_s, np.rint(index).astype(int), targets, radius_m
                    )
                # Every frame, so a render that dies keeps what it wrote.
                _write_truth(into, cameras, truth)
    if not written:
        raise ValueError(
            f"the rig has no camera in any of {outputs.bands}: nothing to render"
        )
    return written + _write_truth(into, cameras, truth)
