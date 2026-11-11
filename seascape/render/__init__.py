"""Render the cameras a scenario asks for: one image each, per band and frame.

EO reaches 8 bits through a camera the build puts in the compositor. LWIR cannot: its
pixels are radiance in W m^-2 sr^-1, which Blender would clip to white, so an ir frame
is rendered float and written here as a thermal camera writes one: a png as 16-bit
centikelvin, a jpg as 8-bit grey through `agc.Agc`.
"""

import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Collection
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import bpy
import cv2
import numpy as np

from seascape.assets import manifest
from seascape.blender import blend, scene, sea
from seascape.blender.blend import node
from seascape.blender.compositor import INDEX_ROLL, JPEG_QUALITY, unroll
from seascape.blender.vessels import heading_rad, waterline_m
from seascape.config import (
    Camera,
    Crop,
    ImageFormat,
    Mount,
    Scenario,
    Stitch,
    recorded_name,
    sources,
    substream,
)
from seascape.outputs import agc, labels, microbolometer, stitch
from seascape.outputs.calibration import Calibration, CameraCalibration
from seascape.physics import earth


def _frame(path: Path) -> np.ndarray:
    """An image as written, top row first: RGB, or its one channel."""
    if path.suffix == ".exr":
        return blend.read(path)[::-1, :, :3]
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"cannot read {path}")
    return image if image.ndim == 2 else image[..., 2::-1]  # cv2 reads BGR(A)


def _write_thermal(
    exr: Path,
    fmt: ImageFormat,
    tone: agc.Agc,
    camera: Camera,
    noise: np.random.Generator,
) -> None:
    """Rewrite a float LWIR render beside it as a camera reads it, and delete the exr.

    Args:
        exr: The float render, radiance in its first channel.
        fmt: The format written; a png holds counts, any other the tone-mapped frame.
        tone: The camera's automatic gain control, kelvin to 8-bit grey.
        camera: Whose lens and focal plane the frame goes through.
        noise: Draws the focal plane's temporal noise, its NETD.
    """
    # cv2 writes the top row first
    radiance = blend.read(exr)[::-1, :, 0].astype(np.float64)
    t_k = microbolometer.seen(radiance, camera, noise)
    out = exr.with_suffix(f".{fmt}")
    if fmt == "png":
        image, params = agc.counts(t_k), []
    else:
        image, params = tone(t_k), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    if not cv2.imwrite(str(out), image, params):
        raise OSError(f"cannot write {out}")
    exr.unlink()


def _index_output(folder: Path) -> bpy.types.CompositorNodeOutputFile:
    """Write each render's object-index pass as float EXR named `<file_name>index.exr`.

    Cycles takes it on the ray through the pixel centre, where calibration.json puts
    the pixel, whatever the pixel filter.

    Args:
        folder: Where the passes are written.
    """
    bpy.context.view_layer.use_pass_object_index = True
    # The build's tree: a new one would drop what it does to the picture.
    tree = bpy.context.scene.compositing_node_group
    layers = next(n for n in tree.nodes if n.bl_idname == "CompositorNodeRLayers")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = "index.output"
    output.directory = str(folder)
    output.format.media_type = "IMAGE"
    output.format.file_format = "OPEN_EXR"
    output.format.color_depth = "32"
    output.file_output_items.new("FLOAT", "index")
    tree.links.new(
        unroll(tree, layers.outputs["Object Index"], INDEX_ROLL), output.inputs["index"]
    )
    return output


def _video_output(folder: Path) -> None:
    """Write each render's frame, as the scene writes it but cut by a Crop node, named
    `<file_name>video.<format>`; `_cut` sets the crop.

    Args:
        folder: Where the frames are written.
    """
    tree = bpy.context.scene.compositing_node_group
    frame = next(n for n in tree.nodes if n.bl_idname == "NodeGroupOutput")
    picture = frame.inputs["Image"].links[0].from_socket
    crop = node(tree, "CompositorNodeCrop", {"Image": picture})
    # The render's own output pads a cropped frame back to the render's size.
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.name = "video.output"
    output.directory = str(folder)
    output.format.media_type = "IMAGE"
    settings = bpy.context.scene.render.image_settings
    output.format.file_format = settings.file_format
    output.format.color_depth = settings.color_depth
    output.format.quality = settings.quality
    output.file_output_items.new("RGBA", "video")
    tree.links.new(crop.outputs["Image"], output.inputs["video"])


def _cut(name: str, width_px: int, height_px: int, crop: Crop | None) -> None:
    """Name the next render's cropped frame, and cut it to `crop`; None writes none.

    Args:
        name: The camera's name, which the file takes.
        width_px: Before the crop.
        height_px: Before the crop.
        crop: What the camera's video keeps of its frame.
    """
    output = bpy.context.scene.compositing_node_group.nodes["video.output"]
    output.file_name = f"{name}."
    output.mute = crop is None
    if crop is None:
        return
    sockets = output.inputs["video"].links[0].from_node.inputs
    sockets["X"].default_value = crop.left_px
    # Blender's rows count up from the bottom.
    sockets["Y"].default_value = crop.bottom_px
    sockets["Width"].default_value = width_px - crop.left_px - crop.right_px
    sockets["Height"].default_value = height_px - crop.top_px - crop.bottom_px


def _targets(built: scene.Built) -> list[labels.Target]:
    return [
        labels.Target(
            pass_index=anchor.pass_index,
            name=anchor.name,
            category_id=manifest()[asset].category_id,
            category=manifest()[asset].category,
            supercategory=manifest()[asset].supercategory,
            centre_m=tuple(anchor.matrix_world.translation.xy),
            waterline_m=waterline_m(anchor),
            heading_deg=math.degrees(heading_rad(anchor)) % 360.0,
            dims_m=(anchor["length_m"], anchor["beam_m"], anchor["height_m"]),
            engine_off_min=anchor["engine_off_min"],
        )
        for asset, anchors in built.targets.items()
        for anchor in anchors
    ]


def _write_truth(
    into: Path,
    cameras: list[CameraCalibration],
    raw: list[CameraCalibration],
    truth: labels.Labels,
) -> list[Path]:
    return [Calibration(cameras=cameras, raw=raw).write(into), truth.write(into)]


def _commit() -> str | None:
    """This module's commit, else the image's, else None."""
    here = Path(__file__)
    try:
        # A wheel in another project's .venv sits in a work tree that does not track it.
        _git(here.parent, "ls-files", "--error-unmatch", here.name)
        # --exclude: a tag would otherwise replace the hash with its own name.
        return _git(
            here.parent, "describe", "--always", "--dirty", "--abbrev=40", "--exclude=*"
        )
    except (OSError, subprocess.CalledProcessError):
        # The Dockerfile's build arg: an image carries no .git.
        return os.environ.get("SEASCAPE_COMMIT") or None


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def _info(scenario: Scenario) -> dict[str, Any]:
    """COCO's info block, and what it takes to reproduce the render."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    gpu = bpy.context.scene.cycles.device == "GPU"
    return {
        "version": version("seascape"),
        "date_created": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "blender": bpy.app.version_string,
        "blender_build": bpy.app.build_hash.decode(),
        # One card is listed under CUDA and OPTIX alike, by the same name.
        "backend": preferences.compute_device_type if gpu else "CPU",
        "devices": [d.name for d in preferences.devices if d.use] if gpu else ["CPU"],
        "scenario": scenario.model_dump(mode="json"),
    }


def _render_still(path: Path, what: str) -> None:
    """Render the active camera to `path`, which `render.filepath` names."""
    result = bpy.ops.render.render(write_still=True)
    # A GPU that dies partway can leave the frame unwritten, silently.
    if "FINISHED" not in result or not path.is_file() or not path.stat().st_size:
        raise RuntimeError(
            f"{what} was not written to {path} (Blender returned {sorted(result)})"
        )


type Shot = tuple[Mount, CameraCalibration, np.ndarray, np.ndarray]


def _stem(camera: str, frame: int | None) -> str:
    return camera if frame is None else f"{camera}/{frame:04d}"


def _stitch(
    shots: list[Shot], into: Path, frame: int | None, fmt: ImageFormat
) -> tuple[list[Shot], list[Path]]:
    """`shots` with each stitch's two frames replaced by the one stitched from them.

    Args:
        shots: One frame's shots, its cameras as written.
        into: The render's folder.
        frame: The frame's number in a sequence; None for a still.
        fmt: The format written; an exr records no videos.
    """
    if fmt == "exr":
        return shots, []
    by_camera = {(s[0].rig_name, s[0].camera_name): s for s in shots}
    rigs = {m.rig_name: m.rig for m, *_ in shots}
    written = []
    for rig_name, rig in rigs.items():
        for name, video in rig.videos.items():
            # Its cameras share a band, so another band's frame has neither.
            if not isinstance(video, Stitch) or (
                (rig_name, video.cameras[0]) not in by_camera
            ):
                continue
            pair = [by_camera.pop((rig_name, c)) for c in video.cameras]
            (mount, left, *_), (_, right, *_) = pair
            out = recorded_name(rig_name, name)
            camera = stitch.fused(left, right, out, f"{_stem(out, frame)}.{fmt}")
            image, index = stitch.stitch(
                (pair[0][3], pair[1][3]),
                (pair[0][2], pair[1][2]),
                (left, right),
                camera,
            )
            path = into / camera.image
            path.parent.mkdir(exist_ok=True)
            params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if fmt == "jpg" else []
            # cv2 writes BGR.
            bgr = image if image.ndim == 2 else image[..., ::-1]
            if not cv2.imwrite(str(path), bgr, params):
                raise OSError(f"cannot write {path}")
            by_camera[(rig_name, name)] = (mount, camera, index, _frame(path))
            written.append(path)
    return list(by_camera.values()), written


def _shoot(
    built: scene.Built,
    mount: Mount,
    rng: np.random.Generator,
    frame: int | None,
    fmt: ImageFormat,
    thermal: tuple[agc.Agc, np.random.Generator] | None,
    into: Path,
    passes: Path,
) -> tuple[Shot, CameraCalibration | None, list[Path]]:
    """Render one camera's frame, and cut from it the frame its rig records.

    Args:
        built: The scene, at the frame's time.
        mount: The camera to render through.
        rng: Draws its sensor's noise.
        frame: The frame's number in a sequence; None for a still.
        fmt: The format written.
        thermal: The AGC and NETD noise an ir frame is written through; None keeps
            the render.
        into: The render's folder.
        passes: Where the compositor writes the index pass and the cropped frame.

    Returns:
        The labelled frame as its mount, calibration, object index and image; the
        camera as rendered, if a video records it; and the files written.
    """
    scene.look_through(built.cameras, mount, rng)
    # An exr is the radiance the cameras saw, before any recorder.
    videos = mount.rig.videos if fmt != "exr" else {}
    video = next(
        (
            (recorded_name(mount.rig_name, name), recorded)
            for name, recorded in videos.items()
            if mount.camera_name in sources(recorded)
        ),
        None,
    )
    crop = video[1] if video and isinstance(video[1], Crop) else None
    width, height = mount.camera.width_px, mount.camera.height_px
    tree = bpy.context.scene.compositing_node_group
    tree.nodes["index.output"].file_name = f"{mount.name}."
    _cut(mount.name, width, height, crop)
    bpy.context.scene.render.filepath = str(into / _stem(mount.name, frame))
    ext = "exr" if thermal else fmt
    rendered = into / f"{_stem(mount.name, frame)}.{ext}"
    _render_still(rendered, f"camera {mount.name}'s frame")
    camera = scene.calibrate(built, mount, f"{_stem(mount.name, frame)}.{fmt}")
    # An undistortion renders its pinhole; the camera recorded is the rig's.
    own = mount._replace(camera=mount.rig.cameras[mount.camera_name])
    source = scene.calibrate(built, own, camera.image)
    index = blend.read(passes / f"{mount.name}.index.exr")[::-1, :, 0]
    written = []
    # A stitch's cameras keep their names; `_stitch` names what it makes of them.
    if video is not None and not isinstance(video[1], Stitch):
        out = video[0]
        camera = camera.model_copy(update={"name": out})
        if crop is not None:
            # The camera's own frame stays, unlabelled.
            written.append(rendered)
            rendered = into / f"{_stem(out, frame)}.{ext}"
            rendered.parent.mkdir(exist_ok=True)
            # The passes' temporary folder can sit on another filesystem.
            shutil.move(passes / f"{mount.name}.video.{ext}", rendered)
            image = f"{_stem(out, frame)}.{fmt}"
            camera = camera.model_copy(update={"image": image}).cropped(crop)
            rows = slice(crop.top_px, height - crop.bottom_px)
            index = index[rows, crop.left_px : width - crop.right_px]
    if thermal:
        tone, noise = thermal
        _write_thermal(rendered, fmt, tone, mount.camera, noise)
    written.append(into / camera.image)
    taken = (mount, camera, np.rint(index).astype(int), _frame(into / camera.image))
    return taken, source if video else None, written


def render(
    scenario: Scenario,
    into: Path,
    advance: Callable[[], None] = lambda: None,
    frames: Collection[int] | None = None,
) -> list[Path]:
    """Write one image per camera and frame, their calibration and their labels. A
    sequence puts each camera's frames in a folder of its own.

    Args:
        scenario: What to render; each band is built culled to its own cameras at
            their resolution.
        into: Where the images and their truth go; made if missing.
        advance: Runs after each image.
        frames: The frames rendered, by number; None renders them all.
    """
    # Blender resolves a relative render.filepath against the .blend, not the shell.
    into = into.resolve()
    into.mkdir(parents=True, exist_ok=True)
    outputs = scenario.outputs
    radius_m = earth.earth_radius_m(scenario.sea.refraction_k)
    sequence = len(outputs.times_s) > 1
    written: list[Path] = []
    cameras: list[CameraCalibration] = []
    raw: list[CameraCalibration] = []
    truth = labels.Labels()
    with tempfile.TemporaryDirectory() as tmp:
        passes = Path(tmp)
        for band in outputs.bands:
            # The default bands can name one no rig has a camera for.
            mounts = scenario.band_mounts(band)
            if not mounts:
                continue
            thermal = band == "ir" and outputs.format != "exr"
            # Every frame here is one of the scenario's cameras, at its resolution.
            built = scene.build(scenario, band, cull=True)
            # Here, not in render_pre, where Blender prints an error and renders on.
            sea.remember_foam()
            # After a build: the first one picks the device.
            truth.info = truth.info or _info(scenario)
            _index_output(passes)
            _video_output(passes)
            sc = bpy.context.scene
            tones = {m.name: agc.Agc(m.camera.agc, 1 / outputs.fps) for m in mounts}
            noises = {
                m.name: substream(scenario.seed, f"microbolometer/{m.name}")
                for m in mounts
            }
            for frame, time_s in enumerate(outputs.times_s):
                if frames is not None and frame not in frames:
                    continue
                sc.frame_set(frame)
                # Both read matrix_world, which moves with the frame.
                targets = _targets(built)
                shot = []
                for mount in mounts:
                    taken, source, files = _shoot(
                        built,
                        mount,
                        substream(scenario.seed, f"sensor/{mount.name}"),
                        frame if sequence else None,
                        outputs.format,
                        (tones[mount.name], noises[mount.name]) if thermal else None,
                        into,
                        passes,
                    )
                    shot.append(taken)
                    raw += [source] if source else []
                    written += files
                    advance()
                shot, files = _stitch(
                    shot, into, frame if sequence else None, outputs.format
                )
                written += files
                cameras += [camera for _, camera, _, _ in shot]
                faints = [labels.faint(i, f, targets) for _, _, i, f in shot]
                if outputs.keep_if_any_sees:
                    presents = [labels.present(i, targets) for _, _, i, _ in shot]
                    faints = labels.agreed(faints, presents)
                for (mount, camera, index, image), low in zip(
                    shot, faints, strict=True
                ):
                    truth.add(
                        camera,
                        time_s,
                        index,
                        image,
                        targets,
                        radius_m,
                        mount.camera,
                        faint=low,
                    )
                # Every frame, so a render that dies keeps what it wrote.
                _write_truth(into, cameras, raw, truth)
    if not written:
        raise ValueError(
            f"no rig has a camera in any of {outputs.bands}: nothing to render"
        )
    return written + _write_truth(into, cameras, raw, truth)
