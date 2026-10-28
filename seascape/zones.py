"""Sheets for giving an asset's parts their thermal zones, and for checking them.

Workbench renders each view three ways: every face's part as a flat float (no
anti-aliasing, so a pixel names one part), studio-lit grey for shape, and the asset's
own textures. Every sheet is cut from those three, so the render count does not grow
with the number of parts.
"""

import colorsys
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

import bpy
import cv2
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw

from seascape.assets import manifest
from seascape.blender import blend, faces
from seascape.blender.faces import Parts, face_zones
from seascape.blender.meshes import _corners, _import, _meshes
from seascape.outputs.montage import INK, MATTE, font
from seascape.physics.thermal import ZoneName


class Aim(NamedTuple):
    name: str
    # Bearing from the asset's centre, bow 0, starboard 90.
    bearing_deg: float
    elevation_deg: float


VIEWS: tuple[Aim, ...] = (
    Aim("bow", 0.0, 15.0),
    Aim("port bow", 300.0, 20.0),
    Aim("port beam", 270.0, 10.0),
    Aim("port quarter", 225.0, 20.0),
    Aim("stern", 180.0, 15.0),
    Aim("starboard quarter", 135.0, 20.0),
    Aim("starboard beam", 90.0, 10.0),
    Aim("top", 270.0, 80.0),
)
VIEW_PX = (1600, 1200)
TILE = (400, 300)
COLUMNS = 4
CAPTION_H = 22
TITLE_H = 48
# Workbench draws in half floats. A software rasteriser's rounding read 1023 back as
# 1022; base-256 digits sit where a half float steps by 1/8, clear of it.
DIGIT = 256
# The pixels of the full view a part needs for a mark on an overview, and an island
# for a sheet of its own.
MIN_MARK_PX = 1500
# A part covering less of a tile than this is marked beside itself.
SMALL_TILE_PX = 2000
# The UV texels an island needs for a mark on the atlas.
MIN_ATLAS_PX = 150
ATLAS_PX = 1024

REST = (170, 170, 170)
HIGHLIGHT = (255, 110, 0)
ZONE_COLOURS: dict[ZoneName, tuple[int, int, int]] = {
    "paint_hull": (200, 40, 40),
    "paint_superstructure": (240, 240, 240),
    "deck": (60, 160, 60),
    "glass": (40, 200, 255),
    "bare_metal": (150, 150, 170),
    "rubber": (40, 40, 40),
    "fabric": (230, 180, 60),
    "funnel": (250, 120, 0),
    "exhaust": (255, 0, 200),
    "engine_cowl": (130, 60, 200),
    "wood_composite": (140, 90, 40),
}

Kind = Literal["slots", "islands"]


def _setup() -> None:
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x, sc.render.resolution_y = VIEW_PX
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.dither_intensity = 0.0
    view = sc.view_settings
    view.view_transform, view.look = "Standard", "None"
    view.exposure, view.gamma = 0.0, 1.0
    shading = sc.display.shading
    shading.show_cavity = shading.show_shadows = False
    shading.show_specular_highlight = False
    shading.single_color = (0.8, 0.8, 0.8)


def _textures() -> None:
    """Workbench's texture mode draws a material's active image node, or its viewport
    colour: point both at the base colour."""
    for material in bpy.data.materials:
        bsdf = faces.bsdf(material)
        if bsdf is None:
            continue
        source = faces.source(bsdf.inputs["Base Color"])
        if source is not None:
            material.node_tree.nodes.active = source[2]
        else:
            material.diffuse_color = (*bsdf.inputs["Base Color"].default_value[:3], 1)


def _paint(found: Parts, per_face: list[np.ndarray]) -> None:
    """Each face's value as a corner colour: its base-`DIGIT` digits in red, green and
    blue, each exact in Workbench's half floats."""
    for obj, value in zip(found.objects, per_face, strict=True):
        mesh = obj.data
        total = faces.ints(mesh.polygons, "loop_total", len(mesh.polygons))
        colours = np.zeros((len(mesh.loops), 4), dtype=np.float32)
        loop = np.repeat(value, total)
        for channel in range(3):
            colours[:, channel] = loop // DIGIT**channel % DIGIT
        colours[:, 3] = 1.0
        if "part" in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes["part"])
        attribute = mesh.color_attributes.new("part", "FLOAT_COLOR", "CORNER")
        attribute.data.foreach_set("color", colours.ravel())
        mesh.color_attributes.active_color = attribute
        mesh.color_attributes.render_color_index = mesh.color_attributes.find("part")


def _aim(found: Parts, bearing_deg: float, elevation_deg: float) -> None:
    corners = np.array([tuple(c) for c in _corners(found.objects)])
    centre = Vector(tuple((corners.min(axis=0) + corners.max(axis=0)) / 2))
    bearing, elevation = math.radians(bearing_deg), math.radians(elevation_deg)
    # A direction, not a rotation: no negation.
    toward = Vector(
        (
            math.sin(bearing) * math.cos(elevation),
            math.cos(bearing) * math.cos(elevation),
            math.sin(elevation),
        )
    )
    camera = bpy.context.scene.camera
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = (-toward).to_track_quat("-Z", "Y")
    rotation = camera.rotation_quaternion.to_matrix()
    right, up = rotation.col[0], rotation.col[1]
    offsets = [Vector(tuple(c)) - centre for c in corners]
    half_w = max(abs(o.dot(right)) for o in offsets)
    half_h = max(abs(o.dot(up)) for o in offsets)
    aspect = VIEW_PX[0] / VIEW_PX[1]
    radius = max(o.length for o in offsets)
    camera.data.ortho_scale = 2.1 * max(half_w, half_h * aspect)
    camera.data.clip_end = 4 * radius
    camera.location = centre + 2 * radius * toward


def _shoot(path: Path, color_type: str, light: str, aa: str, exr: bool) -> None:
    sc = bpy.context.scene
    sc.display.shading.color_type = color_type
    sc.display.shading.light = light
    sc.display.render_aa = aa
    settings = sc.render.image_settings
    settings.file_format = "OPEN_EXR" if exr else "PNG"
    settings.color_depth = "32" if exr else "8"
    settings.color_mode = "RGBA"
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


@dataclass
class View:
    name: str
    part: np.ndarray  # (h, w) the painted value, -1 where nothing is
    grey: np.ndarray  # (h, w) studio-lit shape, 0 to 1
    textured: np.ndarray  # (h, w, 3) uint8


def _views(found: Parts, per_face: list[np.ndarray]) -> list[View]:
    _setup()
    _textures()
    _paint(found, per_face)
    data = bpy.data.cameras.new("zones")
    data.type = "ORTHO"
    camera = bpy.data.objects.new("zones", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    views = []
    with tempfile.TemporaryDirectory() as tmp:
        for aim in VIEWS:
            _aim(found, aim.bearing_deg, aim.elevation_deg)
            ids, grey, tex = (
                Path(tmp) / f"{k}{s}"
                for k, s in (("id", ".exr"), ("grey", ".png"), ("tex", ".png"))
            )
            _shoot(ids, "VERTEX", "FLAT", "OFF", exr=True)
            _shoot(grey, "SINGLE", "STUDIO", "FXAA", exr=False)
            _shoot(tex, "TEXTURE", "STUDIO", "FXAA", exr=False)
            rgba = blend.read(ids)[::-1]
            digits = np.rint(rgba[..., :3]).astype(np.int64)
            value = digits @ (DIGIT ** np.arange(3))
            part = np.where(rgba[..., 3] > 0.5, value, -1)
            views.append(
                View(
                    aim.name,
                    part.astype(np.int64),
                    np.asarray(Image.open(grey).convert("L"), np.float32) / 255,
                    np.asarray(_on_backdrop(Image.open(tex))),
                )
            )
    return views


def parts(name: str) -> Parts:
    """Import `name` as an EO build fits it into the current file, with islands
    numbered by object name, then first face."""
    roots = _import(name, "eo")
    bpy.context.view_layer.update()
    return faces.found(_meshes(roots))


def _on_backdrop(image: Image.Image) -> Image.Image:
    """A dark hull shows on mid grey, not on the matte."""
    backdrop = Image.new("RGBA", image.size, (110, 110, 110, 255))
    return Image.alpha_composite(backdrop, image.convert("RGBA")).convert("RGB")


def palette(n: int) -> np.ndarray:
    """`n` colours apart from their neighbours: golden-ratio hues, three values."""
    hues = (np.arange(n) * 0.618034) % 1.0
    return np.array(
        [
            [
                round(255 * c)
                for c in colorsys.hsv_to_rgb(h, 0.75, (1.0, 0.8, 0.6)[k % 3])
            ]
            for k, h in enumerate(hues)
        ],
        dtype=np.uint8,
    ).reshape(-1, 3)


def _lit(part: np.ndarray, grey: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """`colours` (h, w, 3) shaded by a view's grey, on the matte where no part is."""
    shade = 0.35 + 0.65 * grey[..., None]
    image = np.clip(colours * shade, 0, 255).astype(np.uint8)
    image[part < 0] = MATTE
    return image


def _interior(mask: np.ndarray) -> tuple[int, int]:
    """The pixel of `mask` furthest from its edge, where a mark covers the least."""
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return int(x), int(y)


def _mark(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    edge: tuple[int, ...],
    taken: list[tuple[float, float, float, float]] | None = None,
) -> None:
    typeface = font(15)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=typeface)
    w, h = right - left + 8, bottom - top + 6
    box = (xy[0] - w / 2, xy[1] - h / 2, xy[0] + w / 2, xy[1] + h / 2)
    if taken is not None:
        if any(
            box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3]
            for t in taken
        ):
            return
        taken.append(box)
    draw.rectangle(box, fill=(0, 0, 0), outline=tuple(edge), width=2)
    draw.text((box[0] + 4 - left, box[1] + 3 - top), text, font=typeface, fill=INK)


def _grid(tiles: list[tuple[Image.Image, str]], title: list[str]) -> Image.Image:
    rows = max(1, math.ceil(len(tiles) / COLUMNS))
    sheet = Image.new(
        "RGB",
        (TILE[0] * COLUMNS, TITLE_H + rows * (TILE[1] + CAPTION_H)),
        MATTE,
    )
    draw = ImageDraw.Draw(sheet)
    for k, line in enumerate(title[:2]):
        draw.text((8, 4 + 22 * k), line, font=font(18 if k == 0 else 14), fill=INK)
    for k, (tile, caption) in enumerate(tiles):
        x = (k % COLUMNS) * TILE[0]
        y = TITLE_H + (k // COLUMNS) * (TILE[1] + CAPTION_H)
        sheet.paste(tile, (x, y))
        draw.text((x + 6, y + TILE[1] + 3), caption, font=font(14), fill=INK)
    return sheet


def _crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    """A 4:3 box around `mask`, at least a quarter of the view wide."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    width = max(
        1.6 * (xs.max() - xs.min() + 1), 1.6 * (ys.max() - ys.min() + 1) * w / h
    )
    width = min(max(width, w / 4), w)
    height = width * h / w
    left = min(max(cx - width / 2, 0), w - width)
    top = min(max(cy - height / 2, 0), h - height)
    return round(left), round(top), round(left + width), round(top + height)


def part_sheet(
    views: list[View], value: int, title: list[str], label: str
) -> Image.Image | None:
    """One part bright, the rest grey, cropped to it in each view that shows it."""
    tiles = []
    for view in views:
        mask = view.part == value
        if mask.sum() == 0:
            continue
        box = _crop(mask)
        rows, cols = slice(box[1], box[3]), slice(box[0], box[2])
        mask = mask[rows, cols]
        colours = np.where(mask[..., None], HIGHLIGHT, REST)
        lit = _lit(view.part[rows, cols], view.grey[rows, cols], colours)
        scale = TILE[0] / (box[2] - box[0])
        tile = Image.fromarray(lit).resize(TILE, Image.Resampling.LANCZOS)
        x, y = _interior(mask)
        x, y = x * scale, y * scale
        draw = ImageDraw.Draw(tile)
        if mask.sum() * scale**2 < SMALL_TILE_PX:
            # A mark on a small part would hide it: beside it, on a leader.
            beside = (x, y - 40) if y > 60 else (x, y + 40)
            draw.line([(x, y), beside], fill=HIGHLIGHT, width=2)
            x, y = beside
        _mark(draw, (x, y), label, HIGHLIGHT)
        tiles.append((tile, f"{view.name}: {int(mask.sum())} px"))
    return _grid(tiles, title) if tiles else None


def overview(
    views: list[View], colours: np.ndarray, labels: list[str], title: list[str]
) -> Image.Image:
    """Each view as textured, then with every part in its colour and mark."""
    tiles = []
    for view in views:
        textured = Image.fromarray(view.textured).resize(TILE, Image.Resampling.LANCZOS)
        tiles.append((textured, view.name))
    for view in views:
        painted = colours[np.clip(view.part, 0, None)]
        tile = Image.fromarray(_lit(view.part, view.grey, painted)).resize(
            TILE, Image.Resampling.LANCZOS
        )
        draw = ImageDraw.Draw(tile)
        scale = TILE[0] / VIEW_PX[0]
        counts = np.bincount(view.part[view.part >= 0], minlength=len(labels))
        taken: list[tuple[float, float, float, float]] = []
        for value in np.argsort(-counts, kind="stable"):
            if counts[value] < MIN_MARK_PX:
                break
            if not labels[value]:
                continue
            x, y = _interior(view.part == value)
            _mark(
                draw,
                (x * scale, y * scale),
                labels[value],
                tuple(colours[value]),
                taken,
            )
        tiles.append((tile, f"{view.name}, marked"))
    return _grid(tiles, title)


def legend(colours: np.ndarray, labels: list[str], names: list[str]) -> Image.Image:
    rows = math.ceil(len(names) / 3)
    line = 20
    image = Image.new("RGB", (TILE[0] * COLUMNS, 8 + rows * line), MATTE)
    draw = ImageDraw.Draw(image)
    width = TILE[0] * COLUMNS // 3
    for k, (label, name) in enumerate(zip(labels, names, strict=True)):
        x, y = (k // rows) * width + 8, 4 + (k % rows) * line
        draw.rectangle((x, y + 2, x + 14, y + 16), fill=tuple(colours[k]))
        draw.text((x + 20, y), f"{label}  {name}"[:52], font=font(14), fill=INK)
    return image


def _stack(*images: Image.Image) -> Image.Image:
    sheet = Image.new(
        "RGB", (max(i.width for i in images), sum(i.height for i in images)), MATTE
    )
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    return sheet


# By image name, for one `sheets` run: Base Color, Alpha and Metallic share atlases.
_texel_cache: dict[str, np.ndarray] = {}


def _texels(image: bpy.types.Image) -> np.ndarray:
    """An image's RGBA, top row first, no side over `ATLAS_PX`, read once a run."""
    if (cached := _texel_cache.get(image.name)) is not None:
        return cached
    width, height = image.size
    step = max(1, math.ceil(max(width, height) / ATLAS_PX))
    texels = blend.pixels(image)[::-step, ::step]
    _texel_cache[image.name] = texels
    return texels


def _uv_triangles(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    """Each triangle's UVs, (n, 3, 2), and its face."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    n = len(mesh.loop_triangles)
    loops = faces.ints(mesh.loop_triangles, "loops", 3 * n).reshape(-1, 3)
    face = faces.ints(mesh.loop_triangles, "polygon_index", n)
    if not mesh.uv_layers:
        return np.zeros((0, 3, 2)), np.zeros(0, dtype=np.int64)
    uv = faces.floats(mesh.uv_layers.active.data, "uv", 2 * len(mesh.loops)).reshape(
        -1, 2
    )
    return uv[loops], face


def _raster(
    shape: tuple[int, int], triangles: list[tuple[int, np.ndarray]]
) -> np.ndarray:
    """Which part covers each texel, -1 for none; a later part paints over."""
    h, w = shape
    code = np.zeros((h, w, 3), dtype=np.uint8)
    for value, uv in triangles:
        if len(uv) == 0:
            continue
        # A tiled material's UVs run past 1: each triangle moves to the tile its
        # centre is in.
        tile = uv - np.floor(uv.mean(axis=-2, keepdims=True))
        points = np.stack([tile[..., 0] * w, (1 - tile[..., 1]) * h], axis=-1)
        v = value + 1
        corners = np.rint(points).astype(np.int32)
        polygons = [corners[k] for k in range(len(corners))]
        colour = (float(v & 255), float((v >> 8) & 255), float((v >> 16) & 255))
        cv2.fillPoly(code, polygons, colour)
        # fillPoly can leave unfilled the edge two triangles share, and a sliver that
        # covers no texel centre.
        cv2.polylines(code, polygons, isClosed=True, color=colour)
    c = code.astype(np.int64)
    return c[..., 0] + (c[..., 1] << 8) + (c[..., 2] << 16) - 1


def _hex(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{round(255 * float(c)):02x}" for c in np.clip(rgb, 0, 1))


def _srgb(linear: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1's transfer function."""
    linear = np.clip(linear, 0, 1)
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )


def _channel_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(values.mean()), 3),
        "min": round(float(values.min()), 3),
        "over_half": round(float((values > 0.5).mean()), 3),
    }


def _input(
    material: bpy.types.Material | None,
    socket: str,
    triangles: list[tuple[int, np.ndarray]],
) -> dict[str, Any]:
    """A BSDF input as the shader reads it: its value, or the texels the material's
    faces cover."""
    bsdf = faces.bsdf(material)
    if bsdf is None:
        if socket == "Base Color" and material is not None:
            return {"colour": _hex(_srgb(np.array(material.diffuse_color[:3])))}
        return {}
    source = faces.source(bsdf.inputs[socket])
    if source is None:
        value = bsdf.inputs[socket].default_value
        if socket == "Base Color":
            return {"colour": _hex(_srgb(np.array(value[:3])))}
        return {"value": round(float(value), 3)}
    image, channel, _ = source
    texels = _texels(image)
    covered = _raster(texels.shape[:2], triangles) >= 0
    if not covered.any():
        return {"texture": image.name, "texels": 0}
    picked = texels[covered]
    out: dict[str, Any] = {"texture": image.name, "texels": int(covered.sum())}
    if socket == "Base Color":
        # Texels as stored: an sRGB image's are already display values.
        mean = picked[:, :3].mean(axis=0)
        is_linear = image.colorspace_settings.name not in {"sRGB", "sRGB EOTF"}
        out["colour"] = _hex(_srgb(mean) if is_linear else mean)
    else:
        out |= _channel_stats(picked[:, channel if channel is not None else 0])
    return out


def _heights(found: Parts, per_face: list[np.ndarray], n: int) -> np.ndarray:
    """Each part's lowest and highest point above the waterline, (n, 2)."""
    low = np.full(n, np.inf)
    high = np.full(n, -np.inf)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj, value in zip(found.objects, per_face, strict=True):
        mesh = obj.data
        # Posed: a skinned mesh's own vertices are its rest pose. A pose moves
        # vertices and keeps the faces, so the loops below still index them.
        evaluated = obj.evaluated_get(depsgraph)
        posed = evaluated.to_mesh()
        co = faces.floats(posed.vertices, "co", 3 * len(posed.vertices)).reshape(-1, 3)
        evaluated.to_mesh_clear()
        if len(co) != len(mesh.vertices):
            raise ValueError(f"{obj.name}: a modifier changes its vertices")
        m = np.array(obj.matrix_world)
        z = co @ m[2, :3] + m[2, 3]
        total = faces.ints(mesh.polygons, "loop_total", len(mesh.polygons))
        loop_value = np.repeat(value, total)
        loop_z = z[faces.ints(mesh.loops, "vertex_index", len(mesh.loops))]
        np.minimum.at(low, loop_value, loop_z)
        np.maximum.at(high, loop_value, loop_z)
    return np.stack([low, high], axis=1)


def _triangle_counts(found: Parts, per_face: list[np.ndarray], n: int) -> np.ndarray:
    counts = np.zeros(n, dtype=np.int64)
    for obj, value in zip(found.objects, per_face, strict=True):
        total = faces.ints(obj.data.polygons, "loop_total", len(obj.data.polygons))
        np.add.at(counts, value, total - 2)
    return counts


def _slot_records(found: Parts) -> list[dict[str, Any]]:
    n = len(found.materials)
    heights = _heights(found, found.face_material, n)
    triangles = _triangle_counts(found, found.face_material, n)
    uvs = [_uv_triangles(obj) for obj in found.objects]
    records = []
    for k, name in enumerate(found.materials):
        material = bpy.data.materials.get(name) if name else None
        mine = [
            (k, uv[found.face_material[i][face] == k])
            for i, (uv, face) in enumerate(uvs)
        ]
        records.append(
            {
                "slot": k + 1,
                "material": name,
                "objects": sorted(
                    obj.name
                    for obj, f in zip(found.objects, found.face_material, strict=True)
                    if (f == k).any()
                ),
                "triangles": int(triangles[k]),
                "base_colour": _input(material, "Base Color", mine),
                "alpha": _input(material, "Alpha", mine),
                "metallic": _input(material, "Metallic", mine),
                "height_m": [round(float(h), 2) for h in heights[k]],
            }
        )
    return records


def _island_records(
    found: Parts,
) -> tuple[list[dict[str, Any]], dict[str, Image.Image]]:
    """Each island's record, its colour the mean texel at its triangles' centres, and
    each base colour atlas with every island drawn on it."""
    n = len(found.islands)
    heights = _heights(found, found.face_island, n)
    triangles = _triangle_counts(found, found.face_island, n)
    flat = {
        name: _input(bpy.data.materials.get(name) if name else None, "Base Color", [])
        for name in found.materials
        if faces.texture(name) is None
    }
    colour = [flat.get(i.material, {}).get("colour", "") for i in found.islands]
    atlases: dict[str, Image.Image] = {}
    uvs = [_uv_triangles(obj) for obj in found.objects]
    all_uv = np.concatenate([u for u, _ in uvs])
    all_island = np.concatenate(
        [found.face_island[i][face] for i, (_, face) in enumerate(uvs)]
    )
    island_material = np.array([i.material for i in found.islands])
    for name in found.materials:
        source = faces.texture(name)
        if source is None or source.name in atlases:
            continue
        rgb = _texels(source)[..., :3]
        if source.colorspace_settings.name not in {"sRGB", "sRGB EOTF"}:
            rgb = _srgb(rgb)
        # Each island's triangles, in island order: a later one paints over.
        mine = (island_material == name)[all_island]
        uv, island = all_uv[mine], all_island[mine]
        if not len(island):  # a slot no face uses
            continue
        order = np.argsort(island, kind="stable")
        uv, island = uv[order], island[order]
        ids, starts = np.unique(island, return_index=True)
        cover = _raster(
            rgb.shape[:2],
            list(zip(ids.tolist(), np.split(uv, starts[1:]), strict=True)),
        )
        # Sampled at triangle centres, not from the raster: where islands share
        # texels, the raster gives them all to the last.
        h, w = rgb.shape[:2]
        centre = uv.mean(axis=1) % 1.0
        x = np.minimum((centre[:, 0] * w).astype(np.int64), w - 1)
        y = np.minimum(((1 - centre[:, 1]) * h).astype(np.int64), h - 1)
        sampled = rgb[y, x]
        counts = np.bincount(island, minlength=n)
        sums = np.stack(
            [np.bincount(island, sampled[:, c], minlength=n) for c in range(3)], axis=1
        )
        for j in ids:
            colour[j] = _hex(sums[j] / counts[j])
        texels = np.bincount(cover.ravel() + 1, minlength=n + 1)[1:]
        atlases[source.name] = _atlas(rgb, cover, texels)
    records = [
        {
            "island": i.id,
            "object": i.object,
            "material": i.material,
            "first_face": i.first_face,
            "faces": i.faces,
            "triangles": int(triangles[i.id]),
            "colour": colour[i.id],
            "height_m": [round(float(h), 2) for h in heights[i.id]],
        }
        for i in found.islands
    ]
    return records, atlases


def _atlas(rgb: np.ndarray, cover: np.ndarray, texels: np.ndarray) -> Image.Image:
    colours = palette(len(texels))
    image = (np.clip(rgb, 0, 1) * 255).astype(np.float32)
    inside = cover >= 0
    image[inside] = 0.5 * image[inside] + 0.5 * colours[cover[inside]]
    edges = (
        cv2.morphologyEx(
            (cover + 1).astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3))
        )
        > 0
    )
    image[edges] = 0
    atlas = Image.fromarray(image.astype(np.uint8))
    draw = ImageDraw.Draw(atlas)
    taken: list[tuple[float, float, float, float]] = []
    for j in np.argsort(-texels, kind="stable"):
        if texels[j] < MIN_ATLAS_PX:
            break
        _mark(draw, _interior(cover == j), str(j), tuple(colours[j]), taken)
    return atlas


def sheets(
    name: str, into: Path, kind: Kind | None = None, material: str | None = None
) -> list[Path]:
    """Write the sheets for labelling `name`'s parts, and their records, into `into`.

    Slots by default, islands for an asset of one material. Islands of `material`
    alone, if given, with the rest grey.
    """
    if kind == "slots" and material is not None:
        raise ValueError("a material picks islands, not slots")
    asset = manifest()[name]
    _texel_cache.clear()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    found = parts(name)
    if material is not None and material not in found.materials:
        raise ValueError(f"{name} has no material {material!r}")
    if kind is None:
        kind = "slots" if material is None and len(found.materials) > 1 else "islands"
    into.mkdir(parents=True, exist_ok=True)
    head = f"{name}: {asset.category}, {asset.size}, waterline at 0 m"
    if kind == "slots":
        per_face = found.face_material
        labels = [str(k + 1) for k in range(len(found.materials))]
        names = [m or "(no material)" for m in found.materials]
        records: list[dict[str, Any]] = _slot_records(found)
        atlases: dict[str, Image.Image] = {}
    else:
        n = len(found.islands)
        own = np.array(
            [material in {None, i.material} for i in found.islands] + [False]
        )
        # Another material's islands are one grey part, unmarked.
        per_face = [np.where(own[i], i, n) for i in found.face_island]
        labels = [str(i.id) if own[i.id] else "" for i in found.islands] + [""]
        names = [f"{i.object} / {i.material}" for i in found.islands]
        records, atlases = _island_records(found)
    views = _views(found, per_face)
    colours = palette(len(labels))
    if kind == "islands":
        colours[-1] = REST
    written = []
    sheet = overview(
        views,
        colours,
        labels,
        [
            head,
            f"{sum(map(bool, labels))} {kind}"
            + (f" of {material}" if material else ""),
        ],
    )
    if kind == "slots":
        sheet = _stack(sheet, legend(colours, labels, names))
    sheet.save(path := into / "overview.png")
    written.append(path)
    for texture, atlas in atlases.items():
        atlas.save(path := into / f"atlas-{_safe(texture)}.png")
        written.append(path)
    written += _part_sheets(views, records, labels, names, kind, into)
    body = {
        "asset": name,
        "description": asset.description,
        "draught_m": asset.draught_m,
        f"{kind[:-1]}_count": len(records),
        **({"island_digest": found.island_digest} if kind == "islands" else {}),
        kind: [r for k, r in enumerate(records) if labels[k]],
    }
    (path := into / f"{kind}.json").write_text(json.dumps(body, indent=1) + "\n")
    written.append(path)
    _texel_cache.clear()
    return written


def _part_sheets(
    views: list[View],
    records: list[dict[str, Any]],
    labels: list[str],
    names: list[str],
    kind: Kind,
    into: Path,
) -> list[Path]:
    """A sheet for each marked part large enough to see, noted in its record."""
    best = np.zeros(len(labels), dtype=np.int64)
    for view in views:
        counts = np.bincount(view.part[view.part >= 0], minlength=len(labels))
        best = np.maximum(best, counts)
    written = []
    for k, record in enumerate(records):
        record["visible_px"] = int(best[k])
        record["sheet"] = None
        if best[k] < (MIN_MARK_PX if kind == "islands" else 1) or not labels[k]:
            continue
        title = [f"{kind[:-1]} {labels[k]}: {names[k]}", _summary(record)]
        sheet = part_sheet(views, k, title, labels[k])
        if sheet is not None:
            sheet.save(path := into / f"{kind[:-1]}-{labels[k]}.png")
            record["sheet"] = path.name
            written.append(path)
    return written


def _safe(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def _summary(record: dict[str, Any]) -> str:
    low, high = record["height_m"]
    base = record.get("base_colour", {}).get("colour") or record.get("colour", "")
    alpha = record.get("alpha", {})
    a = alpha.get("value", alpha.get("mean", 1.0))
    span = f"{low:.1f} to {high:.1f} m"
    return f"{record['triangles']:,} tris, {span}, colour {base}, alpha {a}"


def contact(name: str, out: Path) -> Path:
    """The views coloured by the zones `assets.toml` gives `name`, for approval."""
    asset = manifest()[name]
    if asset.zones is None:
        raise ValueError(f"{name} has no zones in assets.toml")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    found = parts(name)
    faces.check(name, found, asset.zones)
    vocabulary = list(get_args(ZoneName.__value__))
    colours = np.array([ZONE_COLOURS[z] for z in vocabulary], dtype=np.uint8)
    per_face = face_zones(found, asset.zones)
    views = _views(found, per_face)
    triangles = _triangle_counts(found, per_face, len(vocabulary))
    present = np.flatnonzero(triangles)
    shares = [f"{100 * triangles[z] / triangles.sum():.0f}%" for z in present]
    head = f"{name}: thermal zones, {asset.category}, {asset.size}"
    numbering = f"{len(found.islands)} islands, digest {found.island_digest}"
    sheet = overview(
        views, colours, vocabulary, [head, f"share of triangles below; {numbering}"]
    )
    sheet = _stack(
        sheet, legend(colours[present], shares, [vocabulary[z] for z in present])
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out
