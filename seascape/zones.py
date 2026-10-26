"""Sheets for giving an asset's parts their thermal zones, and for checking them.

A part is a material slot, or an island: the faces of one material in one object that
share a vertex. Workbench renders every view three times: each face's part as a flat
float (no anti-aliasing, so a pixel names one part), studio-lit grey for shape, and the
asset's own textures. The sheets are composed from those, so a view renders once
whatever the number of parts.
"""

import colorsys
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import bpy
import cv2
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont

from seascape import render, scene
from seascape.assets import Zones, manifest
from seascape.thermal import ZoneName

# Name, bearing of the camera from the asset's centre (bow at 0), elevation.
VIEWS: tuple[tuple[str, float, float], ...] = (
    ("bow", 0.0, 15.0),
    ("port bow", 300.0, 20.0),
    ("port beam", 270.0, 10.0),
    ("port quarter", 225.0, 20.0),
    ("stern", 180.0, 15.0),
    ("starboard quarter", 135.0, 20.0),
    ("starboard beam", 90.0, 10.0),
    ("top", 270.0, 80.0),
)
VIEW_PX = (1600, 1200)
TILE = (400, 300)
COLUMNS = 4
CAPTION_H = 22
TITLE_H = 48
# Half floats hold every integer to 2048.
DIGIT = 1024
# The pixels of the full view a part needs for a mark on an overview, and an island
# for a sheet of its own.
MIN_MARK_PX = 1500
# A part covering less of a tile than this is marked beside itself.
SMALL_TILE_PX = 2000
# The UV texels an island needs for a mark on the atlas.
MIN_ATLAS_PX = 150
ATLAS_PX = 1024

MATTE = (24, 24, 24)
INK = (232, 232, 232)
REST = (170, 170, 170)
HIGHLIGHT = (255, 110, 0)
ZONE_COLOURS: dict[str, tuple[int, int, int]] = {
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


@dataclass(frozen=True)
class Island:
    id: int
    object: str
    material: str
    first_face: int
    faces: int


@dataclass
class Parts:
    """An imported asset's faces, each with its material and island."""

    objects: list[bpy.types.Object]  # by name
    materials: list[str]  # by name
    face_material: list[np.ndarray]  # per object, an index into `materials`
    face_island: list[np.ndarray]  # per object, an index into `islands`
    islands: list[Island]


def _ints(collection: Any, attribute: str, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    collection.foreach_get(attribute, out)
    return out


def _floats(collection: Any, attribute: str, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.float64)
    collection.foreach_get(attribute, out)
    return out


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Each node's connected component, as the smallest-numbered node reached in it."""
    label = np.arange(n)
    while True:
        least = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, least)
        np.minimum.at(new, b, least)
        new = new[new]
        if np.array_equal(new, label):
            return label
        label = new


def _face_components(
    mesh: bpy.types.Mesh, material: np.ndarray, textured: np.ndarray
) -> np.ndarray:
    """Per face, a label shared by the faces of its material that touch it: at a
    vertex, and where the material is `textured`, at a UV too, so that an atlas splits
    where its texture does.

    By position, not vertex index: glTF splits a vertex at every UV and normal seam, so
    Blender's own linked and loose parts stop at each one.
    """
    n_faces, n_loops = len(mesh.polygons), len(mesh.loops)
    co = _floats(mesh.vertices, "co", 3 * len(mesh.vertices)).reshape(-1, 3)
    low, high = co.min(axis=0), co.max(axis=0)
    span = max(float(np.linalg.norm(high - low)), 1e-12)
    start = _ints(mesh.polygons, "loop_start", n_faces)
    total = _ints(mesh.polygons, "loop_total", n_faces)
    face = np.repeat(np.arange(n_faces), total)
    # A millionth of the part's size, or of the texture: coincident to any modeller.
    key = np.zeros((n_loops, 6), dtype=np.int64)
    key[:, :3] = np.round((co - low) / span * 1e6)[
        _ints(mesh.loops, "vertex_index", n_loops)
    ]
    if mesh.uv_layers:
        uv = _floats(mesh.uv_layers.active.data, "uv", 2 * n_loops).reshape(-1, 2)
        key[:, 3:5] = np.round(uv * 1e6) * textured[material[face], None]
    # A vertex per material, so slots that meet stay apart.
    key[:, 5] = material[face]
    node = np.unique(key, axis=0, return_inverse=True)[1].ravel()
    following = np.arange(n_loops) + 1
    following[start + total - 1] = start
    label = _components(int(node.max()) + 1, node, node[following])
    return label[node[start]]


def parts(name: str) -> Parts:
    """Import `name`, fitted as a build fits it, into the current file."""
    roots = scene._import(name, "eo")
    bpy.context.view_layer.update()
    objects = sorted(scene._meshes(roots), key=lambda o: o.name)
    for obj in objects:
        # A shared mesh would give two objects' faces one colour.
        if obj.data.users > 1:
            obj.data = obj.data.copy()
    slot_names = [
        [slot.material.name if slot.material else "" for slot in obj.material_slots]
        or [""]
        for obj in objects
    ]
    materials = sorted({n for names in slot_names for n in names})
    textured = np.array([_texture(n) is not None for n in materials])
    face_material, face_island, islands = [], [], []
    for obj, names in zip(objects, slot_names, strict=True):
        mesh = obj.data
        index = _ints(mesh.polygons, "material_index", len(mesh.polygons))
        lookup = np.array([materials.index(n) for n in names])
        material = lookup[np.minimum(index, len(names) - 1)]
        label = _face_components(mesh, material, textured)
        _, first, inverse = np.unique(label, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        face_island.append(len(islands) + rank[inverse])
        counts = np.bincount(inverse, minlength=len(first))
        islands += [
            Island(
                len(islands) + r,
                obj.name,
                materials[material[first[k]]],
                int(first[k]),
                int(counts[k]),
            )
            for r, k in enumerate(order)
        ]
        face_material.append(material)
    return Parts(objects, materials, face_material, face_island, islands)


def face_zones(found: Parts, zones: Zones) -> list[np.ndarray]:
    """Per object, each face's index into the zone vocabulary."""
    vocabulary = list(get_args(ZoneName.__value__))
    by_material = np.array(
        [vocabulary.index(zones.materials[m]) for m in found.materials]
    )
    by_island = np.array(
        [
            vocabulary.index(zones.islands[i.id])
            if i.id in zones.islands
            else by_material[found.materials.index(i.material)]
            for i in found.islands
        ]
    )
    return [by_island[island] for island in found.face_island]


# The views.


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


def _textures(found: Parts) -> None:
    """Workbench's texture mode draws a material's active image node, or its viewport
    colour: point both at the base colour."""
    for material in bpy.data.materials:
        bsdf = _bsdf(material)
        if bsdf is None:
            continue
        source = _source(bsdf.inputs["Base Color"])
        if source is not None:
            material.node_tree.nodes.active = source[2]
        else:
            material.diffuse_color = (*bsdf.inputs["Base Color"].default_value[:3], 1)
    del found


def _paint(found: Parts, per_face: list[np.ndarray]) -> None:
    """Each face's value as a flat colour Workbench can draw: base `DIGIT` digits in
    red, green and blue, each exact in the half float Workbench draws in."""
    for obj, value in zip(found.objects, per_face, strict=True):
        mesh = obj.data
        total = _ints(mesh.polygons, "loop_total", len(mesh.polygons))
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
    corners = np.array([tuple(c) for c in scene._corners(found.objects)])
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
    _textures(found)
    _paint(found, per_face)
    data = bpy.data.cameras.new("zones")
    data.type = "ORTHO"
    camera = bpy.data.objects.new("zones", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    views = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, bearing_deg, elevation_deg in VIEWS:
            _aim(found, bearing_deg, elevation_deg)
            ids, grey, tex = (
                Path(tmp) / f"{k}{s}"
                for k, s in (("id", ".exr"), ("grey", ".png"), ("tex", ".png"))
            )
            _shoot(ids, "VERTEX", "FLAT", "OFF", exr=True)
            _shoot(grey, "SINGLE", "STUDIO", "FXAA", exr=False)
            _shoot(tex, "TEXTURE", "STUDIO", "FXAA", exr=False)
            rgba = render._pixels(ids)[::-1]
            digits = np.rint(rgba[..., :3]).astype(np.int64)
            value = digits @ (DIGIT ** np.arange(3))
            part = np.where(rgba[..., 3] > 0.5, value, -1)
            views.append(
                View(
                    name,
                    part.astype(np.int64),
                    np.asarray(Image.open(grey).convert("L"), np.float32) / 255,
                    np.asarray(_on_backdrop(Image.open(tex))),
                )
            )
    return views


# The sheets.


def _on_backdrop(image: Image.Image) -> Image.Image:
    """A dark hull shows on mid grey, not on the matte."""
    backdrop = Image.new("RGBA", image.size, (110, 110, 110, 255))
    return Image.alpha_composite(backdrop, image.convert("RGBA")).convert("RGB")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


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
    font = _font(15)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
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
    draw.text((box[0] + 4 - left, box[1] + 3 - top), text, font=font, fill=INK)


def _grid(tiles: list[tuple[Image.Image, str]], title: list[str]) -> Image.Image:
    rows = max(1, math.ceil(len(tiles) / COLUMNS))
    sheet = Image.new(
        "RGB",
        (TILE[0] * COLUMNS, TITLE_H + rows * (TILE[1] + CAPTION_H)),
        MATTE,
    )
    draw = ImageDraw.Draw(sheet)
    for k, line in enumerate(title[:2]):
        draw.text((8, 4 + 22 * k), line, font=_font(18 if k == 0 else 14), fill=INK)
    for k, (tile, caption) in enumerate(tiles):
        x = (k % COLUMNS) * TILE[0]
        y = TITLE_H + (k // COLUMNS) * (TILE[1] + CAPTION_H)
        sheet.paste(tile, (x, y))
        draw.text((x + 6, y + TILE[1] + 3), caption, font=_font(14), fill=INK)
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
        draw.text((x + 20, y), f"{label}  {name}"[:52], font=_font(14), fill=INK)
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


# What the shader reads, per part.


def _bsdf(material: bpy.types.Material) -> Any:
    if material is None or material.node_tree is None:
        return None
    return next(
        (n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None
    )


def _source(socket: Any) -> tuple[bpy.types.Image, int | None, Any] | None:
    """The image feeding `socket` through any chain of nodes, its channel (None for
    colour), and its node."""
    if not socket.is_linked:
        return None
    link = socket.links[0]
    node, out = link.from_node, link.from_socket.name
    if node.type == "TEX_IMAGE":
        return (node.image, 3 if out == "Alpha" else None, node) if node.image else None
    channel = {"Red": 0, "Green": 1, "Blue": 2}.get(out)
    for socket_in in node.inputs:
        found = _source(socket_in)
        if found is not None:
            image, inner, tex = found
            return image, channel if channel is not None else inner, tex
    return None


def _texture(name: str) -> bpy.types.Image | None:
    """The image a material's base colour reads, if any."""
    bsdf = _bsdf(bpy.data.materials.get(name)) if name else None
    source = _source(bsdf.inputs["Base Color"]) if bsdf is not None else None
    return source[0] if source is not None else None


def _texels(image: bpy.types.Image) -> np.ndarray:
    """An image's RGBA, top row first, no side over `ATLAS_PX`."""
    width, height = image.size
    buffer = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    step = max(1, math.ceil(max(width, height) / ATLAS_PX))
    return buffer.reshape(height, width, 4)[::-step, ::step]


def _uv_triangles(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    """Each triangle's UVs, (n, 3, 2), and its face."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    n = len(mesh.loop_triangles)
    loops = _ints(mesh.loop_triangles, "loops", 3 * n).reshape(-1, 3)
    face = _ints(mesh.loop_triangles, "polygon_index", n)
    if not mesh.uv_layers:
        return np.zeros((0, 3, 2)), np.zeros(0, dtype=np.int64)
    uv = _floats(mesh.uv_layers.active.data, "uv", 2 * len(mesh.loops)).reshape(-1, 2)
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
        points = np.stack([uv[..., 0] * w, (1 - uv[..., 1]) * h], axis=-1)
        v = value + 1
        corners = np.rint(points).astype(np.int32)
        polygons = [corners[k] for k in range(len(corners))]
        colour = (float(v & 255), float((v >> 8) & 255), float((v >> 16) & 255))
        cv2.fillPoly(code, polygons, colour)
        # fillPoly leaves the edges two triangles share, and a sliver no texel centre.
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
    bsdf = _bsdf(material)
    if bsdf is None:
        if socket == "Base Color" and material is not None:
            return {"colour": _hex(_srgb(np.array(material.diffuse_color[:3])))}
        return {}
    source = _source(bsdf.inputs[socket])
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
    for obj, value in zip(found.objects, per_face, strict=True):
        mesh = obj.data
        co = _floats(mesh.vertices, "co", 3 * len(mesh.vertices)).reshape(-1, 3)
        m = np.array(obj.matrix_world)
        z = co @ m[2, :3] + m[2, 3]
        total = _ints(mesh.polygons, "loop_total", len(mesh.polygons))
        loop_value = np.repeat(value, total)
        loop_z = z[_ints(mesh.loops, "vertex_index", len(mesh.loops))]
        np.minimum.at(low, loop_value, loop_z)
        np.maximum.at(high, loop_value, loop_z)
    return np.stack([low, high], axis=1)


def _triangle_counts(found: Parts, per_face: list[np.ndarray], n: int) -> np.ndarray:
    counts = np.zeros(n, dtype=np.int64)
    for obj, value in zip(found.objects, per_face, strict=True):
        total = _ints(obj.data.polygons, "loop_total", len(obj.data.polygons))
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
    """Each island, with its mean texel colour, and each base colour atlas with every
    island drawn on it."""
    n = len(found.islands)
    heights = _heights(found, found.face_island, n)
    triangles = _triangle_counts(found, found.face_island, n)
    colour = ["" for _ in range(n)]
    atlases: dict[str, Image.Image] = {}
    uvs = [_uv_triangles(obj) for obj in found.objects]
    for name in found.materials:
        bsdf = _bsdf(bpy.data.materials.get(name)) if name else None
        source = _source(bsdf.inputs["Base Color"]) if bsdf is not None else None
        if source is None or source[0].name in atlases:
            continue
        texels = _texels(source[0])
        rgb = texels[..., :3]
        if source[0].colorspace_settings.name not in {"sRGB", "sRGB EOTF"}:
            rgb = _srgb(rgb)
        # Each island's triangles, in island order: a later one paints over.
        uv = np.concatenate([u for u, _ in uvs])
        island = np.concatenate(
            [found.face_island[i][face] for i, (_, face) in enumerate(uvs)]
        )
        mine = np.array([i.material == name for i in found.islands])[island]
        uv, island = uv[mine], island[mine]
        order = np.argsort(island, kind="stable")
        uv, island = uv[order], island[order]
        ids, starts = np.unique(island, return_index=True)
        cover = _raster(
            rgb.shape[:2],
            list(zip(ids.tolist(), np.split(uv, starts[1:]), strict=True)),
        )
        # At each triangle's centre: islands share texels, so a raster's later
        # island would take an earlier one's.
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
        atlases[source[0].name] = _atlas(rgb, cover, texels)
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


# The command.


def sheets(
    name: str, into: Path, kind: Kind | None = None, material: str | None = None
) -> list[Path]:
    """Write the sheets for labelling `name`'s parts, and their records, into `into`.

    Slots by default, islands for an asset of one material. Islands of `material`
    alone, if given, with the rest grey.
    """
    asset = manifest()[name]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    found = parts(name)
    kind = kind or ("slots" if len(found.materials) > 1 else "islands")
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
        own = np.array([material in {None, i.material} for i in found.islands] + [0])
        if material is not None and not own.any():
            raise ValueError(f"{name} has no material {material!r}")
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
    best = np.zeros(len(labels), dtype=np.int64)
    for view in views:
        counts = np.bincount(view.part[view.part >= 0], minlength=len(labels))
        best = np.maximum(best, counts)
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
    body = {
        "asset": name,
        "description": asset.description,
        "draught_m": asset.draught_m,
        f"{kind[:-1]}_count": len(records),
        kind: [r for k, r in enumerate(records) if labels[k]],
    }
    (path := into / f"{kind}.json").write_text(json.dumps(body, indent=1) + "\n")
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
    check(name, found, asset.zones)
    vocabulary = list(get_args(ZoneName.__value__))
    colours = np.array([ZONE_COLOURS[z] for z in vocabulary], dtype=np.uint8)
    per_face = face_zones(found, asset.zones)
    views = _views(found, per_face)
    triangles = _triangle_counts(found, per_face, len(vocabulary))
    present = np.flatnonzero(triangles)
    shares = [f"{100 * triangles[z] / triangles.sum():.0f}%" for z in present]
    head = f"{name}: thermal zones, {asset.category}, {asset.size}"
    sheet = overview(views, colours, vocabulary, [head, "share of triangles below"])
    sheet = _stack(
        sheet, legend(colours[present], shares, [vocabulary[z] for z in present])
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def check(name: str, found: Parts, zones: Zones) -> None:
    """Refuse zones that name a material the import lacks, or count other islands."""
    if missing := sorted(set(zones.materials) - set(found.materials)):
        raise ValueError(f"{name}: no material {missing} in the import")
    if unzoned := sorted(set(found.materials) - set(zones.materials)):
        raise ValueError(f"{name}: no zone for {unzoned}")
    if zones.island_count is not None and zones.island_count != len(found.islands):
        raise ValueError(
            f"{name}: {len(found.islands)} islands, assets.toml counts "
            f"{zones.island_count}: the import's faces moved"
        )
