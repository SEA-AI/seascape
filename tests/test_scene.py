"""The built scene, measured.

Blender is one global session, so each class rebuilds in its own band on entry.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from statistics import NormalDist
from typing import Any

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape.assets import Buoy, Debris, Hull, Turbine, manifest, skies
from seascape.blender import blend, scene, sea
from seascape.blender.heat import _sunlit_emission, surroundings, zone_skin
from seascape.blender.meshes import _fit, _images, _joined, box_corners, mesh_objects
from seascape.blender.rigs import boresight_deg, views
from seascape.blender.sea import memory
from seascape.blender.sea.daylight import SUN_SLOPE_RADIUS
from seascape.blender.sea.geometry import _triples
from seascape.blender.sea.pixel import FOOTPRINT_RANGE_M
from seascape.blender.sea.radiance import (
    CLOUD_BEARINGS,
    CLOUD_MIRRORS,
    CLOUD_SLOPES,
    CLOUD_VIEWS,
    MIRROR_SAMPLES,
    SLOPE_ROWS,
    VIEW_ROWS,
    _cloud_image,
    _mirror_columns,
    _row_sigmas,
    _view_rows,
)
from seascape.blender.sea.wind import (
    GUST_CELLS,
    GUST_SPACING_M,
    GUST_TILE_M,
    _looked_up_sigma,
)
from seascape.blender.sky import _haze_ranges_m, _sky_round, haze, sea_light
from seascape.blender.vessels import (
    Condition,
    Hulls,
    _vessel,
    heading_rad,
    waterline_m,
)
from seascape.config import (
    Band,
    Camera,
    Mount,
    Object,
    Outputs,
    Rig,
    Scenario,
    Sky,
    load,
    substream,
)
from seascape.outputs.calibration import CameraCalibration
from seascape.physics import aerosol, atmosphere, earth, lwir, thermal, waves
from tests.scenarios import (
    BASELINE,
    DRIFTING,
    OPEN_SEA,
    preset_target,
    target,
    variant,
    wave_field,
    white_photo,
)

SCENARIO = load(BASELINE)
# The manifest fields a fit does not read.
ASSET: dict[str, Any] = {
    "description": "x",
    "category": "x",
    "category_id": 1,
    "url": "x",
    "sha256": "0" * 64,
    "triangles": 1,
    "texture_px": (),
    "licence": "x",
    "attribution": "x",
}
# A lens of the test's own: no shipped preset is pinned.
LENS: dict[str, Any] = {
    "band": "eo",
    "hfov_deg": 49.0,
    "width_px": 3840,
    "height_px": 2160,
}


def camera_of(mount: Mount) -> bpy.types.Object:
    return bpy.data.objects[mount.name]


def _in_frame(camera: CameraCalibration, direction: Vector | np.ndarray) -> bool:
    """Whether a world direction from the camera lands on its sensor."""
    rotation = np.array(camera.extrinsics["world"])[:3, :3]
    x, y, z = np.array(camera.K) @ (rotation.T @ np.asarray(direction))
    # Pixel centres sit at integers, so the sensor spans -0.5 to size - 0.5.
    return bool(
        z > 0
        and -0.5 <= x / z <= camera.width_px - 0.5
        and -0.5 <= y / z <= camera.height_px - 0.5
    )


def test_the_sun_is_out_of_every_frame() -> None:
    scenario = load(BASELINE)
    ownship = scenario.ownship.model_copy(update={"asset": None})
    bare = scenario.model_copy(
        update={"objects": [], "targets": None, "ownship": ownship}
    )
    built = scene.build(bare, "eo")
    sun = np.array(scenario.sky.sun_vector)
    for mount in scenario.mounts:
        camera = scene.calibrate(built, mount, "")
        assert not _in_frame(camera, sun), mount.name


def upstream(socket: bpy.types.NodeSocket) -> set[str]:
    """The names of every node feeding `socket`."""
    seen, todo = set(), [socket]
    while todo:
        for link in todo.pop().links:
            if link.from_node.name not in seen:
                seen.add(link.from_node.name)
                todo.extend(link.from_node.inputs)
    return seen


_MATH = {
    "ADD": lambda a, b, *_: a + b,
    "SUBTRACT": lambda a, b, *_: a - b,
    "MULTIPLY": lambda a, b, *_: a * b,
    "MULTIPLY_ADD": lambda a, b, c: a * b + c,
    "FRACT": lambda a, *_: a - math.floor(a),
    "ABSOLUTE": lambda a, *_: abs(a),
    "ARCTAN2": lambda a, b, *_: math.atan2(a, b),
}


def _evaluated(socket: bpy.types.NodeSocket) -> float:
    """A chain of Value and Math nodes, and the vectors they pass through, evaluated
    as Blender does.
    """
    node = socket.node
    if node.bl_idname == "ShaderNodeValue":
        return socket.default_value
    if node.bl_idname == "ShaderNodeSeparateXYZ":
        packed = node.inputs["Vector"].links[0].from_socket
        assert packed.node.bl_idname == "ShaderNodeCombineXYZ"
        component = packed.node.inputs[socket.name]
        if component.is_linked:
            return _evaluated(component.links[0].from_socket)
        return component.default_value
    inputs = (
        _evaluated(i.links[0].from_socket) if i.is_linked else i.default_value
        for i in node.inputs
    )
    return _MATH[node.operation](*inputs)


def under_the_haze(material: bpy.types.Material) -> bpy.types.Node:
    output = material.node_tree.get_output_node("CYCLES")
    haze = output.inputs["Surface"].links[0].from_node
    return haze.inputs["Shader"].links[0].from_node


def _sea_node(name: str) -> bpy.types.ShaderNode:
    return bpy.data.materials["sea"].node_tree.nodes[name]


def baked(name: str) -> np.ndarray:
    """The red channel of a lookup image, rows bottom first."""
    return np.squeeze(blend.pixels(bpy.data.images[name])[..., 0])


def _hazed_sky_m(
    sky: Sky, elevation: np.ndarray | float, cloud: float | None = None
) -> np.ndarray:
    """`lwir.sky_radiance`, or `lwir.cloudy_sky_radiance` under a `cloud` share, at
    `elevation` through the aerosol.
    """
    if cloud is None:
        radiance = lwir.sky_radiance(elevation, sky.t_air_k, sky.atmosphere)
    else:
        radiance = lwir.cloudy_sky_radiance(
            elevation, cloud, sky.cloud_base_m, sky.t_air_k, sky.atmosphere
        )
    return aerosol.sky_radiance(
        radiance, elevation, sky.visibility_km, sky.t_air_k, sky.atmosphere
    )


def _texel_elevations() -> np.ndarray:
    """The elevation at each texel centre of a table read by sin(elevation)."""
    return np.arcsin((np.arange(blend.CURVE_SAMPLES) + 0.5) / blend.CURVE_SAMPLES)


def _blocked(origin: Vector, target: Vector) -> bool:
    ray = target - origin
    hit, *_ = bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(),
        origin,
        ray.normalized(),
        distance=ray.length * 0.999,
    )
    return bool(hit)


def _footprints_m(n: int) -> np.ndarray:
    """The footprint at each texel centre of a table the shader reads by log
    footprint.
    """
    low, high = FOOTPRINT_RANGE_M
    return low * (high / low) ** ((np.arange(n) + 0.5) / n)


def counts() -> tuple[int, ...]:
    return tuple(
        len(block) for block in (bpy.data.objects, bpy.data.materials, bpy.data.images)
    )


def _normal_waves(
    scenario: Scenario, kept: tuple[waves.Wave, ...] | None
) -> tuple[tuple[waves.Wave, ...], tuple[waves.Wave, ...]]:
    """The scenario's wind and swell, or the `kept` part of them, less the waves drawn
    as geometry.
    """
    wind, swell = scenario.wind_waves(), scenario.swell_waves()
    if kept is not None:
        wind = tuple(w for w in wind if w in kept)
        swell = tuple(w for w in swell if w in kept)
    return sea.displaced((wind, swell))[1]


def shaded(
    scenario: Scenario, kept: tuple[waves.Wave, ...] | None = None
) -> list[waves.Wave]:
    """What the normal draws of the scenario's field, or of the `kept` part of it."""
    wind, swell = _normal_waves(scenario, kept)
    return [*wind, *swell]


def drawn(
    scenario: Scenario, kept: tuple[waves.Wave, ...] | None = None
) -> list[tuple[waves.Wave, bpy.types.NodeInputs, int]]:
    """Each wave of the built sea's normal, the inputs of the node drawing it, and its
    lane there, of the scenario's field or the `kept` part of it, less the waves drawn
    as geometry. Every other lane must be empty.
    """
    nodes = bpy.data.node_groups["sea_waves"].nodes
    wind, swell = _normal_waves(scenario, kept)
    triples = _triples(wind) + _triples(swell)
    found = []
    for i, triple in enumerate(triples):
        inputs = nodes[f"waves_{i}"].inputs
        for lane, wave in enumerate(triple):
            if wave is None:
                assert inputs["Slope X"].default_value[lane] == 0.0
                assert inputs["Slope Y"].default_value[lane] == 0.0
                assert inputs["Lift"].default_value[lane] == 0.0
            else:
                found.append((wave, inputs, lane))
    return found


class TestEoBand:
    """Where things are, where cameras look, and what the EO band draws."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> scene.Built:
        return scene.build(SCENARIO, "eo")

    def test_a_flat_rig_points_where_the_scenario_asked(self) -> None:
        """The one negation. Two of them cancel and the whole rig mirrors unnoticed."""
        for mount in SCENARIO.mounts:
            bearing, _ = boresight_deg(camera_of(mount))
            assert bearing == pytest.approx(mount.nominal_bearing_deg, abs=1e-6)

    def test_the_foam_grid_takes_each_frame_s_view_where_the_water_drifted_from(
        self, built: scene.Built
    ) -> None:
        """Water seen at x at time t is on the drifting grid at x - v t."""
        clip = variant(BASELINE, outputs={"duration_s": 1.0})
        calm = variant(BASELINE, outputs={"duration_s": 1.0}, sea={"wind_speed_mps": 0})
        drift = sea.surface_drift(clip.sea, clip.outputs)
        assert sea.surface_drift(calm.sea, calm.outputs) == (0.0, 0.0) != drift
        moved = memory._rays(clip, views(clip, "eo", built.cameras))
        held = memory._rays(calm, views(calm, "eo", built.cameras))
        per_frame = len(moved) // len(clip.outputs.times_s)
        for i, ((at, _, _), (still, _, _)) in enumerate(zip(moved, held, strict=True)):
            t_s = clip.outputs.times_s[i // per_frame]
            expected = still - Vector((drift[0] * t_s, drift[1] * t_s, 0.0))
            assert (at - expected).length < 1e-9

    def test_the_foam_grid_takes_each_camera_s_view_at_its_own_aspect(
        self, built: scene.Built
    ) -> None:
        tall = variant(BASELINE, cameras={"eo": {"width_px": 512, "height_px": 640}})
        mount = tall.band_mounts("eo")[0]
        back = built.cameras[mount.name].matrix_world.to_3x3().inverted()
        rays = memory._rays(
            tall, [v for v in views(tall, "eo", built.cameras) if v.mount == mount]
        )
        seen = [back @ ray for _, ray, _ in rays]
        x = max(abs(v.x / v.z) for v in seen)
        y = max(abs(v.y / v.z) for v in seen)
        assert y / x == pytest.approx(640 / 512)

    def test_cameras_carry_their_field_of_view_horizontally(self) -> None:
        """AUTO fits the angle to the longer image side, flipping a portrait sensor."""
        for mount in SCENARIO.mounts:
            data = camera_of(mount).data
            assert data.sensor_fit == "HORIZONTAL"
            assert math.degrees(data.angle_x) == pytest.approx(mount.camera.hfov_deg)

    def test_every_3d_view_clips_past_the_sea(self) -> None:
        corner_m = math.sqrt(2) * sea.sea_reach_m(SCENARIO)
        views = [
            space
            for screen in bpy.data.screens
            for area in screen.areas
            if area.type == "VIEW_3D"
            for space in area.spaces
            if space.type == "VIEW_3D"
        ]

        assert views, "no 3D view to clip: the assertions below would pass on nothing"
        for space in views:
            assert space.clip_end > corner_m

    def test_the_far_clip_clears_every_target(self) -> None:
        furthest = max(spec.range_m for spec in SCENARIO.objects)
        for mount in SCENARIO.mounts:
            assert camera_of(mount).data.clip_end > furthest

    def test_a_target_lands_at_its_range_and_bearing(self) -> None:
        for spec in SCENARIO.objects:
            east, north, _ = bpy.data.objects[spec.asset].location
            assert math.hypot(east, north) == pytest.approx(spec.range_m)
            assert math.degrees(math.atan2(east, north)) == pytest.approx(
                spec.bearing_deg
            )

    def test_the_ship_is_in_every_frame(self, built: scene.Built) -> None:
        """Both bands' tests measure the one ship."""
        (spec,) = SCENARIO.objects
        ship = bpy.data.objects[spec.asset].matrix_world.translation
        for mount in SCENARIO.mounts:
            eye = camera_of(mount).matrix_world.translation
            camera = scene.calibrate(built, mount, "")
            assert _in_frame(camera, ship - eye), mount.name

    def test_a_target_is_fitted_to_its_manifest_length(self) -> None:
        """The mesh arrives in its author's units; unfitted it is a speck at range."""
        for spec in SCENARIO.objects:
            anchor = bpy.data.objects[spec.asset]
            (attitude,) = anchor.children
            into_hull = attitude.matrix_world.inverted()
            corners = [
                into_hull @ part.matrix_world @ Vector(corner)
                for part in mesh_objects([anchor])
                for corner in part.bound_box
            ]
            axes = list(zip(*corners, strict=True))
            asset = manifest()[spec.asset]
            if isinstance(asset, Hull):
                assert max(axes[1]) - min(axes[1]) == pytest.approx(asset.length_m)
            elif isinstance(asset, Buoy):
                assert max(axes[2]) - min(axes[2]) == pytest.approx(asset.height_m)
            # 1 mm: the fit runs through float32 mesh coordinates.
            assert min(axes[2]) == pytest.approx(-asset.draught_m, abs=1e-3), (
                "keel sits at the manifest draught, not on the surface"
            )
            assert max(axes[2]) > 0.0, "and the rest of it is above water"

    def test_each_wave_fades_at_its_own_wavenumber(self) -> None:
        for wave, inputs, lane in drawn(SCENARIO):
            assert inputs["Half K Sq"].default_value[lane] == pytest.approx(
                (wave.k_rad_m / 2) ** 2, rel=1e-6
            )

    def test_each_wave_fades_over_the_headings_it_stands_for(self) -> None:
        for wave, inputs, lane in drawn(SCENARIO):
            assert tuple(
                inputs[name].default_value[lane] for name in ("Tx", "Ty", "Band")
            ) == pytest.approx(waves.fade_axis(wave), rel=1e-6, abs=1e-7)

    def test_the_sea_carries_the_scenario_s_wave_field(self) -> None:
        field = wave_field(SCENARIO)
        assert len(field) == waves.WIND_WAVES
        carried = drawn(SCENARIO)
        tall, _ = sea.displaced((SCENARIO.wind_waves(), SCENARIO.swell_waves()))
        assert tall
        assert [wave for wave, _, _ in carried] == [w for w in field if w not in tall]
        for wave, inputs, lane in carried:
            k_east, k_north = wave.k_east_rad_m, wave.k_north_rad_m
            assert tuple(
                inputs[name].default_value[lane]
                for name in ("Kx", "Ky", "W", "Phase", "Slope X", "Slope Y")
            ) == pytest.approx(
                (
                    k_east,
                    k_north,
                    -wave.omega_rad_s,
                    wave.phase_rad,
                    wave.amplitude_m * k_east,
                    wave.amplitude_m * k_north,
                ),
                rel=1e-6,
                abs=1e-9,
            )

    def test_the_sea_reaches_past_its_own_horizon(self) -> None:
        """The grid must contain the tangent point, or its edge becomes the horizon."""
        corners = [
            bpy.data.objects["sea"].matrix_world @ Vector(c)
            for c in bpy.data.objects["sea"].bound_box
        ]
        reach = min(max(abs(v.x), abs(v.y)) for v in corners)
        horizon = earth.horizon_m(
            SCENARIO.rigs["bow"].height_m, SCENARIO.sea.refraction_k
        )

        assert reach > horizon
        assert reach > max(spec.range_m for spec in SCENARIO.objects)

    def test_a_hull_floats_on_the_sea_and_not_on_the_tangent_plane(self) -> None:
        """Hulls left at z = 0 fly with range and bearing still right, so nothing else
        catches it.
        """
        radius = earth.earth_radius_m(SCENARIO.sea.refraction_k)

        for spec in SCENARIO.objects:
            anchor = bpy.data.objects[spec.asset]
            east, north, up = anchor.matrix_world.translation

            assert up == pytest.approx(earth.sea_z_m(east, north, radius), abs=1e-3)

    def test_a_hull_beyond_the_horizon_is_cut_off(self) -> None:
        """A target past the horizon shows its waterline when it should be hull-down."""
        eye = Vector((0.0, 0.0, SCENARIO.rigs["bow"].height_m))
        radius = earth.earth_radius_m(SCENARIO.sea.refraction_k)
        beyond = 18_000.0
        surface = beyond * beyond / (2.0 * radius)
        horizon = earth.horizon_m(
            SCENARIO.rigs["bow"].height_m, SCENARIO.sea.refraction_k
        )
        assert horizon < beyond < sea.sea_reach_m(SCENARIO)

        waterline = _blocked(eye, Vector((0.0, beyond, -surface)))
        mast = _blocked(eye, Vector((0.0, beyond, 30.0 - surface)))

        assert waterline, "the bulge has to hide the hull"
        assert not mast, "and leave what stands above it"

    def test_the_glitter_spreads_over_the_slope_each_pixel_leaves_out(self) -> None:
        assert _sea_node("sea_specular").inputs["Roughness"].is_linked
        wind, swell = SCENARIO.wind_waves(), SCENARIO.swell_waves()
        speed = SCENARIO.sea.wind_speed_mps
        axis_rad = math.radians(SCENARIO.sea.wind_from_deg)
        for i, axis in enumerate(("upwind", "crosswind")):
            table = baked(f"sea_unresolved_variance_{axis}")
            footprints = _footprints_m(len(table))
            for t in (0, len(table) // 2, len(table) - 1):
                expected = waves.unresolved_slope_variances(
                    speed, axis_rad, wind, swell, footprints[t]
                )[i]
                assert table[t] == pytest.approx(expected, rel=1e-5), axis

    def test_a_gust_adds_cox_and_munk_s_variance_at_the_local_wind(self) -> None:
        roughness = _sea_node("sea_specular")
        assert {
            "gust",
            "sea_gust_variance",
            "sea_unresolved_variance_upwind",
            "sea_unresolved_variance_crosswind",
        } <= upstream(roughness.inputs["Roughness"])
        speed = SCENARIO.sea.wind_speed_mps
        per_gust = _sea_node("sea_gust_variance").inputs["Value_001"].default_value
        assert per_gust == pytest.approx(waves.gust_slope_variance(speed), rel=1e-5)
        tile = baked("sea_gust")
        seeded = waves.von_karman_field(
            substream(SCENARIO.seed, "sea/gust"),
            GUST_CELLS,
            GUST_SPACING_M,
            waves.GUST_LENGTH_M,
        )
        assert tile == pytest.approx(seeded / _looked_up_sigma(seeded), abs=1e-5)

    def test_the_gusts_are_carried_downwind_at_the_mean_wind(self) -> None:
        carry = _sea_node("gust_carry")
        per_m = 1 / GUST_TILE_M
        downwind = math.radians(SCENARIO.sea.wind_from_deg + 180.0)
        speed = SCENARIO.sea.wind_speed_mps
        # The noise read at x - U t is the noise at x carried U t downwind.
        assert tuple(carry.inputs["Vector"].default_value) == pytest.approx(
            (
                -speed * per_m * math.sin(downwind),
                -speed * per_m * math.cos(downwind),
                0,
            ),
            abs=1e-9,
        )

    def test_foam_glitter_and_gusts_read_the_sea_the_waves_are_drawn_on(self) -> None:
        """The foam's memory and lace, the glitter and the gusts all read the water's
        rest position, so none slides off the crests.
        """
        tree = bpy.data.materials["sea"].node_tree
        memory = [n for n in tree.nodes if n.name.startswith("sea_foam_")]
        assert memory
        for socket in (
            *(n.inputs["Vector"] for n in memory),
            tree.nodes["whitecap_lace_axes"].inputs["X"],
            tree.nodes["glitter_cells"].inputs["Vector"],
            tree.nodes["gust_axes"].inputs["X"],
        ):
            assert "sea_rest" in upstream(socket), socket.node.name

    @pytest.mark.parametrize("tile", ["sea_foam", "sea_foam_lattice", "sea_gust"])
    def test_the_foam_tile_reads_at_unit_sigma(self, tile: str) -> None:
        assert _looked_up_sigma(baked(tile)) == pytest.approx(1.0, rel=1e-5)

    def test_the_lobe_keeps_its_mean_roughness(self) -> None:
        """The glitter carries the unresolved slope's spread in EO."""
        assert "sea_group_scale" not in bpy.data.materials["sea"].node_tree.nodes

    def test_the_sea_glitters_one_specular_point_per_cell(self) -> None:
        wind = SCENARIO.wind_waves()
        nodes = bpy.data.materials["sea"].node_tree.nodes
        scale = nodes["glitter_cells"].inputs["Scale"].default_value
        assert scale == pytest.approx(1 / math.sqrt(waves.specular_cell_m2(wind)))
        twinkle = nodes["glitter_draw"].inputs[0].links[0].from_node
        assert twinkle.inputs[1].default_value == pytest.approx(waves.twinkle_hz(wind))

    def test_the_glint_is_the_sky_texture_s_sun(self) -> None:
        sky = bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]
        assert sky.sun_size == pytest.approx(4 * SUN_SLOPE_RADIUS)

    def test_the_air_hazes_towards_the_horizon_sky(self) -> None:
        nodes = bpy.data.node_groups["haze"].nodes
        beta = nodes["haze_beta"]
        assert "haze_start" in upstream(beta.inputs[0])
        assert beta.inputs[1].default_value == pytest.approx(
            SCENARIO.sky.extinction_per_m
        )
        sky = bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]
        airlight = nodes["haze_sky"]
        for name in ("sun_elevation", "sun_rotation", "aerosol_density"):
            assert getattr(airlight, name) == getattr(sky, name)

    def test_the_airlight_is_the_sky_ahead_of_the_ray(self) -> None:
        """Incoming points back at the camera; unflipped, the haze would take the sky
        behind it.
        """
        nodes = bpy.data.node_groups["haze"].nodes
        ahead, horizon = nodes["haze_ahead"], nodes["haze_horizon"]
        assert ahead.inputs["Vector"].links[0].from_socket.name == "Incoming"
        assert ahead.inputs["Scale"].default_value == -1.0
        assert horizon.inputs[0].links[0].from_node == ahead
        assert tuple(horizon.inputs[1].default_value) == (-1.0, -1.0, 0.0)
        (into_sky,) = horizon.outputs["Vector"].links
        assert into_sky.to_node.name == "haze_sky"
        assert into_sky.is_valid

    def test_every_surface_mixes_towards_the_airlight(self) -> None:
        mix = bpy.data.node_groups["haze"].nodes["haze_mix"]
        assert mix.inputs[1].links[0].from_node.bl_idname == "NodeGroupInput"
        assert mix.inputs[2].links[0].from_node.name == "haze_airlight"
        for material in bpy.data.materials:
            output = material.node_tree.get_output_node("CYCLES")
            (surface,) = output.inputs["Surface"].links
            assert surface.from_node.name == "haze", material.name

    def test_no_surface_is_sampled_as_a_light(self) -> None:
        for material in bpy.data.materials:
            assert material.cycles.emission_sampling == "NONE", material.name

    def test_a_hazed_material_is_not_hazed_again(self) -> None:
        material = bpy.data.materials["sea"]
        haze(material)
        assert under_the_haze(material).bl_idname != "ShaderNodeGroup"

    def test_building_twice_leaves_the_same_scene(self) -> None:
        """Node trees leak when a build appends to what is already there."""
        before = counts()
        scene.build(SCENARIO, "eo")
        assert counts() == before


def test_a_fetch_grows_the_scenario_s_sea_no_further() -> None:
    young, full = variant(sea={"fetch_km": 10.0}), variant()

    longest_young = min(w.omega_rad_s for w in young.wind_waves())
    longest_full = min(w.omega_rad_s for w in full.wind_waves())

    assert longest_young > longest_full


@pytest.mark.parametrize("band", ["eo", "ir"])
def test_a_saved_build_reopens_with_its_lookup_tables(
    band: Band, tmp_path: Path
) -> None:
    """`seascape build` saves a .blend for Blender to open; an unpacked table would
    reopen as its fill colour.
    """
    scene.build(load(OPEN_SEA), band)
    tables = [
        image
        for image in bpy.data.images
        if image.packed_file or image.source == "GENERATED"
    ]
    assert tables
    path = tmp_path / f"{band}.blend"
    # A copy, so the session keeps its own file.
    bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
    with bpy.data.libraries.load(str(path)) as (_, loaded):
        loaded.images = [image.name for image in tables]
    for written, read in zip(tables, loaded.images, strict=True):
        before, after = (np.empty(len(written.pixels), np.float32) for _ in range(2))
        written.pixels.foreach_get(before)
        read.pixels.foreach_get(after)
        bpy.data.images.remove(read)
        assert np.array_equal(before, after), written.name


def test_an_image_inside_a_node_group_is_counted() -> None:
    image = bpy.data.images.new("deep", 8, 4)
    group = bpy.data.node_groups.new("wrap", "ShaderNodeTree")
    group.nodes.new("ShaderNodeTexImage").image = image
    outer = bpy.data.node_groups.new("outer", "ShaderNodeTree")
    outer.nodes.new("ShaderNodeGroup").node_tree = group
    assert _images(outer) == {image}


def test_a_turned_part_is_measured_by_its_vertices() -> None:
    """A diagonal rod turned onto x has no y extent; its `bound_box`, turned with it,
    does.
    """
    mesh = bpy.data.meshes.new("rod")
    mesh.from_pydata([(-1, -1, 0), (1, 1, 0), (1, 1, 0.01)], [], [(0, 1, 2)])
    rod = bpy.data.objects.new("rod", mesh)
    bpy.context.scene.collection.objects.link(rod)
    rod.rotation_euler.z = math.radians(-45.0)
    bpy.context.view_layer.update()
    try:
        ys = [c.y for c in box_corners([rod])]
    finally:
        bpy.data.objects.remove(rod)
        bpy.data.meshes.remove(mesh)

    assert max(ys) - min(ys) == pytest.approx(0.0, abs=1e-6)


def test_a_skinned_part_joins_in_its_pose(skinned: bpy.types.Object) -> None:
    """The joined hull holds the part's posed vertices, not its rest pose."""
    part = skinned
    expected = [
        part.matrix_world @ v.co
        for v in part.evaluated_get(bpy.context.evaluated_depsgraph_get())
        .to_mesh()
        .vertices
    ]

    hull = _joined([part.parent])
    try:
        got = [v.co.copy() for v in hull.data.vertices]
    finally:
        bpy.data.objects.remove(hull)

    assert np.array(got) == pytest.approx(np.array(expected))
    assert expected[0].z == pytest.approx(5.0), "the pose moves the part"


@pytest.mark.parametrize(
    ("bow_deg", "bow_corner"),
    [(0.0, (0, 5, 0)), (90.0, (5, 0, 0)), (180.0, (0, -5, 0)), (270.0, (-5, 0, 0))],
)
def test_a_hull_is_fitted_along_its_own_bow_axis(bow_deg, bow_corner) -> None:
    """180 is its own inverse: the shipped hull passes with a sign error or the
    length measured along the beam. Any other bow catches both.
    """
    asset = Hull(
        **ASSET,
        kind="hull",
        material="steel",
        length_m=200.0,
        draught_m=5.0,
        bow_deg=bow_deg,
    )
    long, beam = Vector(bow_corner), Vector((-bow_corner[1], bow_corner[0], 0)) / 5
    corners = [
        s * long + b * beam + Vector((0, 0, z))
        for s in (-1, 1)
        for b in (-1, 1)
        for z in (0, 1)
    ]

    fit = _fit(corners, asset)
    fitted = [fit @ c for c in corners]

    ys = [c.y for c in fitted]
    assert max(ys) - min(ys) == pytest.approx(200.0), "scaled along the bow axis"
    assert max(c.x for c in fitted) - min(c.x for c in fitted) == pytest.approx(40.0)
    assert min(c.z for c in fitted) == pytest.approx(-5.0), "keel at the draught"
    assert (fit @ long).y == pytest.approx(100.0), "and the bow ends up at +Y"


def test_debris_is_fitted_along_its_longest_side_where_it_was_authored() -> None:
    debris = Debris(
        **ASSET, kind="debris", material="wood", length_m=6.0, draught_m=1.5
    )
    corners = [Vector((x, y, z)) for x in (0, 1) for y in (0, 3) for z in (0, 1)]

    fitted = [_fit(corners, debris) @ c for c in corners]

    assert max(c.y for c in fitted) - min(c.y for c in fitted) == pytest.approx(6.0)
    assert max(c.x for c in fitted) == pytest.approx(1.0), "not turned to its bow"
    assert min(c.z for c in fitted) == pytest.approx(-1.5)


def test_a_buoy_is_fitted_keel_to_top_where_it_was_authored() -> None:
    buoy = Buoy(**ASSET, kind="buoy", material="steel", height_m=6.0, draught_m=1.5)
    corners = [Vector((x, y, z)) for x in (9, 11) for y in (0, 1) for z in (3, 6)]

    fitted = [_fit(corners, buoy) @ c for c in corners]

    assert max(c.z for c in fitted) == pytest.approx(4.5), "6 m tall, 1.5 m under"
    assert min(c.x for c in fitted) == pytest.approx(-2.0), "centred, not turned"
    assert min(c.y for c in fitted) == pytest.approx(-1.0)


def test_a_turbine_is_fitted_on_its_tower() -> None:
    turbine = manifest()["turbine_15mw_monopile"].model_copy(
        update={"height_m": 200.0, "draught_m": 20.0}
    )
    # Its tower at the origin, its rotor 10 units ahead and 50 either side.
    corners = [Vector((x, y, z)) for x in (-50, 50) for y in (-5, 10) for z in (0, 100)]

    fitted = [_fit(corners, turbine) @ c for c in corners]

    assert min(c.y for c in fitted) == pytest.approx(-10.0), "on its tower"
    assert min(c.x for c in fitted) == pytest.approx(-100.0)
    assert min(c.z for c in fitted) == pytest.approx(-20.0)


def test_a_pitched_rig_rolls_the_horizon_of_its_off_axis_cameras() -> None:
    """Horizon rolls by asin(sin(pitch) sin(yaw))."""
    pitch_deg, yaw_deg = -5.0, 40.0
    scenario = variant(
        rig={"pitch_deg": pitch_deg},
        cameras={"eo": Camera(**LENS, yaw_deg=yaw_deg)},
    )
    expected = math.degrees(
        math.asin(math.sin(math.radians(pitch_deg)) * math.sin(blend.yaw(yaw_deg)))
    )

    scene.build(scenario, "eo")
    across = bpy.data.objects[scenario.mounts[0].name].matrix_world.to_3x3() @ Vector(
        (1.0, 0.0, 0.0)
    )

    assert math.degrees(math.asin(across.normalized().z)) == pytest.approx(
        expected, abs=1e-6
    )


def test_a_pitched_lens_points_exactly_where_it_was_asked_to() -> None:
    """Rx inside the camera's yaw: neither angle disturbs the other, nor the horizon."""
    pitch_deg = -10.0
    lens = Camera(**LENS, yaw_deg=40.0, pitch_deg=pitch_deg)
    scenario = variant(rig={"yaw_deg": -60.0}, cameras={"eo": lens})

    scene.build(scenario, "eo")
    mount = scenario.mounts[0]

    camera = bpy.data.objects[mount.name]
    bearing, elevation = boresight_deg(camera)
    assert bearing == pytest.approx(mount.nominal_bearing_deg, abs=1e-4)
    assert elevation == pytest.approx(pitch_deg, abs=1e-4)
    across = camera.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))
    assert across.normalized().z == pytest.approx(0.0, abs=1e-6)


def test_pitch_moves_an_off_axis_camera_off_its_bearing_and_leaves_the_centre() -> None:
    """The rig is yawed off the bow, so its pitch sits between two yaws and
    moves the off-axis lens off both its bearing and its own pitch. The centre camera
    holds to microdegrees, not zero: matrix_world is float32.
    """
    scenario = variant(
        rig={"pitch_deg": -5.0, "yaw_deg": -60.0},
        cameras={
            "eo": Camera(**LENS),
            "eo_off": Camera(**LENS, yaw_deg=40.0, pitch_deg=-10.0),
        },
    )
    scene.build(scenario, "eo")
    centre, off_axis = (
        (boresight_deg(bpy.data.objects[m.name]), m.nominal_bearing_deg)
        for m in scenario.band_mounts("eo")
    )

    assert centre[0][0] == pytest.approx(centre[1], abs=1e-4)
    (bearing, elevation), nominal = off_axis
    assert abs(bearing - nominal) > 0.1
    assert abs(elevation - (-10.0)) > 0.1


def test_an_ownship_with_no_hull_still_carries_the_rig() -> None:
    scene.build(variant(ownship={"roll_deg": 5.0}), "eo")

    right = camera_of(SCENARIO.mounts[0]).matrix_world.to_3x3() @ Vector(
        (1.0, 0.0, 0.0)
    )

    assert math.degrees(math.asin(-right.z)) == pytest.approx(5.0)


def test_a_lighter_ownship_lifts_its_rig_with_its_hull() -> None:
    ship = manifest()["cargo_ship"]
    assert ship.draught_range_m is not None
    lightest_m, _ = ship.draught_range_m
    rise_m = ship.draught_m - lightest_m
    heights = []
    for draught_m in (None, lightest_m):
        built = scene.build(
            variant(ownship={"asset": "cargo_ship", "draught_m": draught_m}), "eo"
        )
        hull = mesh_objects(built.vessel.children_recursive)
        top_m = max(c.z for c in box_corners(hull))
        camera_m = camera_of(SCENARIO.mounts[0]).matrix_world.translation.z
        heights.append((top_m, camera_m))

    (top_m, camera_m), (lifted_top_m, lifted_camera_m) = heights
    assert lifted_top_m - top_m == pytest.approx(rise_m, abs=1e-3)
    assert lifted_camera_m - camera_m == pytest.approx(rise_m, abs=1e-3)


def test_a_lighter_ownship_sees_as_far_as_its_rig_raised_by_the_rise() -> None:
    ship = manifest()["cargo_ship"]
    assert ship.draught_range_m is not None
    lightest_m, _ = ship.draught_range_m
    yacht = target("yacht", 900.0)
    lighter = variant(
        ownship={"asset": "cargo_ship", "draught_m": lightest_m}, objects=[yacht]
    )
    (rig,) = lighter.rigs.values()
    raised = variant(
        ownship={"asset": "cargo_ship"},
        rig={"height_m": rig.height_m + lighter.ownship.rise_m},
        objects=[yacht],
    )
    built = scene.build(lighter, "eo")
    (anchor,) = built.targets["yacht"]

    assert sea.sea_reach_m(lighter) == pytest.approx(sea.sea_reach_m(raised))
    assert scene._sightline_m(lighter, [anchor]) == pytest.approx(
        scene._sightline_m(raised, [anchor])
    )


def test_an_ownship_hull_rides_the_sea_under_its_script() -> None:
    """The sea pitches, rolls and heaves the hull's attitude, under its scripted
    motion and course, and the cameras stay fixed in the vessel frame.
    """
    scenario = variant(
        ownship={
            "asset": "container",
            "speed_mps": 2.0,
            "roll": {"amplitude_deg": 3.0, "period_s": 8.0},
            "heave": {"amplitude_m": 0.2, "period_s": 6.0},
        },
        outputs={"fps": 2, "duration_s": 3.0},
    )
    built = scene.build(scenario, "eo")
    wind, swell = scenario.wind_waves(), scenario.swell_waves()
    tall, _ = sea.displaced((wind, swell))
    (attitude,) = built.vessel.children
    radius = earth.earth_radius_m(scenario.sea.refraction_k)
    roll_s, heave_s = scenario.outputs.period_s(8.0), scenario.outputs.period_s(6.0)
    rocked, mounted, still = [], [], []

    for frame, t_s in enumerate(scenario.outputs.times_s):
        bpy.context.scene.frame_set(frame)
        east, north, up = built.vessel.matrix_world.translation
        rest = waves.gerstner_rest_m(tall, np.array(east), np.array(north), t_s)
        plane = waves.Waterplane(
            *map(float, rest),
            heading_rad(built.vessel),
            built.vessel["length_m"],
            built.vessel["beam_m"],
        )
        pitch, roll = waves.attitude(wind + swell, plane, t_s)
        rocked.append(abs(pitch) + abs(roll))
        script = 0.2 * math.sin(math.tau * t_s / heave_s)
        still.append(up - earth.sea_z_m(east, north, radius) - script)
        calibration = scene.calibrate(built, scenario.mounts[0], "frame.png")
        mounted.append(np.array(calibration.extrinsics["vessel"]))

        assert built.vessel.rotation_euler[1] == pytest.approx(
            math.radians(3.0) * math.sin(math.tau * t_s / roll_s), abs=1e-6
        )
        assert tuple(attitude.rotation_euler) == pytest.approx(
            (pitch, -roll, 0.0), abs=1e-6
        )
        # Straight up, however the script tilts the hull.
        lift = attitude.matrix_world.translation - built.vessel.matrix_world.translation
        assert tuple(lift) == pytest.approx(
            (0.0, 0.0, waves.heave_m(tall, plane, t_s)), abs=1e-6
        )

    assert max(rocked) > math.radians(1.0), "a calm sea would pass vacuously"
    assert still == pytest.approx([still[0]] * len(still), abs=1e-4)
    assert np.allclose(mounted, mounted[0], atol=1e-6)


def test_the_far_clip_clears_a_hull_down_target() -> None:
    """Hull down from a low camera, its upperworks still up."""
    range_m = 20000.0
    scenario = variant(
        rig={"height_m": 5.0},
        cameras={"eo": {"hfov_deg": 2.0, "width_px": 960, "height_px": 540}},
        objects=[preset_target("multipurpose_freighter", range_m)],
    )
    built = scene.build(scenario, "eo")
    (anchor,) = built.targets["multipurpose_freighter"]
    corners = box_corners(mesh_objects([anchor]))
    top_m = max(c.z for c in corners) - anchor.matrix_world.translation.z
    k = scenario.sea.refraction_k
    assert range_m < earth.horizon_m(5.0, k) + earth.horizon_m(top_m, k)
    camera = built.cameras["bow_eo"]
    eye = camera.matrix_world.translation
    assert camera.data.clip_end > max((c - eye).length for c in corners)


def test_a_band_the_rig_cannot_see_is_an_error() -> None:
    """Otherwise `next()` raises StopIteration, naming nothing."""
    eo_only = Scenario(rigs={"bow": Rig(height_m=12.0, cameras={"eo": Camera(**LENS)})})
    with pytest.raises(ValueError, match="no rig has an ir camera"):
        scene.build(eo_only, "ir")


def test_the_ir_sky_and_air_follow_the_chosen_atmosphere() -> None:
    """Not the default: a scene that dropped `atmosphere` would read lwir's own."""
    # Validated, not `model_copy`: validation gives the air and sea the profile's own.
    data = SCENARIO.model_dump()
    data["sky"] = {k: v for k, v in data["sky"].items() if k != "t_air_k"}
    data["sea"] = {k: v for k, v in data["sea"].items() if k != "t_sea_k"}
    data["sky"]["atmosphere"] = "tropical"
    data["objects"] = []
    tropical = Scenario.model_validate(data)
    assert tropical.sea.t_sea_k == atmosphere.SURFACE_SEA_K["tropical"]
    assert tropical.sky.atmosphere != atmosphere.ATMOSPHERE
    assert tropical.sky.t_air_k == atmosphere.SURFACE_AIR_K["tropical"]
    scene.build(tropical, "ir")
    sky = tropical.sky
    assert baked("sky_radiance") == pytest.approx(
        _hazed_sky_m(sky, _texel_elevations()), rel=1e-5
    )
    ranges = _haze_ranges_m(bpy.context.scene.camera.data.clip_end)
    depth = lwir.path_optical_depth(ranges, sky.visibility_km, "tropical")
    assert baked("haze_depth") == pytest.approx(depth, rel=1e-5)


@pytest.mark.usefixtures("empty_blend")
def test_a_hull_given_one_temperature_keeps_one_skin() -> None:
    """`t_k` overrides the zones: one emission, warmed by `solar_gain_k` in sun."""
    asset = SCENARIO.objects[0].asset
    hulls = Hulls(SCENARIO.sky, surroundings(SCENARIO))
    anchor = _vessel(asset, condition=Condition(t_k=300.0), hulls=hulls)
    hull = mesh_objects([anchor])[0]
    skins = {slot.material for slot in hull.material_slots}
    assert len(skins) == 1
    emission = next(iter(skins)).node_tree.nodes["Emission"]
    assert emission.inputs["Strength"].default_value == pytest.approx(
        lwir.band_radiance(300.0), rel=1e-5
    )


@pytest.mark.usefixtures("empty_blend")
def test_hulls_stopped_at_different_times_keep_their_own_machinery() -> None:
    hulls = Hulls(SCENARIO.sky, surroundings(SCENARIO))
    asset = SCENARIO.objects[0].asset
    skins = []
    for minutes in (0.0, 30.0):
        anchor = _vessel(
            asset, condition=Condition(engine_off_min=minutes), hulls=hulls
        )
        assert anchor["engine_off_min"] == minutes
        hull = mesh_objects([anchor])[0]
        skins.append({s.material.name for s in hull.material_slots})
    running, stopped = skins
    machinery = {n for n in running if n.startswith("thermal_exhaust")}
    assert machinery
    assert not machinery & stopped
    assert "thermal_paint_hull" in running & stopped


@pytest.mark.usefixtures("empty_blend")
def test_a_hull_on_a_crest_keeps_its_waterline() -> None:
    """The waves lift the attitude past the hull's draught."""
    anchor = _vessel("zodiac", condition=Condition(), hulls=Hulls(SCENARIO.sky, None))
    bpy.context.view_layer.update()
    level = waterline_m(anchor)
    (attitude,) = anchor.children
    attitude.location.z = 2.0
    bpy.context.view_layer.update()
    assert waterline_m(anchor) == pytest.approx(level, abs=1e-4)


@pytest.mark.usefixtures("empty_blend")
def test_a_lighter_draught_raises_the_hull_and_its_waterline() -> None:
    """The keel rises by the asset's draught minus the object's, and the labels'
    height and waterline go with it.
    """
    hulls = Hulls(SCENARIO.sky, None)
    level, raised = (
        _vessel("bulk_carrier", condition=Condition(), hulls=hulls, rise_m=rise_m)
        for rise_m in (0.0, 5.0)
    )
    bpy.context.view_layer.update()
    (attitude,) = raised.children
    into_attitude = attitude.matrix_world.inverted()
    keel_m = min(
        (into_attitude @ part.matrix_world @ Vector(corner)).z
        for part in mesh_objects([raised])
        for corner in part.bound_box
    )
    assert keel_m == pytest.approx(5.0 - manifest()["bulk_carrier"].draught_m, abs=1e-3)
    assert raised["height_m"] == pytest.approx(level["height_m"] + 5.0)
    # Both hulls stand at the origin, so their waterlines are comparable as they are.
    # Wall-sided amidships, it keeps its beam; its bulb and stern change its length.
    (beam_m, length_m), (raised_beam_m, raised_length_m) = (
        np.ptp(waterline_m(a), axis=0) for a in (level, raised)
    )
    assert raised_beam_m == pytest.approx(beam_m, abs=0.1)
    assert abs(raised_length_m - length_m) > 1.0, "the waterline stayed where it was"


@pytest.mark.usefixtures("empty_blend")
def test_paint_repaints_only_the_painted_zones() -> None:
    heat = surroundings(SCENARIO)
    white = zone_skin("paint_hull", heat, 0.14)
    black = zone_skin("paint_hull", heat, 0.97)
    assert white != black
    assert zone_skin("glass", heat, 0.14) == zone_skin("glass", heat)


class TestIrBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "ir")

    def test_the_air_takes_the_band_s_optical_depth_by_range(self) -> None:
        """The shader reads this by log range at texel centres."""
        assert "sky_radiance.001" not in bpy.data.images, "the haze reads the world's"
        table = baked("haze_depth")
        ranges = _haze_ranges_m(bpy.context.scene.camera.data.clip_end)
        sky = SCENARIO.sky
        depth = lwir.path_optical_depth(ranges, sky.visibility_km, sky.atmosphere)
        assert table == pytest.approx(depth, rel=1e-5)

    def test_the_sea_does_not_glitter(self) -> None:
        """The emissivity table already takes the whole unresolved slope."""
        assert "glitter_cells" not in bpy.data.materials["sea"].node_tree.nodes

    def test_the_unresolved_slope_spreads_as_the_short_waves_groups(self) -> None:
        """The shader reads this by log footprint at texel centres."""
        table = baked("sea_mss_spread")
        footprints = _footprints_m(len(table))
        speed, fetch = SCENARIO.sea.wind_speed_mps, SCENARIO.sea.fetch_m
        for t in (0, len(table) // 2, len(table) - 1):
            expected = waves.mss_spread(speed, fetch, footprints[t])
            assert table[t] == pytest.approx(expected, rel=1e-5, abs=1e-7)
        nodes = bpy.data.materials["sea"].node_tree.nodes
        assert "sea_group_scale" in upstream(nodes["sea_radiance"].inputs[0])
        assert "sea_rest" in upstream(nodes["sea_group_draw"].inputs[0])

    def test_the_baked_sky_runs_cold_towards_the_zenith(self) -> None:
        """The shader reads this by sin(elevation) at texel centres."""
        curve = baked("sky_radiance")
        expected = _hazed_sky_m(SCENARIO.sky, _texel_elevations())
        assert curve == pytest.approx(expected, rel=1e-5)
        ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
        assert curve[-1] < 0.5 * ambient, "the zenith is much colder than ambient"
        assert np.all(np.diff(curve) <= 1e-6), "radiance falls towards the zenith"

    def test_a_zone_emits_its_emissivity_and_reflects_the_rest(self) -> None:
        """A pure emitter renders a hull one flat value whichever way it is turned;
        glass and bare metal mirror the sky, paint scatters it.
        """
        zoned = {m.name.removeprefix("thermal_") for m in bpy.data.materials}
        assert {"paint_hull", "glass"} <= zoned, (
            "the freighter's labels reach the build"
        )
        for name in ("paint_hull", "glass"):
            tree = bpy.data.materials[f"thermal_{name}"].node_tree
            skin = next(
                n
                for n in tree.nodes
                if n.bl_idname == "ShaderNodeMixShader"
                and not n.inputs["Factor"].is_linked
                and "Emission" in upstream(n.inputs["Shader_001"])
                and n.inputs["Factor"].default_value != thermal.SEA_EMISSIVITY
            )
            zone = thermal.ZONES[name]
            assert skin.inputs["Factor"].default_value == pytest.approx(zone.emissivity)
            scatter = skin.inputs["Shader"].links[0].from_node.bl_idname
            # Blender 4 merged Glossy into Anisotropic and kept Anisotropic's bl_idname.
            assert scatter == (
                "ShaderNodeBsdfAnisotropic"
                if name == "glass"
                else "ShaderNodeBsdfDiffuse"
            )

    def test_a_zone_s_table_is_its_heat_balance(self) -> None:
        """The lookup image holds the zone's band radiance over
        `thermal.temperature_table_k`.
        """
        env = surroundings(SCENARIO).env
        _, _, table = thermal.temperature_table_k(thermal.ZONES["paint_hull"], env)
        expected = np.vectorize(lwir.band_radiance)(table)
        assert baked("thermal_paint_hull") == pytest.approx(expected, rel=1e-5)

    def test_the_sea_reflects_what_it_does_not_emit(self) -> None:
        """Emission alone goes dark toward grazing, where emissivity falls to zero."""
        radiance = _sea_node("sea_radiance")
        assert radiance.operation == "MULTIPLY_ADD"
        assert "sea_reflection" in upstream(radiance.inputs[2])
        emission = radiance.outputs["Value"].links[0].to_node
        assert emission.bl_idname == "ShaderNodeEmission"

    def test_the_sea_samples_only_what_its_tables_lack(self) -> None:
        """The sky is a lookup; one lobe, weighted 1 - eps, traces the rest."""
        tree = bpy.data.materials["sea"].node_tree
        (lobe,) = (n for n in tree.nodes if n.bl_idname.startswith("ShaderNodeBsdf"))
        assert lobe.name == "sea_traced"
        assert lobe.distribution == "BECKMANN"
        weight = lobe.inputs["Color"].links[0].from_node
        assert weight.operation == "SUBTRACT"
        assert weight.inputs[0].default_value == 1.0
        assert weight.inputs[1].links[0].from_node.name == "sea_emissivity"
        assert {"sea_unresolved_wide", "sea_unresolved_narrow"} <= upstream(
            lobe.inputs["Roughness"]
        )

    def test_a_rough_glossy_ray_finds_no_sky(self) -> None:
        """The sea's tables already hold it."""
        tree = bpy.data.worlds["sky"].node_tree
        (path,) = (n for n in tree.nodes if n.bl_idname == "ShaderNodeLightPath")
        assert path.outputs["Is Glossy Ray"].is_linked
        assert path.outputs["Is Singular Ray"].is_linked
        assert path.name in upstream(tree.nodes["Background"].inputs["Color"])
        assert bpy.data.worlds["sky"].cycles.sampling_method == "NONE"

    def test_what_a_rough_glossy_ray_meets_returns_its_excess_over_the_sky(
        self,
    ) -> None:
        """The sky ahead of the ray, taken off outside the haze."""
        nodes = bpy.data.node_groups["haze"].nodes
        less = nodes["haze_less_sky"]
        (path,) = (
            n
            for n in nodes
            if n.bl_idname == "ShaderNodeLightPath"
            and n.outputs["Is Glossy Ray"].is_linked
        )
        assert {path.name, "haze_ahead"} <= upstream(less.inputs["Color"])
        assert "haze_horizon" not in upstream(less.inputs["Color"])
        (output,) = (n for n in nodes if n.bl_idname == "NodeGroupOutput")
        both = output.inputs["Shader"].links[0].from_node
        assert {link.from_node.name for s in both.inputs for link in s.links} == {
            "haze_mix",
            "haze_less_sky",
        }

    def test_the_reflection_and_the_emissivity_share_one_slope(self) -> None:
        """Both come from the pixel's unresolved slope, and must move together."""
        tree = bpy.data.materials["sea"].node_tree
        for name in ("sea_emissivity", "sea_reflection"):
            table = next(
                n
                for n in tree.nodes
                if n.bl_idname == "ShaderNodeTexImage" and n.image.name == name
            )
            lookup = table.inputs["Vector"].links[0].from_node
            assert {
                "sea_unresolved_variance_upwind",
                "sea_unresolved_variance_crosswind",
                "sea_gust_variance",
            } <= upstream(lookup.inputs["Y"]), name

    def test_the_baked_reflection_matches_the_sky_it_reflects(self) -> None:
        table = baked("sea_reflection")
        assert table.shape == (SLOPE_ROWS * VIEW_ROWS, MIRROR_SAMPLES)
        mirror = np.arcsin(_mirror_columns())
        view = np.arcsin(_view_rows())
        sky = _sky_round(SCENARIO.sky, None)
        slope_max = _sea_node("sea_slope_along").inputs[1].default_value
        sigmas = _row_sigmas(slope_max)
        roughest = table[(SLOPE_ROWS - 1) * VIEW_ROWS]  # its grazing view
        grazing = (mirror >= 0) & (mirror < math.radians(5.0))
        flat = lwir.Sky(*sky).reflected(mirror[grazing])
        assert np.all(roughest[grazing] < flat), "facets seen grazing lean to the eye"
        for tile in (0, SLOPE_ROWS - 1):
            for row in (0, VIEW_ROWS - 1):
                scale = table[tile * VIEW_ROWS + row] / lwir.Sky(*sky).reflected(
                    mirror,
                    t_sea_k=SCENARIO.sea.t_sea_k,
                    slope_sigma=sigmas[tile],
                    view_rad=view[row],
                )
                assert scale == pytest.approx(1.0, rel=1e-5)

    def test_the_reflection_between_texels_is_the_sky_it_reflects(self) -> None:
        """Bilinear within a tile, as the image interpolates."""
        table = baked("sea_reflection")
        sky = _sky_round(SCENARIO.sky, None)
        slope_max = _sea_node("sea_slope_along").inputs[1].default_value
        sigmas = _row_sigmas(slope_max)
        rows, cols = VIEW_ROWS, MIRROR_SAMPLES
        rng = np.random.default_rng(0)
        hot = lwir.band_radiance(SCENARIO.sea.t_sea_k)
        for tile in (0, SLOPE_ROWS // 2, SLOPE_ROWS - 1):
            mu = rng.uniform(0.02, 0.98, 40) ** 2  # grazing views too
            sin_mirror = rng.uniform(-0.3, 0.98, 40)  # below the horizon too
            y = np.clip(np.sqrt(mu) * rows - 0.5, 0, rows - 1)
            x = np.clip((sin_mirror + 1) / 2 * cols - 0.5, 0, cols - 1)
            y0, x0 = (
                np.minimum(y.astype(int), rows - 2),
                np.minimum(x.astype(int), cols - 2),
            )
            wy, wx = y - y0, x - x0
            t = table[tile * rows : (tile + 1) * rows]
            between = (1 - wy) * ((1 - wx) * t[y0, x0] + wx * t[y0, x0 + 1]) + wy * (
                (1 - wx) * t[y0 + 1, x0] + wx * t[y0 + 1, x0 + 1]
            )
            exact = np.array(
                [
                    lwir.Sky(*sky).reflected(
                        math.asin(s),
                        t_sea_k=SCENARIO.sea.t_sea_k,
                        slope_sigma=sigmas[tile],
                        view_rad=math.asin(m),
                    )
                    for m, s in zip(mu, sin_mirror, strict=True)
                ]
            )
            theta, eps = lwir.emissivity_curve(
                t_sea_k=SCENARIO.sea.t_sea_k, slope_sigma=sigmas[tile]
            )
            e = np.interp(np.arccos(mu), theta, eps)
            seen, truth = (
                lwir.brightness_temperature(e * hot + (1 - e) * s)
                for s in (between, exact)
            )
            assert seen == pytest.approx(truth, abs=0.25), tile

    def test_emissivity_is_averaged_over_the_unresolved_slopes(self) -> None:
        """Flat Fresnel collapses toward grazing, which is where distant targets sit."""
        table = baked("sea_emissivity")  # cos(theta) along, grazing first
        grazing = math.radians(89.0)
        mu = (np.arange(table.shape[1]) + 0.5) / table.shape[1]
        rough = float(np.interp(math.cos(grazing), mu, table[-1]))

        theta, flat = lwir.emissivity_curve(t_sea_k=SCENARIO.sea.t_sea_k)
        assert rough > 4 * float(np.interp(grazing, theta, flat)), (
            "grazing emissivity is lifted well clear of flat"
        )

    def test_foam_raises_the_emissivity_by_its_cover(self) -> None:
        """The shader reads the excess by cos(theta) at texel centres, scaled by the
        whitecaps' cover.
        """
        table = baked("sea_foam_emissivity")
        excess = lwir.foam_emissivity_excess(np.arccos(blend.texels(len(table))))
        assert table == pytest.approx(excess, abs=1e-6)
        feeding = upstream(_sea_node("sea_emissivity").inputs[0])
        assert any(name.startswith("sea_foam_") for name in feeding), "the grids' foam"

    def test_the_baked_emissivity_matches_the_curve(self) -> None:
        """The shader reads this by cos(theta) at texel centres; the curve is sampled by
        theta. Rows run up the unresolved RMS slope, to twice Cox & Munk's wider axis,
        as a pixel takes it along the view, in the strongest gust.
        """
        table = baked("sea_emissivity")
        rows, width = table.shape
        mu = (np.arange(width) + 0.5) / width
        speed = SCENARIO.sea.wind_speed_mps
        tile = baked("sea_gust")
        gust_max = math.sqrt(2) * np.abs(tile).max() * waves.turbulence_intensity(speed)
        strongest = 2 * max(
            waves.cox_munk_variances(speed)
        ) + gust_max * waves.gust_slope_variance(speed)
        # Per axis: lwir draws each facet's two slopes with this sigma.
        sigma_max = math.sqrt(strongest / 2)
        for row in (0, rows // 2, rows - 1):
            theta, eps = lwir.emissivity_curve(
                t_sea_k=SCENARIO.sea.t_sea_k,
                slope_sigma=(row + 0.5) / rows * sigma_max,
            )
            expected = np.interp(mu, np.cos(theta)[::-1], eps[::-1])
            assert table[row] == pytest.approx(expected, rel=1e-5)
            assert np.all(np.diff(table[row]) >= -1e-6), "rises towards normal"

    def test_radiance_is_not_sent_through_a_film_curve(self) -> None:
        """Blender defaults to AgX."""
        view = bpy.context.scene.view_settings
        assert (view.view_transform, view.look) == ("Standard", "None")
        assert (view.exposure, view.gamma) == (0.0, 1.0)


class TestAnimate:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(variant(outputs={"duration_s": 0.3}))

    @pytest.fixture
    def empty(self) -> bpy.types.Object:
        obj = bpy.data.objects.new("probe", None)
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def test_every_frame_holds_the_value_at_its_time(self, empty) -> None:
        sc = bpy.context.scene
        blend.animate(empty, "location", [0.0, 0.1, 0.2], lambda t: 10 * t, index=0)
        assert (sc.frame_start, sc.frame_end, sc.render.fps, sc.render.fps_base) == (
            0,
            2,
            10,
            1.0,
        )
        for frame, x in enumerate([0.0, 1.0, 2.0]):
            sc.frame_set(frame)
            assert empty.matrix_world.translation.x == pytest.approx(x)

    def test_a_still_is_set_and_not_keyed(self, empty) -> None:
        blend.animate(empty, "location", [0.0], lambda _t: (1.0, 2.0, 3.0))
        assert tuple(empty.location) == (1.0, 2.0, 3.0)
        assert empty.animation_data is None


def test_a_target_underway_runs_along_its_heading_on_the_curved_sea() -> None:
    scenario = load(
        OPEN_SEA,
        [
            "outputs.duration_s = 0.3",
            'objects = [{ asset = "yacht", range_m = 2000.0, bearing_deg = 8.0, '
            "heading_deg = 270.0, speed_mps = 6.0 }]",
        ],
    )
    (spec,) = scenario.objects
    radius = earth.earth_radius_m(scenario.sea.refraction_k)
    anchor = scene.build(scenario).targets[spec.asset][0]
    sc = bpy.context.scene

    start = anchor.matrix_world.translation.copy()
    for frame, t in enumerate(scenario.outputs.times_s[1:], start=1):
        sc.frame_set(frame)
        east, north, up = anchor.matrix_world.translation
        run_east, run_north = east - start.x, north - start.y

        assert math.hypot(run_east, run_north) == pytest.approx(
            spec.speed_mps * t, rel=1e-3
        )
        assert math.degrees(math.atan2(run_east, run_north)) % 360 == pytest.approx(
            spec.heading_deg % 360
        )
        assert up == pytest.approx(earth.sea_z_m(east, north, radius), abs=1e-3)
    sc.frame_set(0)
    assert anchor.matrix_world.translation == start
    assert start.xy.length == pytest.approx(spec.range_m)


def _tall(scenario: Scenario) -> tuple[waves.Wave, ...]:
    return sea.displaced((scenario.wind_waves(), scenario.swell_waves()))[0]


def _ridden_m(
    scenario: Scenario, t_s: float, east_m: float = 0.0, north_m: float = 0.0
) -> float:
    """How high a hull-less ownship rides the water under it at a point."""
    tall = _tall(scenario)
    rest = waves.gerstner_rest_m(tall, np.array(east_m), np.array(north_m), t_s)
    return waves.heave_m(tall, waves.Waterplane(*map(float, rest)), t_s)


class TestOwnshipMotion:
    MOTION = variant(
        outputs={"duration_s": 2.0},
        ownship={
            "roll_deg": 3.0,
            "pitch_deg": -1.0,
            "roll": {"amplitude_deg": 5.0, "period_s": 4.0},
            "pitch": {"amplitude_deg": 2.0, "period_s": 4.0},
            "heave": {"amplitude_m": 0.5, "period_s": 4.0},
        },
    )

    def test_a_quarter_period_in_is_the_peak(self) -> None:
        built = scene.build(self.MOTION)
        anchor, camera = built.vessel, camera_of(SCENARIO.mounts[0])
        mounted = anchor.matrix_world.inverted() @ camera.matrix_world
        bpy.context.scene.frame_set(10)

        pitch, roll, _ = anchor.rotation_euler
        assert math.degrees(pitch) == pytest.approx(-1.0 + 2.0)
        assert math.degrees(roll) == pytest.approx(3.0 + 5.0)
        t_s = self.MOTION.outputs.times_s[10]
        assert anchor.matrix_world.translation.z == pytest.approx(
            0.5 + _ridden_m(self.MOTION, t_s)
        )
        moved = anchor.matrix_world.inverted() @ camera.matrix_world
        assert np.allclose(moved, mounted, atol=1e-5)

    def test_an_ownship_without_motion_only_rides_the_water(self) -> None:
        scenario = variant(outputs={"duration_s": 2.0})
        built = scene.build(scenario)

        action = built.vessel.animation_data.action.name
        (heave,) = (v for name, v in keyed() if name.startswith(action))
        assert heave == pytest.approx(
            [_ridden_m(scenario, t) for t in scenario.outputs.times_s], abs=1e-6
        )


class TestOwnshipUnderway:
    SPEED_MPS = 400.0  # far past a boat's, so the sea's curve is measurable
    STRAIGHT = variant(outputs={"duration_s": 2.0}, ownship={"speed_mps": SPEED_MPS})
    STEERED = variant(
        outputs={"duration_s": 4.0},
        ownship={
            "speed_mps": SPEED_MPS,
            "yaw": {"amplitude_deg": 30.0, "period_s": 4.0},
        },
    )
    TURNING = variant(
        outputs={"duration_s": 4.0},
        ownship={"yaw": {"amplitude_deg": 30.0, "period_s": 4.0}},
    )

    @staticmethod
    def _run(scenario: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The ownship's world position and heading in degrees, and the first
        camera's position, at every frame.
        """
        built = scene.build(scenario)
        camera = camera_of(scenario.mounts[0])
        sc = bpy.context.scene
        hull, bearing, eye = [], [], []
        for frame in range(len(scenario.outputs.times_s)):
            sc.frame_set(frame)
            hull.append(tuple(built.vessel.matrix_world.translation))
            bearing.append(math.degrees(heading_rad(built.vessel)))
            eye.append(tuple(camera.matrix_world.translation))
        return np.array(hull), np.array(bearing), np.array(eye)

    def test_it_sails_its_speed_along_its_bow_and_rides_the_curved_sea(self) -> None:
        hull, bearing, eye = self._run(self.STRAIGHT)

        times = np.array(self.STRAIGHT.outputs.times_s)
        assert hull[:, 0] == pytest.approx(0.0, abs=1e-6)
        assert hull[:, 1] == pytest.approx(self.SPEED_MPS * times, abs=1e-3)
        assert bearing == pytest.approx(0.0, abs=1e-6)
        radius = earth.earth_radius_m(self.STRAIGHT.sea.refraction_k)
        for (east, north, up), t_s in zip(hull, times, strict=True):
            expected = earth.sea_z_m(east, north, radius) + _ridden_m(
                self.STRAIGHT, t_s, east, north
            )
            assert up == pytest.approx(expected, abs=1e-4)
        assert eye[:, :2] - eye[0, :2] == pytest.approx(hull[:, :2] - hull[0, :2])

    def test_a_yawing_ownship_sails_along_the_heading_it_swings_through(self) -> None:
        hull, bearing, _ = self._run(self.STEERED)

        step = np.diff(hull[:, :2], axis=0)
        times = np.array(self.STEERED.outputs.times_s)
        assert np.linalg.norm(step, axis=1) / np.diff(times) == pytest.approx(
            self.SPEED_MPS, rel=2e-3
        )
        # A step runs along the mean of its two ends' headings.
        along = np.degrees(np.arctan2(step[:, 0], step[:, 1]))
        assert along == pytest.approx((bearing[1:] + bearing[:-1]) / 2, abs=0.5)
        quarter = np.argmin(abs(times - 4.0 / 4))
        assert bearing[quarter] == pytest.approx(30.0, abs=1e-3)

    def test_the_track_integrates_the_heading_to_the_closed_form(self) -> None:
        hull, _, _ = self._run(self.STEERED)

        end_s = self.STEERED.outputs.times_s[-1]
        theta = np.linspace(0.0, end_s, 400_001)
        heading = np.radians(30.0) * np.sin(2.0 * np.pi * theta / 4.0)
        east = self.SPEED_MPS * np.trapezoid(np.sin(heading), theta)
        north = self.SPEED_MPS * np.trapezoid(np.cos(heading), theta)
        assert tuple(hull[-1, :2]) == pytest.approx((east, north), rel=1e-3)

    def test_a_ship_without_speed_turns_in_place(self) -> None:
        hull, bearing, _ = self._run(self.TURNING)

        assert hull[:, :2] == pytest.approx(0.0, abs=1e-9)
        assert np.ptp(bearing) == pytest.approx(60.0, abs=1.0)

    def test_the_sea_reaches_a_horizon_past_the_farthest_camera(self) -> None:
        built = scene.build(self.STRAIGHT)
        farthest = max(
            math.hypot(*v.matrix.translation.xy)
            for v in views(self.STRAIGHT, "eo", built.cameras)
        )
        eye_m = max(rig.height_m for rig in self.STRAIGHT.rigs.values())
        horizon = earth.horizon_m(eye_m, self.STRAIGHT.sea.refraction_k)
        assert sea.sea_reach_m(self.STRAIGHT) - farthest >= horizon


def _finest_m(scenario: Scenario, built: scene.Built) -> float:
    return scene._finest_footprint_m(scenario, views(scenario, "eo", built.cameras))


class TestSlicks:
    SLICKS = variant(sea={"slick_cover": 0.3, "wind_from_deg": 60.0})

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(cls.SLICKS, "eo")

    def test_the_tile_s_top_cover_lies_under_a_slick(self) -> None:
        threshold = _sea_node("slick").inputs[1].default_value
        assert threshold == pytest.approx(NormalDist().inv_cdf(0.7))
        spacing_m = waves.WINDROW_SPACING_S * self.SLICKS.sea.wind_speed_mps
        seeded = waves.gaussian_ranks(
            waves.von_karman_field(
                substream(self.SLICKS.seed, "sea/slick"),
                GUST_CELLS,
                GUST_SPACING_M,
                spacing_m,
            )
        )
        unit = seeded / _looked_up_sigma(seeded)
        assert baked("sea_windrows") == pytest.approx(unit, abs=1e-5)

    def test_windrows_run_along_the_wind_at_faller_and_woodcock_s_spacing(
        self,
    ) -> None:
        axes = _sea_node("windrows_axes")
        across, along = (
            np.array(axes.inputs[i].links[0].from_node.inputs[1].default_value[:2])
            for i in ("X", "Y")
        )
        toward = math.radians(self.SLICKS.sea.wind_from_deg + 180.0)
        downwind = np.array([math.sin(toward), math.cos(toward)])
        assert along @ downwind == pytest.approx(np.linalg.norm(along))
        # Sockets hold float32.
        assert across @ downwind == pytest.approx(
            0.0, abs=1e-6 * np.linalg.norm(across)
        )
        assert np.linalg.norm(across) == pytest.approx(1 / GUST_TILE_M)
        assert np.linalg.norm(along) == pytest.approx(
            np.linalg.norm(across) / waves.WINDROW_ASPECT
        )
        carry = np.array(_sea_node("windrows_carry").inputs["Vector"].default_value[:2])
        drift_mps = waves.SLICK_DRIFT * self.SLICKS.sea.wind_speed_mps
        assert carry == pytest.approx([0.0, -drift_mps * np.linalg.norm(along)])

    def test_a_slick_reads_the_sea_the_waves_are_drawn_on(self) -> None:
        assert "sea_rest" in upstream(_sea_node("windrows_axes").inputs["X"])

    def test_a_slick_calms_the_waves_it_damps(self) -> None:
        wind = self.SLICKS.wind_waves()
        sea_ = self.SLICKS.sea
        left = set(
            waves.slick_survivors(
                sea_.wind_speed_mps, wind, math.radians(sea_.wind_from_deg)
            )
        )
        calmed = {
            wave
            for wave, inputs, lane in drawn(self.SLICKS)
            if inputs["Calm"].is_linked and inputs["Damped"].default_value[lane]
        }
        assert calmed == set(wind) - left

    def test_under_a_slick_the_pixel_leaves_what_the_slick_does(self) -> None:
        speed = self.SLICKS.sea.wind_speed_mps
        axis_rad = math.radians(self.SLICKS.sea.wind_from_deg)
        left = waves.slick_survivors(speed, self.SLICKS.wind_waves(), axis_rad)
        for i, axis in enumerate(("upwind", "crosswind")):
            table = baked(f"sea_unresolved_slick_variance_{axis}")
            footprints = _footprints_m(len(table))
            for t in (0, len(table) // 2, len(table) - 1):
                expected = waves.unresolved_slope_variances(
                    speed,
                    axis_rad,
                    left,
                    self.SLICKS.swell_waves(),
                    footprints[t],
                    waves.cox_munk_slick_variances,
                )[i]
                assert table[t] == pytest.approx(expected, rel=1e-5, abs=1e-9), axis
        roughness = _sea_node("sea_specular").inputs["Roughness"]
        assert {"slick", "sea_unresolved_slick_variance_upwind"} <= upstream(roughness)


class TestIrSkin:
    """A warm layer, where the sea's emission is read."""

    SKIN = variant(sea={"wind_speed_mps": 4.0, "warm_layer": {}})

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(cls.SKIN, "ir")

    def test_the_sea_emits_at_its_skin_not_its_bulk(self) -> None:
        sea_ = self.SKIN.sea
        t_skin_k = sea_.t_skin_k(sun_up=True)
        assert t_skin_k > sea_.t_sea_k, "the warm layer outweighs the cool skin"
        assert _sea_node("sea_skin").inputs[0].default_value == pytest.approx(
            lwir.band_radiance(t_skin_k), rel=1e-5
        )
        assert "sea_skin" in upstream(_sea_node("sea_radiance").inputs[1])


def test_a_wake_mixes_the_skin_down_to_the_bulk() -> None:
    """Its turbulence takes the warm layer down: in the wake the sea emits its bulk."""
    under_way = variant(
        sea={"wind_speed_mps": 4.0, "warm_layer": {}},
        objects=[preset_target("multipurpose_freighter", 500.0, speed_mps=6.0)],
    )
    scene.build(under_way, "ir")
    mix = _sea_node("sea_skin_radiance")
    skin = lwir.band_radiance(under_way.sea.t_skin_k(sun_up=True))
    bulk = lwir.band_radiance(under_way.sea.t_sea_k)
    assert mix.inputs[2].default_value == pytest.approx(skin, rel=1e-5)
    assert mix.inputs[1].default_value == pytest.approx(bulk - skin, rel=1e-4)
    assert "wake_position" in upstream(mix.inputs[0])
    assert "sea_skin_radiance" in upstream(_sea_node("sea_skin").inputs[0])


def test_the_sea_s_cool_skin_is_thicker_at_night() -> None:
    night = variant(sky={"sun_elevation_deg": -10.0})
    assert not sea_light(night.sky, "ir").sun_up
    assert night.sea.t_skin_k(sun_up=False) < night.sea.t_skin_k(sun_up=True)


def each_frame() -> Iterator[int]:
    sc = bpy.context.scene
    for frame in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(frame)
        yield frame


class TestSeaEvolves:
    SEQUENCE = variant(outputs={"duration_s": 0.3})

    def test_the_sea_keeps_the_frames_time(self) -> None:
        scene.build(self.SEQUENCE, "eo")
        times = [
            _sea_node("sea_time").outputs["Value"].default_value for _ in each_frame()
        ]
        assert times == pytest.approx(self.SEQUENCE.outputs.times_s)

    def test_a_swell_leaves_the_wind_s_waves_alone(self) -> None:
        swell = variant(BASELINE, sea={"swell": {"height_m": 1.5, "period_s": 11.0}})
        wind = wave_field(SCENARIO)
        assert wave_field(swell)[: len(wind)] == wind

    def test_the_sea_draws_a_swell_in_triples_of_its_own(self) -> None:
        swell = variant(sea={"swell": {"height_m": 1.5, "period_s": 11.0}})
        scene.build(swell, "eo")
        assert [wave for wave, _, _ in drawn(swell)] == shaded(swell)

    def test_the_finest_pixel_is_the_camera_s_height_times_its_angle(self) -> None:
        """Over the crest the tall waves can raise, from the water the rig rides."""
        scenario = load(OPEN_SEA)
        built = scene.build(scenario, "eo")
        (mount,) = scenario.band_mounts("eo")
        pixel_rad = math.radians(mount.camera.hfov_deg) / mount.camera.width_px
        crest_m = sum(w.amplitude_m for w in _tall(scenario))
        lowest = min(_ridden_m(scenario, t) for t in scenario.outputs.times_s)
        assert _finest_m(scenario, built) == pytest.approx(
            (mount.rig.height_m + lowest - crest_m) * pixel_rad
        )
        # The shader's pixel, and so the finest one, scales with the render's size.
        bpy.context.scene.render.resolution_percentage = 50
        assert _finest_m(scenario, built) == pytest.approx(
            2 * (mount.rig.height_m + lowest - crest_m) * pixel_rad
        )

    def test_the_finest_pixel_is_taken_at_the_lowest_heave(self) -> None:
        heave = {"amplitude_m": 2.0, "period_s": 4.0}
        scenario = variant(ownship={"heave": heave}, outputs={"duration_s": 4.0})
        built = scene.build(scenario, "eo")
        (mount,) = scenario.band_mounts("eo")
        pixel_rad = math.radians(mount.camera.hfov_deg) / mount.camera.width_px
        crest_m = sum(w.amplitude_m for w in _tall(scenario))
        lowest = min(
            heave["amplitude_m"] * math.sin(2 * math.pi * t / heave["period_s"])
            + _ridden_m(scenario, t)
            for t in scenario.outputs.times_s
        )
        assert _finest_m(scenario, built) == pytest.approx(
            (mount.rig.height_m + lowest - crest_m) * pixel_rad, rel=1e-5
        )

    def test_only_a_culled_build_leaves_out_what_its_cameras_cannot_resolve(
        self,
    ) -> None:
        """Probes and a hand-moved camera need every wave in a default build."""
        scenario = variant(rig={"height_m": 100.0})
        field = wave_field(scenario)
        scene.build(scenario, "eo")
        assert [w for w, _, _ in drawn(scenario)] == shaded(scenario)
        built = scene.build(scenario, "eo", cull=True)
        resolved = waves.resolvable(field, _finest_m(scenario, built))
        kept = [w for w, _, _ in drawn(scenario, resolved)]
        assert kept == shaded(scenario, resolved) != shaded(scenario)

    def test_a_still_leaves_the_sea_unkeyed(self) -> None:
        scene.build(load(OPEN_SEA), "eo")
        assert _sea_node("sea_time").outputs["Value"].default_value == 0.0
        # Drivers make animation data; a still has no action.
        assert bpy.data.materials["sea"].node_tree.animation_data.action is None


class TestLoop:
    LOOP = variant(
        outputs={"duration_s": 30.0, "fps": 1, "loop": True}, sea={"slick_cover": 0.1}
    )

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(cls.LOOP, "eo")

    def test_a_loop_keys_the_time_round_a_circle(self) -> None:
        span_s = self.LOOP.outputs.span_s
        for frame in each_frame():
            turn = 2 * math.pi * self.LOOP.outputs.times_s[frame] / span_s
            cos = _sea_node("sea_cos").outputs["Value"].default_value
            sin = _sea_node("sea_sin").outputs["Value"].default_value
            assert (cos, sin) == pytest.approx(
                (math.cos(turn), math.sin(turn)), abs=1e-6
            )

    def test_every_wave_in_a_loop_turns_a_whole_number_of_times(self) -> None:
        span_s = self.LOOP.outputs.span_s
        for wave in wave_field(self.LOOP):
            turns = wave.omega_rad_s * span_s / (2 * math.pi)
            assert turns == pytest.approx(round(turns), abs=1e-9)

    def test_a_loop_swaps_gust_layers_while_each_weighs_nothing(self) -> None:
        span_s = self.LOOP.outputs.span_s
        for _ in each_frame():
            weight, carried_s = (
                [_evaluated(_sea_node(f"{name}_{i}").outputs["Value"]) for i in (0, 1)]
                for name in ("gust_weight", "gust_carried")
            )
            assert sum(weight) == pytest.approx(1.0, abs=1e-6)
            for w, c in zip(weight, carried_s, strict=True):
                # A layer is carried back to the start where its weight reaches 0.
                assert w == pytest.approx(2 * min(c, span_s - c) / span_s, abs=1e-6)

    def test_a_loop_s_windrows_and_lace_hold_still_with_its_foam(self) -> None:
        nodes = bpy.data.materials["sea"].node_tree.nodes
        for name in ("windrows", "whitecap_lace"):
            carry = _sea_node(f"{name}_carry").inputs["Vector"].default_value
            assert tuple(carry[:2]) == (0.0, 0.0)
            # Nothing moves, so nothing crossfades.
            assert f"{name}_weight_0" not in nodes


class TestOrbitingYachts:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> scene.Built:
        yachts = target("yacht", 200.0, -90.0, orbit={"period_s": 80.0, "count": 2})
        return scene.build(
            variant(DRIFTING, objects=[yachts], outputs={"duration_s": 40.0, "fps": 1})
        )

    def test_a_glb_asset_leaves_its_lights_behind(self) -> None:
        assert not [o for o in bpy.data.objects if o.type == "LIGHT"]

    def test_orbiting_hulls_share_a_lap_clockwise_bow_first(
        self, built: scene.Built
    ) -> None:
        first, second = built.targets["yacht"]
        sc = bpy.context.scene

        for frame in (0, 10, 39):
            sc.frame_set(frame)
            for hull, start_deg in ((first, -90.0), (second, 90.0)):
                bearing_deg = start_deg + 360.0 * frame / 80.0
                bearing = math.radians(bearing_deg)
                east, north, _ = hull.matrix_world.translation
                assert (east, north) == pytest.approx(
                    (200.0 * math.sin(bearing), 200.0 * math.cos(bearing)), abs=1e-3
                )
                assert hull.rotation_euler.z == pytest.approx(
                    blend.yaw(bearing_deg + 90)
                )


def keyed() -> Iterator[tuple[str, np.ndarray]]:
    """Every keyed channel in the scene, its values frame by frame."""
    for action in bpy.data.actions:
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    for curve in bag.fcurves:
                        keys = np.empty(2 * len(curve.keyframe_points))
                        curve.keyframe_points.foreach_get("co", keys)
                        path = f"{curve.data_path}[{curve.array_index}]"
                        yield f"{action.name} {path}", keys[1::2]


class TestDrifting:
    # Every eighth of the drift lands on a frame.
    SCENARIO = variant(DRIFTING, outputs={"fps": 4})

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> scene.Built:
        return scene.build(cls.SCENARIO)

    def test_a_drifting_hull_traces_a_figure_eight_about_its_pose(
        self, built: scene.Built
    ) -> None:
        scenario = self.SCENARIO
        (spec,) = scenario.objects
        assert spec.drift is not None
        anchor = built.targets[spec.asset][0]
        sc = bpy.context.scene
        sc.frame_set(0)
        period_s = scenario.outputs.period_s(spec.drift.period_s)
        heading = math.radians(spec.heading_deg)
        ahead = Vector((math.sin(heading), math.cos(heading)))
        starboard = Vector((math.cos(heading), -math.sin(heading)))
        radius = earth.earth_radius_m(scenario.sea.refraction_k)
        pose = anchor.matrix_world.translation.xy.copy()
        s = math.sqrt(0.5)
        eighths = [(0, 0), (s, 1), (1, 0), (s, -1), (0, 0), (-s, 1), (-1, 0), (-s, -1)]

        for eighth, (across, along) in enumerate(eighths):
            sc.frame_set(round(eighth * period_s / 8 * scenario.outputs.fps))
            east, north, up = anchor.matrix_world.translation
            offset = Vector((east, north)) - pose

            assert offset.dot(starboard) == pytest.approx(
                across * spec.drift.sway_m, abs=1e-2
            )
            assert offset.dot(ahead) == pytest.approx(
                along * spec.drift.surge_m, abs=1e-2
            )
            assert up == pytest.approx(earth.sea_z_m(east, north, radius), abs=1e-3)
            assert anchor.rotation_euler.z == pytest.approx(blend.yaw(spec.heading_deg))
        assert pose.length == pytest.approx(spec.range_m)

    def test_a_hull_rides_the_sea_it_sits_on(self, built: scene.Built) -> None:
        scenario = self.SCENARIO
        (anchor,) = built.targets[scenario.objects[0].asset]
        (attitude,) = anchor.children
        corners = box_corners(mesh_objects([anchor]), attitude.matrix_world.inverted())
        length = max(c.y for c in corners) - min(c.y for c in corners)
        beam = max(c.x for c in corners) - min(c.x for c in corners)
        sc = bpy.context.scene
        for frame in (0, 7, 31):
            sc.frame_set(frame)
            east, north, _ = anchor.matrix_world.translation
            bow = anchor.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))
            heading = math.atan2(bow.x, bow.y)
            assert math.degrees(heading) % 360 == pytest.approx(
                scenario.objects[0].heading_deg % 360, abs=1e-4
            )
            t_s = scenario.outputs.times_s[frame]
            # Only the tall waves move the water the hull sits on.
            assert len(_tall(scenario)) < len(wave_field(scenario))
            rest = waves.gerstner_rest_m(
                _tall(scenario), np.array(east), np.array(north), t_s
            )
            hull = waves.Waterplane(*map(float, rest), heading, length, beam)
            pitch, roll = waves.attitude(wave_field(scenario), hull, t_s)
            assert tuple(attitude.rotation_euler[:2]) == pytest.approx(
                (pitch, -roll), abs=1e-5
            )
            assert attitude.location.z == pytest.approx(
                waves.heave_m(_tall(scenario), hull, t_s), abs=1e-5
            )


def test_a_loop_keyed_one_frame_past_its_end_is_back_at_its_first() -> None:
    """Keyed at the span, every channel is back at its frame-0 value."""
    outputs = TestDrifting.SCENARIO.outputs
    times_s, span_s = outputs.times_s, outputs.span_s
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Outputs, "times_s", property(lambda _: [*times_s, span_s]))
        # span_s counts times_s; hold it at the real clip's.
        patch.setattr(Outputs, "span_s", property(lambda _: span_s))
        scene.build(TestDrifting.SCENARIO)

    channels = dict(keyed())

    assert len(channels) == 13, (
        "the target's xyz, pitch, roll, heave's xyz; the ownship's pitch, roll, "
        "heave; the sea's cos, sin"
    )
    # A few ulps: F-curves are float32.
    unclosed = [
        name
        for name, values in channels.items()
        if abs(values[-1] - values[0]) > 4 * np.spacing(np.float32(abs(values).max()))
    ]
    assert unclosed == []


class TestWakes:
    def _built(self, spec: Object | dict[str, Any]) -> bpy.types.NodeTree:
        scene.build(variant(objects=[spec]), "eo")
        return bpy.data.materials["sea"].node_tree

    def test_a_hull_under_way_leaves_foam_and_arms_on_the_drawn_sea(self) -> None:
        tree = self._built(target("yacht", 300.0, speed_mps=8.0))

        assert "sea_foam" in bpy.data.images
        assert "wake_normal" in tree.nodes
        # Read off the flat sea, the foam slides off the crests.
        assert "sea_rest" in upstream(tree.nodes["wake_position"].inputs["Vector"])
        # The wake and the whitecaps threshold through one quantile table.
        assert "whitecap_albedo" in tree.nodes
        assert [i.name for i in bpy.data.images if "normal_ppf" in i.name] == [
            "normal_ppf"
        ]

    def test_a_slow_ship_s_arms_go_unseen_and_unbuilt(self) -> None:
        tree = self._built(
            preset_target("multipurpose_freighter", 2000.0, speed_mps=5.0)
        )
        assert "sea_foam" in bpy.data.images
        assert "wake_normal" not in tree.nodes

    def test_a_group_under_way_leaves_a_wake(self) -> None:
        ring = {
            "asset": "yacht",
            "count": 2,
            "range_m": 300.0,
            "bearing_deg": [-5.0, 5.0],
            "heading_deg": [90.0, 90.0],
            "speed_mps": 8.0,
        }
        scene.build(variant(targets=ring), "eo")
        assert "sea_foam" in bpy.data.images


def _pixel_node() -> bpy.types.ShaderNode:
    tree = bpy.data.materials["sea"].node_tree
    return next(
        n
        for n in tree.nodes
        if n.bl_idname == "ShaderNodeMath"
        and n.inputs[0].links
        and n.inputs[0].links[0].from_socket.name == "View Distance"
    )


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(load(OPEN_SEA), id="open-sea"),
        # A pixel under 1e-4 rad: a zero footprint leaves the sea a mirror.
        pytest.param(
            variant(
                cameras={"eo": {"hfov_deg": 1.5, "width_px": 640, "height_px": 360}}
            ),
            id="long-lens",
        ),
    ],
)
def test_the_sea_draws_for_the_pixel_the_render_takes(scenario: Scenario) -> None:
    scene.build(scenario, "eo")
    mount = scenario.band_mounts("eo")[0]
    pixel_rad = math.radians(mount.camera.hfov_deg) / mount.camera.width_px
    driver = next(
        c.driver
        for c in bpy.data.materials["sea"].node_tree.animation_data.drivers
        if c.driver.expression == "angle / (width * percent / 100)"
    )
    assert driver.is_simple_expression
    assert _pixel_node().inputs[1].default_value == pytest.approx(pixel_rad)
    bpy.context.scene.render.resolution_percentage = 50
    bpy.context.view_layer.update()
    assert _pixel_node().inputs[1].default_value == pytest.approx(2 * pixel_rad)


def test_the_glitter_and_haze_drivers_run_without_python() -> None:
    scene.build(load(OPEN_SEA), "eo")
    drivers = [
        curve.driver
        for tree in (
            bpy.data.materials["sea"].node_tree,
            bpy.data.node_groups["haze"],
            bpy.data.worlds["sky"].node_tree,
        )
        for curve in tree.animation_data.drivers
    ]
    assert {"1 / samples", "sky"} <= {d.expression for d in drivers}
    assert all(d.is_valid and d.is_simple_expression for d in drivers)


class TestPhotographedSky:
    SKY = "kloofendal_48d_partly_cloudy"

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls, tmp_path_factory: pytest.TempPathFactory) -> None:
        """A tiny photo stands in for the download."""
        with white_photo(tmp_path_factory.mktemp("sky")):
            scene.build(variant(sky={"hdri": cls.SKY}), "eo")

    def _photos(self, tree: bpy.types.NodeTree) -> list[bpy.types.Node]:
        return [n for n in tree.nodes if n.bl_idname == "ShaderNodeTexEnvironment"]

    def test_the_photo_is_the_world(self) -> None:
        tree = bpy.data.worlds["sky"].node_tree
        background = tree.nodes["Background"].inputs["Color"].links[0].from_node
        assert background.bl_idname == "ShaderNodeTexEnvironment"
        assert not [n for n in tree.nodes if n.bl_idname == "ShaderNodeTexSky"]

    def test_the_haze_takes_the_photo_s_horizon(self) -> None:
        (world,) = self._photos(bpy.data.worlds["sky"].node_tree)
        (haze,) = self._photos(bpy.data.node_groups["haze"])
        assert haze.image == world.image


HIDDEN_SUN = {
    "hdri": "overcast_soil",
    "sun_diffuse_fraction": 0.5,
    "sun_elevation_deg": 30,
}


class TestHiddenSun:
    """A white photo stands in for the download."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls, tmp_path_factory: pytest.TempPathFactory) -> None:
        with white_photo(tmp_path_factory.mktemp("sky")):
            scene.build(variant(sky=HIDDEN_SUN), "eo")

    def test_its_beam_is_the_photo_s_diffuse_light_over_k_less_one(self) -> None:
        diffuse = skies.library()["overcast_soil"].diffuse_irradiance
        assert diffuse is not None
        # (1 - k) / k is 1 and sin 30 deg is 1/2.
        assert bpy.data.lights["sun"].energy == pytest.approx(2 * diffuse)

    def test_it_shines_from_the_scenario_s_sun(self) -> None:
        sky = variant(sky=HIDDEN_SUN).sky
        towards = bpy.data.objects["sun"].matrix_world.to_3x3() @ Vector((0, 0, 1))
        assert tuple(towards) == pytest.approx(sky.sun_vector, abs=1e-6)

    def test_a_camera_sees_it(self) -> None:
        assert bpy.data.objects["sun"].visible_camera


class TestPhotographedThermalSky:
    """A white photo, all cloud, stands in for the download."""

    CLOUDY = variant(sky={"hdri": "overcast_soil"})

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls, tmp_path_factory: pytest.TempPathFactory) -> None:
        with white_photo(tmp_path_factory.mktemp("sky"), 128, 64) as patch:
            patch.setattr("seascape.blender.sky.THERMAL_SKY_PX", 64)
            scene.build(cls.CLOUDY, "ir")

    def test_the_world_adds_the_photo_s_share_of_a_whole_cloud_to_the_clear_sky(
        self,
    ) -> None:
        tree = bpy.data.worlds["sky"].node_tree
        (mix,) = (
            tree.nodes[n]
            for n in upstream(tree.nodes["Background"].inputs["Color"])
            if getattr(tree.nodes[n], "operation", None) == "MULTIPLY_ADD"
        )
        images = [i.links[0].from_node.image for i in mix.inputs]
        assert [i.name for i in images] == [
            "cloud_share",
            "cloud_excess",
            "sky_radiance",
        ]
        # The share keeps the photo's resolution; the radiance is solved coarser.
        assert tuple(images[0].size) == (128, 64)
        assert tuple(bpy.data.images["cloudy_sky_radiance"].size) == (64, 32)

    def test_the_world_s_whole_cloud_is_lwir_s(self) -> None:
        top = baked("sky_radiance")[-1] + baked("cloud_excess")[-1]
        expected = _hazed_sky_m(self.CLOUDY.sky, _texel_elevations()[-1], cloud=1.0)
        assert top == pytest.approx(float(expected), rel=1e-5)

    def test_the_haze_takes_its_airlight_from_the_clouds(self) -> None:
        nodes = bpy.data.node_groups["haze"].nodes
        airlight = upstream(nodes["haze_airlight"].inputs["Color"])
        assert any(
            nodes[name].bl_idname == "ShaderNodeTexEnvironment"
            and nodes[name].image.name == "cloud_share"
            for name in airlight
        )

    def test_the_hull_balances_against_the_photo_s_clouds(self) -> None:
        heat = surroundings(self.CLOUDY)
        assert heat.env.cloud_fraction == pytest.approx(1.0)

    def test_a_hidden_sun_s_beam_reaches_the_hull_through_the_clouds(self) -> None:
        env = surroundings(variant(sky=HIDDEN_SUN)).env
        zone = thermal.ZONES["paint_hull"]
        sunlit, shaded = (thermal.lit_temperature_k(zone, env, c, 0.0) for c in (1, 0))
        assert sunlit > shaded

    def test_overhead_reads_a_cloud(self) -> None:
        top = baked("cloudy_sky_radiance")[-1]
        expected = _hazed_sky_m(self.CLOUDY.sky, np.pi / 2, cloud=1.0)
        assert top == pytest.approx(float(expected), rel=1e-3)

    def test_the_sea_reflects_the_clouds(self) -> None:
        """A sea under overcast reflecting a clear sky would read cold."""
        table = baked("sea_reflection")
        mirror = np.arcsin(_mirror_columns())
        sky = self.CLOUDY.sky
        # Off the horizon, where a clear sky runs cold.
        high = mirror[mirror > math.radians(30.0)]
        clear = lwir.sky_radiance(high, sky.t_air_k, sky.atmosphere)
        assert np.all(table[0][mirror > math.radians(30.0)] > clear)

    def test_a_sky_alike_at_every_bearing_reflects_as_its_average(self) -> None:
        assert baked("sea_clouds") == pytest.approx(1.0, rel=1e-6)

    def test_the_photo_is_not_kept(self) -> None:
        assert "white.hdr" not in bpy.data.images


def test_a_cloud_brightens_the_sea_under_its_bearing() -> None:
    """The baked share at a texel centre is `lwir.Sky.reflected` of the bearing's sky
    over the round sky's.
    """
    elevation = np.radians(np.arange(-89.0, 90.0, 2.0))
    clear = lwir.sky_radiance(elevation)
    half = CLOUD_BEARINGS // 2
    radiance = np.tile(clear[:, None], (1, CLOUD_BEARINGS))
    radiance[:, :half] = lwir.cloudy_sky_radiance(elevation, 1.0, 500.0)[:, None]
    slope_max = 0.3
    image = _cloud_image(
        atmosphere.T_SEA_K, sea.Clouds(elevation, radiance, 0.0), slope_max
    )
    table = blend.pixels(image)[..., 0]
    # The roughest tile, its last view row, a mirror row above the horizon.
    tile, view, k = (
        CLOUD_SLOPES - 1,
        CLOUD_VIEWS - 1,
        3 * CLOUD_MIRRORS // 4,
    )
    row = table[(tile * CLOUD_VIEWS + view) * CLOUD_MIRRORS + k]
    root = (k + 0.5) / CLOUD_MIRRORS * 2.0 - 1.0

    def reflect(sky: np.ndarray) -> np.ndarray:
        return lwir.Sky(elevation, sky).reflected(
            math.asin(root**2),
            slope_sigma=_row_sigmas(slope_max, CLOUD_SLOPES)[tile],
            view_rad=math.asin(_view_rows(CLOUD_VIEWS)[view]),
        )

    expected = reflect(radiance) / reflect(radiance.mean(axis=1))
    assert row == pytest.approx(expected, rel=1e-5)
    assert np.all(row[:half] > 1.0), "under the cloud"
    assert np.all(row[half:] < 1.0), "under the clear sky"


@pytest.mark.parametrize(
    "overrides", [{"hdri": "overcast_soil"}, {"sun_elevation_deg": -10.0}]
)
def test_a_sky_without_a_disc_or_with_a_set_sun_grades_no_hull_sunlit(
    overrides: dict[str, Any],
) -> None:
    tree = bpy.data.materials.new("hull").node_tree
    emission = _sunlit_emission(tree, 290.0, variant(sky=overrides).sky).node
    assert emission.bl_idname == "ShaderNodeEmission"
    assert emission.inputs["Strength"].default_value == pytest.approx(
        lwir.band_radiance(290.0)
    )


def test_a_heading_from_the_line_of_sight_turns_from_the_bearing() -> None:
    yacht = target(
        "yacht", 900.0, 30.0, heading_deg=180.0, heading_from="line_of_sight"
    )

    built = scene.build(variant(objects=[yacht]), "eo")

    (anchor,) = built.targets["yacht"]
    assert math.degrees(heading_rad(anchor)) % 360.0 == pytest.approx(210.0)


@pytest.mark.parametrize(
    ("overrides", "meeting"),
    [
        pytest.param(
            {"objects": [target("yacht", 900.0), target("yacht", 900.0)]},
            "yacht",
            id="hulls",
        ),
        pytest.param(
            {"objects": [target("yacht", 900.0), target("lateral_mark", 902.0)]},
            "lateral_mark",
            id="a buoy has a footprint too",
        ),
        pytest.param(
            {"objects": [target("yacht", 30.0)], "ownship": {"asset": "cargo_ship"}},
            "ownship",
            id="target inside the ownship",
        ),
    ],
)
def test_hulls_inside_each_other_are_refused(
    overrides: dict[str, Any], meeting: str
) -> None:
    with pytest.raises(scene.OverlapError, match=meeting):
        scene.build(variant(**overrides), "eo")


EAST_M, NORTH_M = 100.0, 1000.0
CROSSING = [
    target("yacht", 1000.0, heading_deg=90.0, speed_mps=20.0),
    target(
        "yacht",
        math.hypot(EAST_M, NORTH_M),
        math.degrees(math.atan2(EAST_M, NORTH_M)),
    ),
]


def test_a_hull_that_sails_into_another_is_refused() -> None:
    """Apart at the first frame: only the clip's later frames catch it."""
    clip = variant(objects=CROSSING, outputs={"duration_s": 5.0, "fps": 1})

    with pytest.raises(scene.OverlapError):
        scene.build(clip, "eo")


def test_a_still_of_two_hulls_apart_builds() -> None:
    scene.build(variant(objects=CROSSING), "eo")


def test_a_built_hull_measures_its_manifest_length() -> None:
    built = scene.build(variant(objects=[target("yacht", 900.0)]), "eo")

    (anchor,) = built.targets["yacht"]
    hull = manifest()["yacht"]
    assert isinstance(hull, Hull)
    assert anchor["length_m"] == pytest.approx(hull.length_m, rel=1e-6)


def test_a_built_turbine_measures_its_manifest_height() -> None:
    farm = {"asset": "turbine_15mw_floating", "range_m": 900.0, "bearing_deg": 0.0}
    scenario = variant(farms=[farm])
    built = scene.build(scenario, "eo")

    (anchor,) = built.targets["turbine_15mw_floating"]
    turbine = manifest()["turbine_15mw_floating"]
    assert isinstance(turbine, Turbine)
    assert anchor["height_m"] + turbine.draught_m == pytest.approx(turbine.height_m)
    # Past the tips at top dead centre, wherever the rotor's spin has them.
    k = scenario.sea.refraction_k
    eye_m = max(rig.height_m for rig in scenario.rigs.values())
    assert scene._sightline_m(scenario, [anchor]) == pytest.approx(
        earth.horizon_m(eye_m, k) + earth.horizon_m(anchor["height_m"], k)
    )


def test_a_built_buoy_measures_its_manifest_height() -> None:
    built = scene.build(variant(objects=[target("lateral_mark", 900.0)]), "eo")

    (anchor,) = built.targets["lateral_mark"]
    buoy = manifest()["lateral_mark"]
    assert isinstance(buoy, Buoy)
    assert anchor["height_m"] + buoy.draught_m == pytest.approx(buoy.height_m, rel=1e-6)


def test_a_turbine_faces_the_wind_and_turns_about_its_hub_in_either_band() -> None:
    windy = variant(
        farms=[{"asset": "turbine_15mw_monopile", "range_m": 2000.0, "bearing_deg": 0}],
        sea={"wind_from_deg": 60.0},
    )
    hubs = []
    for band in ("eo", "ir"):
        built = scene.build(windy, band)
        (anchor,) = built.targets["turbine_15mw_monopile"]
        (rotor,) = (o for o in anchor.children_recursive if o.get("rotor"))
        shaft = rotor.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))

        assert math.degrees(heading_rad(anchor)) == pytest.approx(60.0)
        assert math.degrees(math.atan2(shaft.x, shaft.y)) == pytest.approx(60.0)
        assert [o for o in anchor.children_recursive if o.get("rotor")] == [rotor]
        hubs.append(rotor.matrix_world.translation.copy())
    assert (hubs[0] - hubs[1]).length == pytest.approx(0.0, abs=1e-3)
    # Its frame is the hub, 150 m up: no face of it is under the waterline.
    assert all(
        node.type != "TEX_COORD"
        for slot in rotor.material_slots
        for node in slot.material.node_tree.nodes
    )


def _rotor_angle(anchor: bpy.types.Object) -> float:
    """Its turn about the shaft, clockwise seen from upwind."""
    (rotor,) = (o for o in anchor.children_recursive if o.get("rotor"))
    return -rotor.rotation_euler.to_matrix().to_euler("YXZ").y


def test_a_rotor_turns_at_its_wind_s_speed_from_an_angle_of_its_own() -> None:
    fps = 4
    farm = {"asset": "turbine_15mw_monopile", "range_m": 2000.0, "bearing_deg": 0.0}
    clip = variant(
        farms=[farm | {"rows": 1, "columns": 2}],
        outputs={"duration_s": 1.0, "fps": fps},
    )
    rpm = clip.farms[0].rpm(clip.sea)

    built = scene.build(clip, "eo")

    sc = bpy.context.scene
    first, second = built.targets["turbine_15mw_monopile"]
    sc.frame_set(0)
    starts = [_rotor_angle(first), _rotor_angle(second)]
    sc.frame_set(1)
    turned = _rotor_angle(first) - starts[0]
    assert math.remainder(turned, 2 * math.pi) == pytest.approx(
        rpm * math.pi / 30 / fps
    )
    symmetry_rad = clip.farms[0].turbine.rotor.symmetry_rad
    assert math.remainder(starts[0] - starts[1], symmetry_rad) != pytest.approx(0)


def test_a_looping_rotor_s_last_frame_is_one_step_short_of_its_symmetry_turn() -> None:
    fps = 4
    farm = {"asset": "turbine_15mw_monopile", "range_m": 2000.0, "bearing_deg": 0.0}
    clip = variant(farms=[farm], outputs={"duration_s": 4.0, "fps": fps, "loop": True})
    rotor = clip.farms[0].turbine.rotor
    symmetry_s = clip.outputs.period_s(
        rotor.symmetry_period_s(clip.farms[0].rpm(clip.sea))
    )

    built = scene.build(clip, "eo")

    sc = bpy.context.scene
    (anchor,) = built.targets["turbine_15mw_monopile"]
    sc.frame_set(0)
    start = _rotor_angle(anchor)
    sc.frame_set(sc.frame_end)
    seam = math.remainder(start - _rotor_angle(anchor), rotor.symmetry_rad)
    assert seam == pytest.approx(rotor.symmetry_rad / (symmetry_s * fps))


def test_the_ownship_is_indexed_after_every_target() -> None:
    built = scene.build(
        variant(
            objects=[target("yacht", 900.0)],
            ownship={"asset": "panamax_container_ship"},
        ),
        "eo",
    )

    (anchor,) = built.targets["yacht"]
    hull = mesh_objects([built.vessel])
    assert hull, "the ownship has no mesh: nothing below ran"
    assert {p.pass_index for p in hull} == {anchor.pass_index + 1}


def test_rigs_too_far_apart_for_a_grid_leave_it_to_a_coarser_one() -> None:
    down = Vector((0.0, 1.0, -0.2)).normalized()
    rays = [(Vector((0.0, y, 20.0)), down, 1e-3) for y in (-150.0, 150.0)]
    assert memory._foam_box(rays, memory.FOAM_NEAR_CELL_M) is None
    cell_m = memory.FOAM_NEAR_CELL_M * memory.FOAM_LEVEL_RATIO
    fit = memory._foam_box(rays, cell_m)
    assert fit is not None
    _, _, width, height, _ = fit
    assert max(width, height) <= memory.FOAM_MAX_CELLS * cell_m


@pytest.mark.parametrize("speed", [3.0, 21.0])
def test_the_shortest_breaker_a_foam_grid_draws_cannot_follow_a_gust(
    speed: float,
) -> None:
    """Why whitecaps break at the mean wind: Plant 1982's growth, 0.04 (u* / c)^2
    omega, takes longer than a gust, its integral scale over the wind, lasts.
    """
    shortest_m = waves.FOAM_CELLS_PER_WAVE * memory.FOAM_NEAR_CELL_M
    k = 2 * math.pi / shortest_m
    omega = math.sqrt(earth.GRAVITY_MS2 * k)
    growth_s = 1 / (
        0.04 * (waves.friction_velocity_mps(speed) * k / omega) ** 2 * omega
    )
    assert growth_s > waves.GUST_LENGTH_M / speed


def test_a_combine_xyz_refuses_a_fourth_component() -> None:
    tree = bpy.data.node_groups.new("combine_probe", "ShaderNodeTree")
    try:
        with pytest.raises(ValueError, match="3 components"):
            blend.combine(tree, 1.0, 2.0, 3.0, 4.0)
    finally:
        bpy.data.node_groups.remove(tree)


FOG = {"visibility_km": 0.5}


def test_in_fog_the_eo_sky_is_its_horizon_s_grey_and_the_sun_dims() -> None:
    fog_ = variant(sky=FOG)
    scene.build(fog_, "eo")
    world = bpy.data.worlds["sky"]
    tree = world.node_tree
    feeding = upstream(tree.nodes["Background"].inputs["Color"])
    assert any(tree.nodes[n].bl_idname == "ShaderNodeRGBToBW" for n in feeding)
    kept = tree.nodes["fog_sun"].outputs["Value"]
    assert kept.default_value == pytest.approx(fog_.sky.sun_transmittance, rel=1e-5)
    world["sun_elevation"] = math.radians(60.0)
    world.update_tag()
    bpy.context.view_layer.update()
    moved = aerosol.sun_transmittance(math.radians(60.0), 0.5)
    assert kept.default_value == pytest.approx(moved, rel=1e-5)
    haze = bpy.data.node_groups["haze"].nodes
    airlight = upstream(haze["haze_airlight"].inputs["Color"])
    assert any(haze[n].bl_idname == "ShaderNodeRGBToBW" for n in airlight)
    assert sea_light(fog_.sky, "eo").sun == fog_.sky.sun_vector


def test_in_mist_the_eo_sky_is_part_way_to_fog_s_grey() -> None:
    mist = variant(sky={"visibility_km": 1.2})
    scene.build(mist, "eo")
    tree = bpy.data.worlds["sky"].node_tree
    feeding = upstream(tree.nodes["Background"].inputs["Color"])
    nodes = [tree.nodes[n] for n in feeding]
    weights = [
        n.inputs["Scale"].default_value
        for n in nodes
        if n.bl_idname == "ShaderNodeVectorMath" and n.operation == "SCALE"
    ]
    assert any(w == pytest.approx(mist.sky.fog_weight) for w in weights)
    assert any(tree.nodes[n].bl_idname == "ShaderNodeRGBToBW" for n in feeding)


def test_in_fog_the_ir_sky_glows_through_it_and_the_hull_s_sun_dims() -> None:
    fog_ = variant(sky=FOG)
    scene.build(fog_, "ir")
    sky = fog_.sky
    expected = _hazed_sky_m(sky, _texel_elevations())
    assert baked("sky_radiance") == pytest.approx(expected, rel=1e-5)
    assert _sky_round(sky, None)[1] == pytest.approx(expected)
    env = surroundings(fog_).env
    assert env.fog_cover == pytest.approx(aerosol.cover(0.5, sky.atmosphere))
    assert env.beam_transmittance == pytest.approx(sky.sun_transmittance)
    assert env.fog_sunlight_transmittance == pytest.approx(
        aerosol.diffuse_transmittance(0.5)
    )
    assert 0.0 < sky.sun_transmittance < 1.0


def test_in_fog_a_photographed_ir_sky_glows_through_it(tmp_path: Path) -> None:
    fogged = variant(sky={"hdri": "overcast_soil", **FOG})
    with white_photo(tmp_path):
        scene.build(fogged, "ir")
    top = baked("cloudy_sky_radiance")[-1]
    expected = _hazed_sky_m(fogged.sky, np.pi / 2, cloud=1.0)
    assert top == pytest.approx(float(expected), rel=1e-3)


@pytest.mark.usefixtures("empty_blend")
def test_the_hull_finds_the_photo_s_clouds_before_the_world_is_built() -> None:
    heat = surroundings(variant(sky={"hdri": "overcast_soil"}))
    assert heat.env.cloud_fraction == pytest.approx(1.0)
