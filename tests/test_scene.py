"""The built scene, measured.

Blender is one global session, so each class rebuilds in its own band on entry.
"""

import math
from collections.abc import Iterator
from pathlib import Path
from statistics import NormalDist

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape import blend, lwir, scene, sea, waves
from seascape.assets import Asset, manifest
from seascape.calibration import CameraCalibration
from seascape.config import Band, Mount, Scenario, load

BASELINE = Path(__file__).parent.parent / "scenarios" / "baseline.toml"
DRIFTING = BASELINE.with_name("drifting.toml")
OPEN_SEA = BASELINE.with_name("open-sea.toml")
SCENARIO = load(BASELINE)
RIG_ONLY = f'extends = "{OPEN_SEA}"\n'


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


@pytest.mark.parametrize("name", ["baseline.toml", "twin-pod.toml"])
def test_the_sun_is_out_of_every_frame(name: str) -> None:
    scenario = load(BASELINE.parent / name)
    ownship = scenario.ownship.model_copy(update={"asset": None})
    bare = scenario.model_copy(
        update={"objects": [], "targets": None, "ownship": ownship}
    )
    built = scene.build(bare, "eo")
    sun = np.array(scene._sun_vector(scenario.sky))
    for mount in scenario.rig.mounts:
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
    "ADD": lambda a, b, c: a + b,
    "SUBTRACT": lambda a, b, c: a - b,
    "MULTIPLY": lambda a, b, c: a * b,
    "MULTIPLY_ADD": lambda a, b, c: a * b + c,
    "FRACT": lambda a, b, c: a - math.floor(a),
    "ABSOLUTE": lambda a, b, c: abs(a),
    "ARCTAN2": lambda a, b, c: math.atan2(a, b),
}


def _evaluated(socket: bpy.types.NodeSocket) -> float:
    """A chain of Value and Math nodes, evaluated as Blender does."""
    node = socket.node
    if node.bl_idname == "ShaderNodeValue":
        return socket.default_value
    inputs = (
        _evaluated(i.links[0].from_socket) if i.is_linked else i.default_value
        for i in node.inputs
    )
    return _MATH[node.operation](*inputs)


def under_the_haze(material: bpy.types.Material) -> bpy.types.Node:
    output = material.node_tree.get_output_node("CYCLES")
    haze = output.inputs["Surface"].links[0].from_node
    return haze.inputs["Shader"].links[0].from_node


def baked(name: str) -> np.ndarray:
    """The red channel of a lookup image, rows bottom first."""
    image = bpy.data.images[name]
    width, height = image.size
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    return np.squeeze(pixels.reshape(height, width, 4)[..., 0])


def _blocked(origin: Vector, target: Vector) -> bool:
    ray = target - origin
    hit, *_ = bpy.context.scene.ray_cast(
        bpy.context.evaluated_depsgraph_get(),
        origin,
        ray.normalized(),
        distance=ray.length * 0.999,
    )
    return bool(hit)


def counts() -> tuple[int, ...]:
    return tuple(
        len(block) for block in (bpy.data.objects, bpy.data.materials, bpy.data.images)
    )


def test_a_named_substream_is_reproducible_and_local_to_its_name() -> None:
    def draw(seed: int, name: str) -> int:
        return int(scene._substream(seed, name).integers(2**31))

    assert draw(7, "sea/surface") == draw(7, "sea/surface")
    assert draw(7, "sea/surface") != draw(8, "sea/surface")
    assert draw(7, "sea/surface") != draw(7, "sky/haze")


class TestGeometry:
    """Everything the band does not change: where things are and where cameras look."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> scene.Built:
        return scene.build(SCENARIO, "eo")

    def test_a_flat_rig_points_where_the_scenario_asked(self) -> None:
        """The one negation. Two of them cancel and the whole rig mirrors unnoticed."""
        for mount in SCENARIO.rig.mounts:
            bearing, _ = scene.boresight_deg(camera_of(mount))
            assert bearing == pytest.approx(mount.nominal_bearing_deg, abs=1e-6)

    def test_cameras_carry_their_field_of_view_horizontally(self) -> None:
        """AUTO fits the angle to the longer image side, flipping a portrait sensor."""
        for mount in SCENARIO.rig.mounts:
            data = camera_of(mount).data
            assert data.sensor_fit == "HORIZONTAL"
            assert math.degrees(data.angle_x) == pytest.approx(mount.camera.hfov_deg)

    def test_the_near_clip_leaves_the_depth_buffer_usable_at_range(self) -> None:
        for mount in SCENARIO.rig.mounts:
            data = camera_of(mount).data
            assert data.clip_start == SCENARIO.rig.near_clip_m
            assert data.clip_start < data.clip_end

    def test_every_3d_view_clips_past_the_sea(self) -> None:
        corner_m = math.sqrt(2) * sea.sea_reach_m(SCENARIO.rig, SCENARIO.sea)
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
            assert space.clip_start == SCENARIO.rig.near_clip_m
            assert space.clip_end > corner_m

    def test_the_far_clip_clears_every_target(self) -> None:
        furthest = max(spec.range_m for spec in SCENARIO.objects)
        for mount in SCENARIO.rig.mounts:
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
        for mount in SCENARIO.rig.mounts:
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
                for part in anchor.children_recursive
                if part.type == "MESH"
                for corner in part.bound_box
            ]
            axes = list(zip(*corners, strict=True))
            asset = manifest()[spec.asset]
            assert max(axes[1]) - min(axes[1]) == pytest.approx(asset.length_m)
            # 1 mm: the fit runs through float32 mesh coordinates.
            assert min(axes[2]) == pytest.approx(-asset.draught_m, abs=1e-3), (
                "keel sits at the manifest draught, not on the surface"
            )
            assert max(axes[2]) > 0.0, "and the rest of it is above water"

    def test_each_wave_fades_as_the_core_s_visibility(self) -> None:
        nodes = bpy.data.materials["sea"].node_tree.nodes
        for i, wave in enumerate(scene.wave_field(SCENARIO)):
            inputs = nodes[f"wave_{i}"].inputs
            gone, whole = waves.fade_footprints_m(2 * math.pi / wave.k_rad_m)
            assert (
                inputs["Gone Sq"].default_value,
                inputs["Whole Sq"].default_value,
            ) == pytest.approx((gone**2, whole**2), rel=1e-6)

    def test_the_sea_carries_the_scenario_s_wave_field(self) -> None:
        nodes = bpy.data.materials["sea"].node_tree.nodes
        field = scene.wave_field(SCENARIO)
        assert len(field) == waves.COMPONENTS
        for i, wave in enumerate(field):
            k_east, k_north = wave.k_east_rad_m, wave.k_north_rad_m
            inputs = nodes[f"wave_{i}"].inputs
            carried = (
                tuple(inputs["Wavenumber"].default_value),
                inputs["Phase"].default_value,
                tuple(inputs["Slope"].default_value),
            )
            assert carried == (
                pytest.approx((k_east, k_north, -wave.omega_rad_s), rel=1e-6),
                pytest.approx(wave.phase_rad, rel=1e-6),
                pytest.approx(
                    (wave.amplitude_m * k_east, wave.amplitude_m * k_north, 0.0),
                    rel=1e-6,
                    abs=1e-9,
                ),
            )

    def test_the_sea_reaches_past_its_own_horizon(self) -> None:
        """The grid must contain the tangent point, or its edge becomes the horizon."""
        corners = [
            bpy.data.objects["sea"].matrix_world @ Vector(c)
            for c in bpy.data.objects["sea"].bound_box
        ]
        reach = min(max(abs(v.x), abs(v.y)) for v in corners)
        horizon = waves.horizon_m(SCENARIO.rig.height_m, SCENARIO.sea.refraction_k)

        assert reach > horizon
        assert reach > max(spec.range_m for spec in SCENARIO.objects)

    def test_a_hull_floats_on_the_sea_and_not_on_the_tangent_plane(self) -> None:
        """Hulls left at z = 0 fly with range and bearing still right, so nothing else
        catches it."""
        radius = waves.earth_radius_m(SCENARIO.sea.refraction_k)

        for spec in SCENARIO.objects:
            anchor = bpy.data.objects[spec.asset]
            east, north, up = anchor.matrix_world.translation

            assert up == pytest.approx(waves.sea_z_m(east, north, radius), abs=1e-3)

    def test_a_hull_beyond_the_horizon_is_cut_off(self) -> None:
        """A target past the horizon shows its waterline when it should be hull-down."""
        eye = Vector((0.0, 0.0, SCENARIO.rig.height_m))
        radius = waves.earth_radius_m(SCENARIO.sea.refraction_k)
        beyond = 18_000.0
        surface = beyond * beyond / (2.0 * radius)
        horizon = waves.horizon_m(SCENARIO.rig.height_m, SCENARIO.sea.refraction_k)
        assert horizon < beyond < sea.sea_reach_m(SCENARIO.rig, SCENARIO.sea)

        waterline = _blocked(eye, Vector((0.0, beyond, -surface)))
        mast = _blocked(eye, Vector((0.0, beyond, 30.0 - surface)))

        assert waterline, "the bulge has to hide the hull"
        assert not mast, "and leave what stands above it"

    def test_building_twice_leaves_the_same_scene(self) -> None:
        """Node trees leak when a build appends to what is already there."""
        before = counts()
        scene.build(SCENARIO, "eo")
        assert counts() == before


@pytest.mark.parametrize("band", ["eo", "ir"])
def test_a_saved_build_reopens_with_its_lookup_tables(
    band: Band, tmp_path: Path
) -> None:
    """`seascape build` saves a .blend for Blender to open; an unpacked table would
    reopen as its fill colour."""
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
    assert scene._images(outer) == {image}


@pytest.mark.parametrize(
    ("bow_deg", "bow_corner"),
    [(0.0, (0, 5, 0)), (90.0, (5, 0, 0)), (180.0, (0, -5, 0)), (270.0, (-5, 0, 0))],
)
def test_a_hull_is_fitted_along_its_own_bow_axis(bow_deg, bow_corner) -> None:
    """180 is its own inverse: the shipped hull passes with a sign error or the
    length measured along the beam. Any other bow catches both."""
    asset = Asset(
        description="x",
        url="x",
        sha256="0" * 64,
        length_m=200.0,
        draught_m=5.0,
        bow_deg=bow_deg,
        triangles=1,
        texture_px=(),
        licence="x",
        attribution="x",
    )
    long, beam = Vector(bow_corner), Vector((-bow_corner[1], bow_corner[0], 0)) / 5
    corners = [
        s * long + b * beam + Vector((0, 0, z))
        for s in (-1, 1)
        for b in (-1, 1)
        for z in (0, 1)
    ]

    fit = scene._fit(corners, asset)
    fitted = [fit @ c for c in corners]

    ys = [c.y for c in fitted]
    assert max(ys) - min(ys) == pytest.approx(200.0), "scaled along the bow axis"
    assert max(c.x for c in fitted) - min(c.x for c in fitted) == pytest.approx(40.0)
    assert min(c.z for c in fitted) == pytest.approx(-5.0), "keel at the draught"
    assert (fit @ long).y == pytest.approx(100.0), "and the bow ends up at +Y"


def test_a_pitched_pod_rolls_the_horizon_of_its_off_axis_cameras(tmp_path) -> None:
    """Horizon rolls by asin(sin(pitch) sin(yaw))."""
    pitch_deg, yaw_deg = -5.0, 40.0
    path = tmp_path / "pitched.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        f"[rig]\npitch_deg = {pitch_deg}\n\n"
        '[[rig.pods]]\nname = "bow"\nyaw_deg = 0.0\n\n'
        f'[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\nyaw_deg = {yaw_deg}\n'
    )
    scenario = load(path)
    expected = math.degrees(
        math.asin(math.sin(math.radians(pitch_deg)) * math.sin(blend.yaw(yaw_deg)))
    )

    scene.build(scenario, "eo")
    across = bpy.data.objects[
        scenario.rig.mounts[0].name
    ].matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))

    assert math.degrees(math.asin(across.normalized().z)) == pytest.approx(
        expected, abs=1e-6
    )


def _lens_pitched(tmp_path, yaw_deg: float, pitch_deg: float):
    path = tmp_path / "lens.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        '[[rig.pods]]\nname = "port"\nyaw_deg = -60.0\n\n'
        f'[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\n'
        f"yaw_deg = {yaw_deg}\npitch_deg = {pitch_deg}\n"
    )
    scenario = load(path)
    scene.build(scenario, "eo")
    return scenario.rig.mounts[0]


def test_a_pitched_lens_points_exactly_where_it_was_asked_to(tmp_path) -> None:
    """Rx inside the camera's yaw: neither angle disturbs the other, nor the horizon."""
    pitch_deg = -10.0

    mount = _lens_pitched(tmp_path, 40.0, pitch_deg)

    camera = bpy.data.objects[mount.name]
    bearing, elevation = scene.boresight_deg(camera)
    assert bearing == pytest.approx(mount.nominal_bearing_deg, abs=1e-4)
    assert elevation == pytest.approx(pitch_deg, abs=1e-4)
    across = camera.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))
    assert across.normalized().z == pytest.approx(0.0, abs=1e-6)


def test_pitch_moves_an_off_axis_camera_off_its_bearing_and_leaves_the_centre(
    tmp_path,
) -> None:
    """The pod is yawed off the bow, so the rig's pitch sits between two yaws and
    moves the off-axis lens off both its bearing and its own pitch. The centre camera
    holds to microdegrees, not zero: matrix_world is float32."""
    path = tmp_path / "pitched.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        "[rig]\npitch_deg = -5.0\n\n"
        '[[rig.pods]]\nname = "port"\nyaw_deg = -60.0\n\n'
        '[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\nyaw_deg = 0.0\n\n'
        '[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\nyaw_deg = 40.0\n'
        "pitch_deg = -10.0\n"
    )
    scenario = load(path)
    scene.build(scenario, "eo")
    centre, off_axis = (
        (scene.boresight_deg(bpy.data.objects[m.name]), m.nominal_bearing_deg)
        for m in scenario.rig.mounts
    )

    assert centre[0][0] == pytest.approx(centre[1], abs=1e-4)
    (bearing, elevation), nominal = off_axis
    assert abs(bearing - nominal) > 0.1
    assert abs(elevation - (-10.0)) > 0.1


def test_an_ownship_with_no_hull_still_carries_the_rig(tmp_path) -> None:
    path = tmp_path / "rolled.toml"
    path.write_text(f"{RIG_ONLY}\n[ownship]\nroll_deg = 5.0\n")
    scene.build(load(path), "eo")

    right = camera_of(SCENARIO.rig.mounts[0]).matrix_world.to_3x3() @ Vector(
        (1.0, 0.0, 0.0)
    )

    assert math.degrees(math.asin(-right.z)) == pytest.approx(5.0)


def test_a_near_clip_past_the_far_plane_is_an_error(tmp_path) -> None:
    path = tmp_path / "deep.toml"
    path.write_text(f'extends = "{BASELINE}"\n\n[rig]\nnear_clip_m = 500000.0\n')

    with pytest.raises(ValueError, match="near clip"):
        scene.build(load(path), "eo")


def test_a_band_the_rig_cannot_see_is_an_error(tmp_path) -> None:
    """Otherwise `next()` raises StopIteration, naming nothing."""
    path = tmp_path / "eo_only.toml"
    path.write_text(
        f'extends = "{BASELINE}"\n\n'
        '[[rig.pods]]\nname = "bow"\nyaw_deg = 0.0\n\n'
        '[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\n'
    )
    with pytest.raises(ValueError, match="no ir camera"):
        scene.build(load(path), "ir")


class TestEoBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "eo")

    def test_the_sea_refracts_at_seawater_ior(self) -> None:
        bsdf = bpy.data.materials["sea"].node_tree.nodes["Principled BSDF"]
        assert bsdf.inputs["IOR"].default_value == pytest.approx(sea.SEAWATER_IOR)

    def test_the_glitter_spreads_over_the_slope_each_pixel_leaves_out(self) -> None:
        nodes = bpy.data.materials["sea"].node_tree.nodes
        assert nodes["Principled BSDF"].inputs["Roughness"].is_linked
        table = baked("sea_unresolved_variance")
        low, high = sea.FOOTPRINT_RANGE_M
        texel = (np.arange(len(table)) + 0.5) / len(table)
        footprints = low * (high / low) ** texel
        wind, swell = scene.wind_waves(SCENARIO), scene.swell_waves(SCENARIO)
        speed = SCENARIO.sea.wind_speed_mps
        for i in (0, len(table) // 2, len(table) - 1):
            expected = waves.unresolved_slope_variance(
                speed, wind, swell, footprints[i]
            )
            assert table[i] == pytest.approx(expected, rel=1e-5)

    def test_a_gust_adds_cox_and_munk_s_variance_at_the_local_wind(self) -> None:
        roughness = bpy.data.materials["sea"].node_tree.nodes["Principled BSDF"]
        assert {"gust", "sea_gust_variance", "sea_unresolved_variance"} <= upstream(
            roughness.inputs["Roughness"]
        )
        speed = SCENARIO.sea.wind_speed_mps
        tree = bpy.data.materials["sea"].node_tree
        per_gust = tree.nodes["sea_gust_variance"].inputs[1].default_value
        assert per_gust == pytest.approx(waves.gust_slope_variance(speed), rel=1e-5)
        tile = baked("sea_gust")
        seeded = waves.von_karman_field(
            scene._substream(SCENARIO.seed, "sea/gust"),
            sea.GUST_CELLS,
            sea.GUST_SPACING_M,
            waves.GUST_LENGTH_M,
        )
        assert tile == pytest.approx(seeded, abs=1e-5)

    def test_the_gusts_are_carried_downwind_at_the_mean_wind(self) -> None:
        carry = bpy.data.materials["sea"].node_tree.nodes["gust_carry"]
        per_m = 1 / sea.GUST_TILE_M
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

    def test_the_sea_whitecaps_past_the_core_s_threshold(self) -> None:
        wind = scene.wind_waves(SCENARIO)
        speed = SCENARIO.sea.wind_speed_mps
        tree = bpy.data.materials["sea"].node_tree
        assert "gust" in upstream(tree.nodes["whitecap_excess"].inputs["Value_001"])
        table = baked("sea_breaking")
        gust_max = (
            math.sqrt(2)
            * np.abs(baked("sea_gust")).max()
            * waves.turbulence_intensity(speed)
        )
        texel = (np.arange(len(table)) + 0.5) / len(table)
        gusts = gust_max * (2 * texel - 1)
        # The threshold at each texel's gust.
        expected = [
            waves.breaking_threshold_g(wind, waves.gusty_whitecap_fraction(speed, g))
            for g in gusts
        ]
        assert table == pytest.approx(expected, rel=1e-5)
        assert np.all(np.diff(table) < 0)

    def test_the_sea_glitters_one_specular_point_per_cell(self) -> None:
        wind = scene.wind_waves(SCENARIO)
        nodes = bpy.data.materials["sea"].node_tree.nodes
        scale = nodes["glitter_cells"].inputs["Scale"].default_value
        assert scale == pytest.approx(1 / math.sqrt(waves.specular_cell_m2(wind)))
        twinkle = nodes["glitter_draw"].inputs[0].links[0].from_node
        assert twinkle.inputs[1].default_value == pytest.approx(waves.twinkle_hz(wind))

    def test_the_glint_is_the_sky_texture_s_sun(self) -> None:
        sky = bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]
        assert sky.sun_size == pytest.approx(4 * sea.SUN_SLOPE_RADIUS)

    def test_the_air_hazes_towards_the_horizon_sky(self) -> None:
        nodes = bpy.data.node_groups["haze"].nodes
        beta = nodes["haze_beta"]
        assert beta.inputs[0].links[0].from_socket.name == "Ray Length"
        assert beta.inputs[1].default_value == pytest.approx(
            SCENARIO.sky.extinction_per_m
        )
        sky = bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]
        airlight = nodes["haze_sky"]
        for name in ("sun_elevation", "sun_rotation", "aerosol_density"):
            assert getattr(airlight, name) == getattr(sky, name)

    def test_the_airlight_is_the_sky_ahead_of_the_ray(self) -> None:
        """Incoming points back at the camera; unflipped, the haze would take the sky
        behind it."""
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

    def test_a_hazed_material_is_not_hazed_again(self) -> None:
        material = bpy.data.materials["sea"]
        scene.haze(material)
        assert under_the_haze(material).bl_idname != "ShaderNodeGroup"


def test_the_ir_sky_and_air_follow_the_chosen_atmosphere() -> None:
    """Not the default: a scene that dropped `atmosphere` would read lwir's own."""
    # Validated, not `model_copy`: validation gives the air and sea the profile's own.
    data = SCENARIO.model_dump()
    data["sky"] = {k: v for k, v in data["sky"].items() if k != "t_air_k"}
    data["sea"] = {k: v for k, v in data["sea"].items() if k != "t_sea_k"}
    data["sky"]["atmosphere"] = "tropical"
    data["objects"] = []
    tropical = Scenario.model_validate(data)
    assert tropical.sea.t_sea_k == lwir.SURFACE_SEA_K["tropical"]
    assert tropical.sky.atmosphere != lwir.ATMOSPHERE
    assert tropical.sky.t_air_k == lwir.SURFACE_AIR_K["tropical"]
    scene.build(tropical, "ir")
    sky = tropical.sky
    centres = (np.arange(scene.CURVE_SAMPLES) + 0.5) / scene.CURVE_SAMPLES
    expected = lwir.sky_radiance(np.arcsin(centres), sky.t_air_k, "tropical")
    assert baked("sky_radiance") == pytest.approx(expected, rel=1e-5)
    ranges = scene._haze_ranges_m(bpy.context.scene.camera.data.clip_end)
    depth = lwir.path_optical_depth(ranges, sky.visibility_km, "tropical")
    assert baked("haze_depth") == pytest.approx(depth, rel=1e-5)


class TestIrBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "ir")

    def test_the_air_takes_the_band_s_optical_depth_by_range(self) -> None:
        """The shader reads this by log range at texel centres."""
        assert "sky_radiance.001" not in bpy.data.images, "the haze reads the world's"
        table = baked("haze_depth")
        ranges = scene._haze_ranges_m(bpy.context.scene.camera.data.clip_end)
        sky = SCENARIO.sky
        depth = lwir.path_optical_depth(ranges, sky.visibility_km, sky.atmosphere)
        assert table == pytest.approx(depth, rel=1e-5)

    def test_the_sea_does_not_glitter(self) -> None:
        """The emissivity table already takes the whole unresolved slope."""
        assert "glitter_cells" not in bpy.data.materials["sea"].node_tree.nodes

    def test_the_baked_sky_runs_cold_towards_the_zenith(self) -> None:
        """The shader reads this by sin(elevation) at texel centres."""
        curve = baked("sky_radiance")

        centres = (np.arange(len(curve)) + 0.5) / len(curve)
        sky = SCENARIO.sky
        expected = lwir.sky_radiance(np.arcsin(centres), sky.t_air_k, sky.atmosphere)
        assert curve == pytest.approx(expected, rel=1e-5)
        ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
        assert curve[-1] < 0.5 * ambient, "the zenith is much colder than ambient"
        assert np.all(np.diff(curve) <= 1e-6), "radiance falls towards the zenith"

    def test_a_vessel_reflects_what_it_does_not_emit(self) -> None:
        """A pure emitter renders a hull one flat value whichever way it is turned."""
        skin = next(m for m in bpy.data.materials if m.name.endswith("_ir"))
        # The outermost mix, not the one grading emission from shaded to sunlit.
        mix = under_the_haze(skin)

        assert scene.PAINT_EMISSIVITY < 1.0, "a blackbody has no angular structure"
        assert mix.inputs["Factor"].default_value == pytest.approx(
            scene.PAINT_EMISSIVITY
        )
        assert mix.inputs[1].links[0].from_node.bl_idname == "ShaderNodeBsdfDiffuse"

    def test_a_vessel_is_hotter_on_the_side_the_sun_is_on(self) -> None:
        skin = next(m for m in bpy.data.materials if m.name.endswith("_ir"))
        grade = under_the_haze(skin).inputs[2].links[0].from_node

        shaded, sunlit = (grade.inputs[i].links[0].from_node for i in (1, 2))
        assert (shaded.bl_idname, sunlit.bl_idname) == (
            "ShaderNodeEmission",
            "ShaderNodeEmission",
        )
        assert sunlit.inputs["Strength"].default_value == pytest.approx(
            lwir.band_radiance(SCENARIO.objects[0].t_k + SCENARIO.sky.solar_gain_k),
            rel=1e-5,
        )
        assert grade.inputs["Factor"].links[0].from_node.operation == "MAXIMUM"

    def test_the_sea_reflects_what_it_does_not_emit(self) -> None:
        """Emission alone goes dark toward grazing, where emissivity falls to zero."""
        mix = bpy.data.materials["sea"].node_tree.nodes["Mix Shader"]
        assert mix.inputs["Factor"].is_linked
        # ShaderNodeBsdfGlossy still reports its pre-4.0 bl_idname.
        assert mix.inputs[1].links[0].from_node.bl_idname == "ShaderNodeBsdfAnisotropic"
        assert mix.inputs[2].links[0].from_node.bl_idname == "ShaderNodeEmission"

    def test_the_reflection_lobe_and_the_emissivity_share_one_slope(self) -> None:
        """Both come from the pixel's unresolved slope, and must move together."""
        tree = bpy.data.materials["sea"].node_tree
        mirror = next(
            n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfAnisotropic"
        )
        table = next(
            n
            for n in tree.nodes
            if n.bl_idname == "ShaderNodeTexImage" and n.image.name == "sea_emissivity"
        )
        lookup = table.inputs["Vector"].links[0].from_node
        for socket in (mirror.inputs["Roughness"], lookup.inputs["Y"]):
            assert {"sea_unresolved_variance", "sea_gust_variance"} <= upstream(socket)

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

    def test_a_target_radiates_at_its_own_temperature(self) -> None:
        """`t_k` is in the scenario; a target rendering at its albedo ignores it."""
        for spec in SCENARIO.objects:
            hull = next(
                part
                for part in bpy.data.objects[spec.asset].children_recursive
                if part.type == "MESH"
            )
            emission = hull.material_slots[0].material.node_tree.nodes["Emission"]
            assert emission.inputs["Strength"].default_value == pytest.approx(
                lwir.band_radiance(spec.t_k), rel=1e-5
            )

    def test_the_baked_emissivity_matches_the_curve(self) -> None:
        """The shader reads this by cos(theta) at texel centres; the curve is sampled by
        theta. Rows run up the unresolved RMS slope, to all of Cox & Munk's in the
        strongest gust."""
        table = baked("sea_emissivity")
        rows, width = table.shape
        mu = (np.arange(width) + 0.5) / width
        speed = SCENARIO.sea.wind_speed_mps
        tile = baked("sea_gust")
        gust_max = math.sqrt(2) * np.abs(tile).max() * waves.turbulence_intensity(speed)
        strongest = waves.cox_munk_slope(
            speed
        ) ** 2 + gust_max * waves.gust_slope_variance(speed)
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
        scene.build(load(OPEN_SEA, ["outputs.duration_s = 0.3"]))

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
        blend.animate(empty, "location", [0.0], lambda t: (1.0, 2.0, 3.0))
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
    radius = waves.earth_radius_m(scenario.sea.refraction_k)
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
        assert up == pytest.approx(waves.sea_z_m(east, north, radius), abs=1e-3)
    sc.frame_set(0)
    assert anchor.matrix_world.translation == start
    assert start.xy.length == pytest.approx(spec.range_m)


class TestOwnshipMotion:
    MOTION = (
        "outputs.duration_s = 2.0",
        "ownship = { roll_deg = 3.0, pitch_deg = -1.0,"
        " roll = { amplitude_deg = 5.0, period_s = 4.0 },"
        " pitch = { amplitude_deg = 2.0, period_s = 4.0 },"
        " heave = { amplitude_m = 0.5, period_s = 4.0 } }",
    )

    def test_a_quarter_period_in_is_the_peak(self) -> None:
        built = scene.build(load(OPEN_SEA, self.MOTION))
        anchor, camera = built.vessel, camera_of(SCENARIO.rig.mounts[0])
        mounted = anchor.matrix_world.inverted() @ camera.matrix_world
        bpy.context.scene.frame_set(10)

        pitch, roll, _ = anchor.rotation_euler
        assert math.degrees(pitch) == pytest.approx(-1.0 + 2.0)
        assert math.degrees(roll) == pytest.approx(3.0 + 5.0)
        assert anchor.matrix_world.translation.z == pytest.approx(0.5)
        moved = anchor.matrix_world.inverted() @ camera.matrix_world
        assert np.allclose(moved, mounted, atol=1e-5)

    def test_an_ownship_without_motion_keys_nothing(self) -> None:
        built = scene.build(load(OPEN_SEA, ["outputs.duration_s = 2.0"]))

        assert built.vessel.animation_data is None


def _sea_node(name: str) -> bpy.types.ShaderNode:
    return bpy.data.materials["sea"].node_tree.nodes[name]


class TestSlicks:
    SLICKS = load(OPEN_SEA, ["sea.slick_cover = 0.3", "sea.wind_from_deg = 60.0"])

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(cls.SLICKS, "eo")

    def test_the_tile_s_top_cover_lies_under_a_slick(self) -> None:
        threshold = _sea_node("slick").inputs[1].default_value
        assert threshold == pytest.approx(NormalDist().inv_cdf(0.7))
        spacing_m = waves.WINDROW_SPACING_S * self.SLICKS.sea.wind_speed_mps
        seeded = waves.von_karman_field(
            scene._substream(self.SLICKS.seed, "sea/slick"),
            sea.GUST_CELLS,
            sea.GUST_SPACING_M,
            spacing_m,
        )
        assert baked("sea_slick") == pytest.approx(seeded, abs=1e-5)

    def test_windrows_run_along_the_wind_at_faller_and_woodcock_s_spacing(
        self,
    ) -> None:
        axes = _sea_node("slick_axes")
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
        assert np.linalg.norm(across) == pytest.approx(1 / sea.GUST_TILE_M)
        assert np.linalg.norm(along) == pytest.approx(
            np.linalg.norm(across) / waves.WINDROW_ASPECT
        )
        carry = np.array(_sea_node("slick_carry").inputs["Vector"].default_value[:2])
        drift_mps = waves.SLICK_DRIFT * self.SLICKS.sea.wind_speed_mps
        assert carry == pytest.approx([0.0, -drift_mps * np.linalg.norm(along)])

    def test_a_slick_calms_the_waves_it_damps(self) -> None:
        wind = scene.wind_waves(self.SLICKS)
        left = set(waves.slick_survivors(self.SLICKS.sea.wind_speed_mps, wind))
        nodes = bpy.data.materials["sea"].node_tree.nodes
        calmed = {
            i for i in range(len(wind)) if nodes[f"wave_{i}"].inputs["Calm"].is_linked
        }
        assert calmed == {i for i, w in enumerate(wind) if w not in left}

    def test_under_a_slick_the_pixel_leaves_what_the_slick_does(self) -> None:
        table = baked("sea_unresolved_slick_variance")
        low, high = sea.FOOTPRINT_RANGE_M
        texel = (np.arange(len(table)) + 0.5) / len(table)
        footprints = low * (high / low) ** texel
        speed = self.SLICKS.sea.wind_speed_mps
        left = waves.slick_survivors(speed, scene.wind_waves(self.SLICKS))
        for i in (0, len(table) // 2, len(table) - 1):
            expected = waves.unresolved_slope_variance(
                speed,
                left,
                scene.swell_waves(self.SLICKS),
                footprints[i],
                waves.cox_munk_slick_slope,
            )
            assert table[i] == pytest.approx(expected, rel=1e-5, abs=1e-9)
        roughness = _sea_node("Principled BSDF").inputs["Roughness"]
        assert {"slick", "sea_unresolved_slick_variance"} <= upstream(roughness)


def each_frame() -> Iterator[int]:
    sc = bpy.context.scene
    for frame in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(frame)
        yield frame


class TestSeaEvolves:
    SEQUENCE = load(OPEN_SEA, ["outputs.duration_s = 0.3"])

    def test_the_sea_keeps_the_frames_time(self) -> None:
        scene.build(self.SEQUENCE, "eo")
        times = [
            _sea_node("sea_time").outputs["Value"].default_value for _ in each_frame()
        ]
        assert times == pytest.approx(self.SEQUENCE.outputs.times_s)

    def test_a_swell_leaves_the_wind_s_waves_alone(self) -> None:
        swell = load(BASELINE, ["sea.swell = { height_m = 1.5, period_s = 11.0 }"])
        wind = scene.wave_field(SCENARIO)
        assert scene.wave_field(swell)[: len(wind)] == wind

    def test_a_still_leaves_the_sea_unkeyed(self) -> None:
        scene.build(load(OPEN_SEA), "eo")
        assert _sea_node("sea_time").outputs["Value"].default_value == 0.0
        # Drivers make animation data; a still has no action.
        assert bpy.data.materials["sea"].node_tree.animation_data.action is None


class TestLoop:
    LOOP = load(
        OPEN_SEA,
        [
            "outputs.duration_s = 30",
            "outputs.fps = 1",
            "outputs.loop = true",
        ],
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
        for wave in scene.wave_field(self.LOOP):
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


class TestOrbitingYachts:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> scene.Built:
        yachts = (
            'objects = [{ asset = "yacht", range_m = 200.0, bearing_deg = -90.0,'
            " orbit = { period_s = 80.0, count = 2 } }]"
        )
        return scene.build(
            load(DRIFTING, [yachts, "outputs.duration_s = 40", "outputs.fps = 1"])
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
    SCENARIO = load(DRIFTING, ["outputs.fps = 4"])

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
        radius = waves.earth_radius_m(scenario.sea.refraction_k)
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
            assert up == pytest.approx(waves.sea_z_m(east, north, radius), abs=1e-3)
            assert anchor.rotation_euler.z == pytest.approx(blend.yaw(spec.heading_deg))
        assert pose.length == pytest.approx(spec.range_m)

    def test_a_hull_rides_the_sea_it_sits_on(self, built: scene.Built) -> None:
        scenario = self.SCENARIO
        (anchor,) = built.targets[scenario.objects[0].asset]
        (attitude,) = anchor.children
        local = attitude.matrix_world.inverted()
        corners = [local @ c for c in scene._corners(scene._meshes([anchor]))]
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
            pitch, roll = waves.attitude(
                scene.wave_field(scenario),
                east,
                north,
                heading,
                length,
                beam,
                scenario.outputs.times_s[frame],
            )
            assert tuple(attitude.rotation_euler[:2]) == pytest.approx(
                (pitch, -roll), abs=1e-5
            )

    def test_a_loop_runs_from_its_last_frame_into_its_first_like_any_other(
        self,
    ) -> None:
        """Wrapped round, no channel bends at the seam more than anywhere else."""
        channels = dict(keyed())

        assert len(channels) == 10, (
            "the target's xyz, pitch, roll; the ownship's pitch, roll, heave; "
            "the sea's cos, sin"
        )
        for name, values in channels.items():
            bend = np.abs(np.diff(np.append(values, values[:2]), 2))
            # A few ulps: F-curves are float32.
            ulp = np.spacing(np.float32(np.abs(values).max()))
            assert bend[-2:].max() <= bend[:-2].max() + 4 * ulp, name


class TestWakes:
    def _built(self, objects: str) -> bpy.types.NodeTree:
        scene.build(load(OPEN_SEA, [f"objects = [{objects}]"]), "eo")
        return bpy.data.materials["sea"].node_tree

    def test_a_hull_under_way_leaves_foam_and_arms(self) -> None:
        tree = self._built(
            '{ asset = "yacht", range_m = 300.0, bearing_deg = 0.0, speed_mps = 8.0 }'
        )
        assert "sea_foam" in bpy.data.images
        assert "wake_normal" in tree.nodes

    def test_a_slow_ship_s_arms_go_unseen_and_unbuilt(self) -> None:
        tree = self._built(
            '{ preset = "container_ship", range_m = 2000.0, bearing_deg = 0.0, '
            "speed_mps = 5.0 }"
        )
        assert "sea_foam" in bpy.data.images
        assert "wake_normal" not in tree.nodes

    def test_a_group_under_way_leaves_a_wake(self) -> None:
        scenario = load(
            OPEN_SEA,
            [
                'targets = { asset = "yacht", count = 2, range_m = 300.0, '
                "bearing_deg = [-5.0, 5.0], heading_deg = [90.0, 90.0], "
                "speed_mps = 8.0 }"
            ],
        )
        scene.build(scenario, "eo")
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


def test_the_sea_draws_for_the_pixel_the_render_takes() -> None:
    scenario = load(OPEN_SEA)
    scene.build(scenario, "eo")
    mount = next(m for m in scenario.rig.mounts if m.camera.kind == "eo")
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
        path = tmp_path_factory.mktemp("sky") / "tiny.hdr"
        image = bpy.data.images.new("tiny", 64, 32, float_buffer=True)
        image.pixels.foreach_set(np.ones(64 * 32 * 4, np.float32))
        image.filepath_raw = str(path)
        image.file_format = "HDR"
        image.save()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(scene, "download", lambda *_: path)
            scene.build(load(OPEN_SEA, [f'sky.hdri = "{cls.SKY}"']), "eo")

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


def test_a_sky_without_a_disc_grades_no_hull_sunlit() -> None:
    sky = load(OPEN_SEA, ['sky.hdri = "overcast_soil"']).sky
    tree = bpy.data.materials.new("hull").node_tree
    assert (
        scene._sunlit_emission(tree, 290.0, sky).node.bl_idname == "ShaderNodeEmission"
    )
