"""The built scene, measured.

Blender is one global session, so each class rebuilds in its own band on entry.
"""

import math
from collections.abc import Iterator
from pathlib import Path

import bpy
import numpy as np
import pytest
from mathutils import Vector

from seascape import blend, lwir, scene, sea, waves
from seascape.assets import Asset, manifest
from seascape.calibration import CameraCalibration
from seascape.config import Mount, load

BASELINE = Path(__file__).parent.parent / "scenarios" / "baseline.toml"
UNDERWAY = BASELINE.with_name("underway.toml")
DRIFTING = BASELINE.with_name("drifting.toml")
SCENARIO = load(BASELINE)
RIG_ONLY = f'extends = "{BASELINE}"\nobjects = []\n'


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
            fade = nodes[f"wave_{i}_fade"]
            gone, whole = waves.fade_footprints_m(2 * math.pi / wave.k_rad_m)
            assert fade.interpolation_type == "SMOOTHSTEP"
            assert (
                fade.inputs["From Min"].default_value,
                fade.inputs["From Max"].default_value,
            ) == pytest.approx((gone**2, whole**2), rel=1e-6)

    def test_the_sea_carries_the_scenario_s_wave_field(self) -> None:
        nodes = bpy.data.materials["sea"].node_tree.nodes
        field = scene.wave_field(SCENARIO)
        assert len(field) == waves.COMPONENTS
        for i, wave in enumerate(field):
            k_east, k_north = wave.k_east_rad_m, wave.k_north_rad_m
            carried = (
                tuple(nodes[f"wave_{i}"].inputs["Vector_001"].default_value),
                nodes[f"wave_{i}_phase"].inputs["Value_001"].default_value,
                tuple(nodes[f"wave_{i}_slope"].inputs["Vector"].default_value),
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

    def test_waves_cost_no_geometry(self) -> None:
        """A displaced sea would change its vertex count with the wind."""
        blowing = SCENARIO.model_copy(
            update={"sea": SCENARIO.sea.model_copy(update={"wind_speed_mps": 18.0})}
        )

        before = len(bpy.data.objects["sea"].data.vertices)
        scene.build(blowing, "eo")
        after = len(bpy.data.objects["sea"].data.vertices)
        scene.build(SCENARIO, "eo")  # the class shares one scene; put it back

        assert before == after == (sea.SEA_CELLS + 1) ** 2

    def test_building_twice_leaves_the_same_scene(self) -> None:
        """Node trees leak when a build appends to what is already there."""
        before = counts()
        scene.build(SCENARIO, "eo")
        assert counts() == before


@pytest.mark.parametrize(
    ("bow_deg", "bow_corner"),
    [(0.0, (0, 5, 0)), (90.0, (5, 0, 0)), (180.0, (0, -5, 0)), (270.0, (-5, 0, 0))],
)
def test_a_hull_is_fitted_along_its_own_bow_axis(bow_deg, bow_corner) -> None:
    """180 is its own inverse: the shipped hull passes with a sign error or the
    length measured along the beam. Any other bow catches both."""
    asset = Asset(
        url="x",
        sha256="0" * 64,
        length_m=200.0,
        draught_m=5.0,
        bow_deg=bow_deg,
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


def test_a_glb_asset_leaves_its_lights_behind() -> None:
    yacht = 'objects = [{ asset = "yacht", range_m = 200.0, bearing_deg = 0.0 }]'
    scene.build(load(BASELINE, [yacht]), "eo")

    assert not [o for o in bpy.data.objects if o.type == "LIGHT"]


@pytest.mark.parametrize("band", ["eo", "ir"])
def test_the_active_camera_belongs_to_the_band_built(band) -> None:
    """Opened on the rig's first camera, an IR build could render through EO optics
    against IR materials, with nothing to say so.
    """
    scene.build(SCENARIO.model_copy(update={"objects": []}), band)
    assert f"_{band}_" in bpy.context.scene.camera.name


@pytest.mark.parametrize("yaw_deg", [-40.0, 0.0, 40.0])
def test_a_pitched_pod_rolls_the_horizon_of_its_off_axis_cameras(
    tmp_path, yaw_deg
) -> None:
    """Horizon rolls by asin(sin(pitch) sin(yaw))."""
    pitch_deg = -5.0
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


def _lens_pitched(
    tmp_path, yaw_deg: float, pitch_deg: float, rig_pitch_deg: float = 0.0
):
    path = tmp_path / "lens.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        f"[rig]\npitch_deg = {rig_pitch_deg}\n\n"
        '[[rig.pods]]\nname = "port"\nyaw_deg = -60.0\n\n'
        f'[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\n'
        f"yaw_deg = {yaw_deg}\npitch_deg = {pitch_deg}\n"
    )
    scenario = load(path)
    scene.build(scenario, "eo")
    return scenario.rig.mounts[0]


@pytest.mark.parametrize("yaw_deg", [-40.0, 0.0, 40.0])
def test_a_pitched_lens_points_exactly_where_it_was_asked_to(tmp_path, yaw_deg) -> None:
    """Rx inside the camera's yaw: neither angle disturbs the other."""
    pitch_deg = -10.0

    mount = _lens_pitched(tmp_path, yaw_deg, pitch_deg)

    bearing, elevation = scene.boresight_deg(bpy.data.objects[mount.name])
    assert bearing == pytest.approx(mount.nominal_bearing_deg, abs=1e-4)
    assert elevation == pytest.approx(pitch_deg, abs=1e-4)


@pytest.mark.parametrize("yaw_deg", [-40.0, 40.0])
def test_a_pitched_pod_disturbs_a_pitched_lens(tmp_path, yaw_deg) -> None:
    """Rx(rig) still sits between the yaws: neither angle survives intact."""
    mount = _lens_pitched(tmp_path, yaw_deg, -10.0, rig_pitch_deg=-5.0)

    bearing, elevation = scene.boresight_deg(bpy.data.objects[mount.name])

    assert abs(bearing - mount.nominal_bearing_deg) > 0.1
    assert abs(elevation - (-10.0)) > 0.1


@pytest.mark.parametrize("yaw_deg", [-40.0, 0.0, 40.0])
def test_a_pitched_lens_keeps_its_horizon_level(tmp_path, yaw_deg) -> None:
    mount = _lens_pitched(tmp_path, yaw_deg, -10.0)

    across = bpy.data.objects[mount.name].matrix_world.to_3x3() @ Vector(
        (1.0, 0.0, 0.0)
    )

    assert across.normalized().z == pytest.approx(0.0, abs=1e-6)


def _pod_pitched(tmp_path, yaw_deg: float, pitch_deg: float = -5.0):
    """A one-pod rig yawed off the bow, so pitch sits between two non-zero yaws."""
    path = tmp_path / "pitched.toml"
    path.write_text(
        f"{RIG_ONLY}\n"
        f"[rig]\npitch_deg = {pitch_deg}\n\n"
        '[[rig.pods]]\nname = "port"\nyaw_deg = -60.0\n\n'
        f'[[rig.pods.cameras]]\npreset = "eo_4k_49deg"\nyaw_deg = {yaw_deg}\n'
    )
    scenario = load(path)
    scene.build(scenario, "eo")
    mount = scenario.rig.mounts[0]
    bearing, _ = scene.boresight_deg(bpy.data.objects[mount.name])
    return bearing, mount.nominal_bearing_deg


@pytest.mark.parametrize("yaw_deg", [-40.0, 40.0])
def test_pitch_moves_an_off_axis_camera_off_its_bearing(tmp_path, yaw_deg) -> None:
    bearing, nominal = _pod_pitched(tmp_path, yaw_deg)

    assert abs(bearing - nominal) > 0.1


def test_pitch_leaves_a_centre_camera_on_its_nominal_bearing(tmp_path) -> None:
    """Exact down the pod axis. Microdegrees, not zero: matrix_world is float32."""
    bearing, nominal = _pod_pitched(tmp_path, 0.0)

    assert bearing == pytest.approx(nominal, abs=1e-4)


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

    def test_the_sea_whitecaps_past_the_core_s_threshold(self) -> None:
        wind = scene.wind_waves(SCENARIO)
        speed = SCENARIO.sea.wind_speed_mps
        tree = bpy.data.materials["sea"].node_tree
        threshold = tree.nodes["whitecap_excess"].inputs["Value_001"].default_value
        assert threshold == pytest.approx(
            waves.breaking_threshold_g(wind, waves.whitecap_fraction(speed)), rel=1e-6
        )
        cosines = [n for n in tree.nodes if getattr(n, "operation", "") == "COSINE"]
        assert len(cosines) == len(wind)

    def test_the_sky_is_lit(self) -> None:
        assert bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]

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
        nodes = bpy.data.materials["haze"].node_tree.nodes
        beta = SCENARIO.sky.extinction_per_m
        assert nodes["haze_beta"].outputs["Value"].default_value == pytest.approx(beta)
        for node, socket in (
            ("haze_absorption", "Density"),
            ("haze_emission", "Strength"),
        ):
            assert nodes[node].inputs[socket].links[0].from_node.name == "haze_beta"
        sky = bpy.data.worlds["sky"].node_tree.nodes["Sky Texture"]
        airlight = nodes["haze_airlight"]
        for name in ("sun_elevation", "sun_rotation", "aerosol_density"):
            assert getattr(airlight, name) == getattr(sky, name)

    def test_the_airlight_is_the_sky_ahead_of_the_ray(self) -> None:
        """Incoming points back at the camera; unflipped, the haze would take the sky
        behind it."""
        nodes = bpy.data.materials["haze"].node_tree.nodes
        ahead, horizon = nodes["haze_ahead"], nodes["haze_horizon"]
        assert ahead.inputs["Vector"].links[0].from_socket.name == "Incoming"
        assert ahead.inputs["Scale"].default_value == -1.0
        assert horizon.inputs[0].links[0].from_node == ahead
        assert tuple(horizon.inputs[1].default_value) == (-1.0, -1.0, 0.0)
        (into_sky,) = horizon.outputs["Vector"].links
        assert into_sky.to_node.name == "haze_airlight"
        assert into_sky.is_valid


class TestIrBand:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def built(cls) -> None:
        scene.build(SCENARIO, "ir")

    def test_the_air_takes_the_band_s_optical_depth_by_range(self) -> None:
        """The shader reads this by log range at texel centres."""
        table = baked("haze_extinction")
        ranges = scene._haze_ranges_m(bpy.context.scene.camera.data.clip_end)
        depth = lwir.path_optical_depth(ranges, SCENARIO.sky.visibility_km)
        assert table == pytest.approx(np.gradient(depth, ranges), rel=1e-5)

    def test_the_sea_does_not_glitter(self) -> None:
        """The emissivity table already takes the whole unresolved slope."""
        assert "glitter_cells" not in bpy.data.materials["sea"].node_tree.nodes

    def test_the_sky_carries_downwelling_radiance(self) -> None:
        """What renders is the Background node, not `World.color`.

        A new world already has `use_nodes` set, so assigning `World.color` changes
        nothing a camera sees.
        """
        background = bpy.data.worlds["sky"].node_tree.nodes["Background"]
        assert background.inputs["Color"].is_linked

    def test_the_baked_sky_runs_cold_towards_the_zenith(self) -> None:
        """The shader reads this by sin(elevation) at texel centres."""
        curve = baked("sky_radiance")

        centres = (np.arange(len(curve)) + 0.5) / len(curve)
        expected = lwir.sky_radiance(np.arcsin(centres), SCENARIO.sky.t_air_k)
        assert curve == pytest.approx(expected, rel=1e-5)
        ambient = lwir.band_radiance(SCENARIO.sky.t_air_k)
        assert curve[-1] < 0.5 * ambient, "the zenith is much colder than ambient"
        assert np.all(np.diff(curve) <= 1e-6), "radiance falls towards the zenith"

    def test_a_vessel_reflects_what_it_does_not_emit(self) -> None:
        """A pure emitter renders a hull one flat value whichever way it is turned."""
        skin = next(m for m in bpy.data.materials if m.name.endswith("_ir"))
        # The outermost mix, not the one grading emission from shaded to sunlit.
        output = next(
            n for n in skin.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
        )
        mix = output.inputs["Surface"].links[0].from_node

        assert scene.PAINT_EMISSIVITY < 1.0, "a blackbody has no angular structure"
        assert mix.inputs["Factor"].default_value == pytest.approx(
            scene.PAINT_EMISSIVITY
        )
        assert mix.inputs[1].links[0].from_node.bl_idname == "ShaderNodeBsdfDiffuse"

    def test_a_vessel_is_hotter_on_the_side_the_sun_is_on(self) -> None:
        skin = next(m for m in bpy.data.materials if m.name.endswith("_ir"))
        output = next(
            n for n in skin.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"
        )
        grade = output.inputs["Surface"].links[0].from_node.inputs[2].links[0].from_node

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

        def upstream(socket: bpy.types.NodeSocket) -> set[str]:
            seen, todo = set(), [socket]
            while todo:
                for link in todo.pop().links:
                    if link.from_node.name not in seen:
                        seen.add(link.from_node.name)
                        todo.extend(link.from_node.inputs)
            return seen

        assert "sea_unresolved_variance" in upstream(mirror.inputs["Roughness"])
        assert "sea_unresolved_variance" in upstream(lookup.inputs["Y"])

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
        theta. Rows run up the unresolved RMS slope, to all of Cox & Munk's."""
        table = baked("sea_emissivity")
        rows, width = table.shape
        mu = (np.arange(width) + 0.5) / width
        # Per axis: lwir draws each facet's two slopes with this sigma.
        sigma_max = waves.cox_munk_slope(SCENARIO.sea.wind_speed_mps) / math.sqrt(2)
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
    @pytest.fixture
    def empty(self) -> bpy.types.Object:
        scene.build(load(BASELINE, ["outputs.duration_s = 0.3"]))
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
    scenario = load(UNDERWAY, ["outputs.duration_s = 0.3"])
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
        built = scene.build(load(BASELINE, self.MOTION))
        anchor, camera = built.vessel, camera_of(SCENARIO.rig.mounts[0])
        mounted = anchor.matrix_world.inverted() @ camera.matrix_world
        bpy.context.scene.frame_set(10)

        pitch, roll, _ = anchor.rotation_euler
        assert math.degrees(pitch) == pytest.approx(-1.0 + 2.0)
        assert math.degrees(roll) == pytest.approx(3.0 + 5.0)
        assert anchor.matrix_world.translation.z == pytest.approx(0.5)
        moved = anchor.matrix_world.inverted() @ camera.matrix_world
        assert np.allclose(moved, mounted, atol=1e-5)

    @pytest.mark.parametrize("duration_s", [0.0, 2.0])
    def test_an_ownship_without_motion_keys_nothing(self, duration_s) -> None:
        built = scene.build(load(BASELINE, [f"outputs.duration_s = {duration_s}"]))

        assert built.vessel.animation_data is None


def _sea_node(name: str) -> bpy.types.ShaderNode:
    return bpy.data.materials["sea"].node_tree.nodes[name]


class TestSeaEvolves:
    SEQUENCE = load(BASELINE, ["outputs.duration_s = 0.3"])
    LOOP = load(
        BASELINE, ["outputs.duration_s = 30", "outputs.fps = 1", "outputs.loop = true"]
    )

    def _each_frame(self) -> Iterator[int]:
        sc = bpy.context.scene
        for frame in range(sc.frame_start, sc.frame_end + 1):
            sc.frame_set(frame)
            yield frame

    @pytest.mark.parametrize("band", ["eo", "ir"])
    def test_the_sea_keeps_the_frames_time(self, band) -> None:
        scene.build(self.SEQUENCE, band)
        times = [
            _sea_node("sea_time").outputs["Value"].default_value
            for _ in self._each_frame()
        ]
        assert times == pytest.approx(self.SEQUENCE.outputs.times_s)

    def test_a_loop_keys_the_time_round_a_circle(self) -> None:
        scene.build(self.LOOP, "eo")
        span_s = self.LOOP.outputs.span_s
        for frame in self._each_frame():
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

    def test_a_swell_leaves_the_wind_s_waves_alone(self) -> None:
        swell = load(BASELINE, ["sea.swell = { height_m = 1.5, period_s = 11.0 }"])
        wind = scene.wave_field(SCENARIO)
        assert scene.wave_field(swell)[: len(wind)] == wind

    def test_a_still_leaves_the_sea_unkeyed(self) -> None:
        scene.build(SCENARIO, "eo")
        assert _sea_node("sea_time").outputs["Value"].default_value == 0.0
        assert bpy.data.materials["sea"].node_tree.animation_data is None


def test_a_drifting_hull_traces_a_figure_eight_about_its_pose() -> None:
    scenario = load(DRIFTING, ["outputs.fps = 8"])
    (spec,) = scenario.objects
    assert spec.drift is not None
    anchor = scene.build(scenario).targets[spec.asset][0]
    period_s = scenario.outputs.period_s(spec.drift.period_s)
    heading = math.radians(spec.heading_deg)
    ahead = Vector((math.sin(heading), math.cos(heading)))
    starboard = Vector((math.cos(heading), -math.sin(heading)))
    radius = waves.earth_radius_m(scenario.sea.refraction_k)
    pose = anchor.matrix_world.translation.xy.copy()
    s = math.sqrt(0.5)
    eighths = [(0, 0), (s, 1), (1, 0), (s, -1), (0, 0), (-s, 1), (-1, 0), (-s, -1)]
    sc = bpy.context.scene

    for eighth, (across, along) in enumerate(eighths):
        sc.frame_set(round(eighth * period_s / 8 * scenario.outputs.fps))
        east, north, up = anchor.matrix_world.translation
        offset = Vector((east, north)) - pose

        assert offset.dot(starboard) == pytest.approx(
            across * spec.drift.sway_m, abs=1e-2
        )
        assert offset.dot(ahead) == pytest.approx(along * spec.drift.surge_m, abs=1e-2)
        assert up == pytest.approx(waves.sea_z_m(east, north, radius), abs=1e-3)
        assert anchor.rotation_euler.z == pytest.approx(blend.yaw(spec.heading_deg))
    assert pose.length == pytest.approx(spec.range_m)


def test_orbiting_hulls_share_a_lap_clockwise_bow_first() -> None:
    yachts = (
        'objects = [{ asset = "yacht", range_m = 200.0, bearing_deg = -90.0,'
        " orbit = { period_s = 80.0, count = 2 } }]"
    )
    scenario = load(DRIFTING, [yachts, "outputs.duration_s = 40", "outputs.fps = 1"])
    first, second = scene.build(scenario).targets["yacht"]
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
            assert hull.rotation_euler.z == pytest.approx(blend.yaw(bearing_deg + 90))


def test_a_hull_rides_the_sea_it_sits_on() -> None:
    scenario = load(DRIFTING, ["outputs.fps = 2"])
    built = scene.build(scenario)
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


def test_a_loop_runs_from_its_last_frame_into_its_first_like_any_other() -> None:
    """Wrapped round, no channel bends at the seam more than it does anywhere else."""
    scene.build(load(DRIFTING, ["outputs.fps = 2"]))
    channels = dict(keyed())

    assert len(channels) == 10, (
        "the target's xyz, pitch, roll; the ownship's pitch, roll, heave; the sea's "
        "cos, sin"
    )
    for name, values in channels.items():
        bend = np.abs(np.diff(np.append(values, values[:2]), 2))
        # A few ulps: F-curves are float32.
        ulp = np.spacing(np.float32(np.abs(values).max()))
        assert bend[-2:].max() <= bend[:-2].max() + 4 * ulp, name
