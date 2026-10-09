"""Loader rules and the shipped presets. No Blender."""

import json
import math
import runpy
import tomllib
import warnings
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from seascape import lwir, skies
from seascape.config import (
    CFG_DIR,
    Band,
    Camera,
    Object,
    Orbit,
    Outputs,
    Rig,
    Samples,
    Scenario,
    Sea,
    Sky,
    Targets,
    json_schema,
    load,
    substream,
)

SCENARIOS = Path(__file__).parents[1] / "scenarios"
BASELINE = SCENARIOS / "baseline.toml"
HERO = SCENARIOS.parent / "docs" / "hero.toml"
RANDOMIZED = SCENARIOS / "randomized.toml"
SCHEMA = Path(__file__).parents[1] / "schema" / "scenario.json"


def variant(tmp_path: Path, body: str) -> Path:
    """A scenario extending the baseline; `body` is the override TOML."""
    path = tmp_path / "variant.toml"
    path.write_text(f'extends = "{BASELINE}"\n\n{body}')
    return path


@pytest.fixture(scope="module")
def baseline() -> Scenario:
    return load(BASELINE)


@pytest.fixture(scope="module")
def twin_pod() -> Scenario:
    return load(SCENARIOS / "twin-pod.toml")


def test_the_baseline_states_its_own_rig(baseline) -> None:
    assert [mount.name for mount in baseline.mounts] == ["bow_eo", "bow_ir"]


def test_the_baseline_carries_no_scenario_a_variant_would_inherit(baseline) -> None:
    """`extends` copies whatever is here; a variant never asked for a ring."""
    assert (baseline.ownship.asset, baseline.targets) == (None, None)


def test_a_preset_supplies_optics_and_the_block_supplies_the_mount(twin_pod) -> None:
    eo, ir = twin_pod.mounts[0], twin_pod.mounts[-1]

    assert (eo.camera.hfov_deg, eo.camera.width_px, eo.camera.height_px) == (
        49.0,
        3840,
        2160,
    )
    assert (ir.camera.hfov_deg, ir.camera.width_px, ir.camera.height_px) == (
        24.0,
        640,
        480,
    )
    assert (eo.rig_name, eo.nominal_bearing_deg) == ("port", -100.0)
    assert (ir.rig_name, ir.nominal_bearing_deg) == ("starboard", 10.0)


def test_a_nominal_bearing_is_its_rig_plus_its_fan(twin_pod) -> None:
    port = twin_pod.rigs["port"]

    assert port.yaw_deg == -60.0
    assert [camera.yaw_deg for camera in port.cameras.values()] == [
        -40.0,
        0.0,
        40.0,
        50.0,
    ]
    assert [mount.nominal_bearing_deg for mount in twin_pod.mounts][:4] == [
        -100.0,
        -60.0,
        -20.0,
        -10.0,
    ]


def test_an_installation_takes_its_model_and_height_from_its_presets(
    twin_pod,
) -> None:
    assert {name: (rig.model, rig.height_m) for name, rig in twin_pod.rigs.items()} == {
        "port": ("Pod", 19.7),
        "starboard": ("Pod", 19.7),
    }


def test_samples_covers_every_band() -> None:
    """`scene._output` reads this with `getattr(samples, band)`, so a band added to the
    Literal without a field here fails mid-build rather than in validation."""
    assert set(Samples.model_fields) == set(get_args(Band.__value__))


def test_an_override_is_the_toml_line_it_would_be_written_as(baseline) -> None:
    scenario = load(BASELINE, ["rigs.bow.pitch_deg = -5"])

    assert scenario.rigs["bow"].pitch_deg == -5.0
    assert scenario.rigs["bow"].height_m == baseline.rigs["bow"].height_m


def test_an_override_resolves_presets_like_a_line_in_the_file(baseline) -> None:
    scenario = load(BASELINE, ['rigs.port = { preset = "port" }'])

    assert list(scenario.rigs) == ["bow", "port"]
    assert scenario.rigs["port"].cameras["ir"].yaw_deg == 50.0


def test_an_override_changes_one_camera_and_leaves_the_others(twin_pod) -> None:
    override = "rigs.port.cameras.eo_c.hfov_deg = 30"
    scenario = load(SCENARIOS / "twin-pod.toml", [override])

    changed = [
        m.name
        for m, n in zip(twin_pod.mounts, scenario.mounts, strict=True)
        if m.camera != n.camera
    ]
    assert changed == ["port_eo_c"]


def test_an_override_merges_a_table_rather_than_replacing_it() -> None:
    scenario = load(BASELINE, ["outputs.samples.eo = 8"])

    assert (scenario.outputs.samples.eo, scenario.outputs.samples.ir) == (8, 64)


def test_overrides_apply_in_order() -> None:
    scenario = load(BASELINE, ["rigs.bow.pitch_deg = -5", "rigs.bow.pitch_deg = -10"])

    assert scenario.rigs["bow"].pitch_deg == -10.0


@pytest.mark.parametrize(
    ("override", "error"),
    [
        pytest.param("rigs.bow.tlit_deg = -5", ValidationError, id="misspelt-key"),
        pytest.param("outputs.format = png", ValueError, id="unquoted-string"),
        pytest.param("garbage", ValueError, id="not-an-assignment"),
    ],
)
def test_a_bad_override_is_refused(override: str, error: type[Exception]) -> None:
    """`tomllib.TOMLDecodeError` is a `ValueError`: the CLI reports both alike."""
    with pytest.raises(error):
        load(BASELINE, [override])


def test_objects_merge_their_preset(baseline) -> None:
    """A list-of-tables preset: asset and temperature from cfg, pose here."""
    obj = baseline.objects[0]
    assert (obj.asset, obj.t_k) == ("container_ship", 295.0)
    assert (obj.range_m, obj.bearing_deg) == (2000.0, 8.0)


def test_a_block_overrides_its_own_preset(tmp_path) -> None:
    """Disjoint keys would pass whichever way the merge ran."""
    scenario = load(
        variant(
            tmp_path,
            '[rigs.bow.cameras.eo]\npreset = "eo_4k_49deg"\nhfov_deg = 10.0\n',
        )
    )
    camera = scenario.rigs["bow"].cameras["eo"]
    assert camera.hfov_deg == 10.0
    assert camera.width_px == 3840  # untouched by the block


def test_a_preset_outranks_an_inherited_value(tmp_path, baseline) -> None:
    """Expanding after the parent merge inverts this, and nothing else notices."""
    (tmp_path / "single.toml").write_text(
        'height_m = 2.0\n\n[cameras.ir]\npreset = "ir_vga_24deg"\n'
    )
    scenario = load(variant(tmp_path, '[rigs.bow]\npreset = "./single.toml"\n'))
    bow = scenario.rigs["bow"]
    assert bow.height_m == 2.0
    assert bow.cameras["ir"].height_px == 480
    assert baseline.rigs["bow"].cameras["ir"].height_px != 480


def test_tables_merge_and_lists_replace(tmp_path, baseline) -> None:
    scenario = load(
        variant(
            tmp_path,
            "[sea]\nwind_speed_mps = 3.0\n\n[rigs.bow.cameras.eo]\nhfov_deg = 30.0\n\n"
            '[[objects]]\nasset = "yacht"\nrange_m = 300.0\nbearing_deg = 0.0\n',
        )
    )
    assert scenario.sea.wind_speed_mps == 3.0
    assert scenario.sea.t_sea_k == baseline.sea.t_sea_k
    assert scenario.rigs["bow"].cameras["eo"].hfov_deg == 30.0
    assert scenario.rigs["bow"].cameras["ir"] == baseline.rigs["bow"].cameras["ir"]
    assert [spec.asset for spec in scenario.objects] == ["yacht"]


@pytest.mark.parametrize("name", ["./mine.toml", "mine.toml"])
def test_preset_can_be_a_path(tmp_path, name) -> None:
    (tmp_path / "mine.toml").write_text('band = "eo"\nhfov_deg = 12.0\n')
    scenario = load(variant(tmp_path, f'[rigs.bow.cameras.eo]\npreset = "{name}"\n'))
    assert scenario.rigs["bow"].cameras["eo"].hfov_deg == 12.0


@pytest.mark.parametrize(
    ("preset", "error", "match"),
    [
        ("not_a_real_preset", FileNotFoundError, "not_a_real_preset.toml"),
        (12, TypeError, "must be a string"),
    ],
)
def test_bad_presets_fail_loudly(tmp_path, preset, error, match) -> None:
    with pytest.raises(error, match=match):
        load(variant(tmp_path, f"[rigs.bow]\npreset = {json.dumps(preset)}\n"))


def test_extends_demands_a_path(tmp_path) -> None:
    path = tmp_path / "bare.toml"
    path.write_text('extends = "baseline"\n')
    with pytest.raises(ValueError, match="must be a path"):
        load(path)


def test_circular_extends_raises_rather_than_recursing(tmp_path) -> None:
    (tmp_path / "a.toml").write_text('extends = "b.toml"\n')
    (tmp_path / "b.toml").write_text('extends = "a.toml"\n')
    with pytest.raises(ValueError, match="circular include"):
        load(tmp_path / "a.toml")


def test_a_mistyped_key_is_an_error_not_a_silent_default(tmp_path) -> None:
    with pytest.raises(ValidationError, match="height_metres"):
        load(variant(tmp_path, "[rigs.bow]\nheight_metres = 22.0\n"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_numbers_are_rejected(tmp_path, value) -> None:
    """`yaw_deg` carries no bound, so nothing else would catch one."""
    with pytest.raises(ValidationError, match="yaw_deg"):
        load(variant(tmp_path, f"[rigs.bow]\nyaw_deg = {value}\n"))


def test_committed_schema_matches_the_models() -> None:
    """Editors validate against the committed file; a stale one is worse than none."""
    assert json.loads(SCHEMA.read_text()) == json_schema(), (
        "schema/scenario.json is stale. Regenerate it:\n"
        "    uv run seascape schema > schema/scenario.json"
    )


def test_every_model_and_field_describes_itself() -> None:
    """The description is the hover doc; a field without one is only a type."""
    schema = Scenario.model_json_schema()
    models = {"Scenario": schema} | {
        name: model for name, model in schema["$defs"].items() if "properties" in model
    }
    missing = [name for name, model in models.items() if not model.get("description")]
    # A bare reference to a model hovers as that model's own description.
    missing += [
        f"{name}.{field}"
        for name, model in models.items()
        for field, spec in model["properties"].items()
        if not spec.get("description")
        and spec.get("$ref", "").removeprefix("#/$defs/") not in models
    ]
    assert not missing


def test_a_named_substream_is_reproducible_and_local_to_its_name() -> None:
    def draw(seed: int, name: str) -> int:
        return int(substream(seed, name).integers(2**31))

    assert draw(7, "sea/surface") == draw(7, "sea/surface")
    assert draw(7, "sea/surface") != draw(8, "sea/surface")
    assert draw(7, "sea/surface") != draw(7, "sky/haze")


SUN = "sky.sun_elevation_deg = { uniform = [5.0, 60.0] }"


def _sun(seed: int, *more: str) -> float:
    sun = load(BASELINE, [SUN, f"seed = {seed}", *more]).sky.sun_elevation_deg
    assert sun is not None
    return sun


def test_a_uniform_draw_is_inside_its_bounds_and_follows_the_seed() -> None:
    suns = [_sun(seed) for seed in range(20)]
    assert all(5.0 <= sun <= 60.0 for sun in suns)
    assert len(set(suns)) == len(suns)
    assert _sun(3) == _sun(3)


def test_a_new_draw_leaves_the_others_alone() -> None:
    assert _sun(3) == _sun(3, "sea.wind_speed_mps = { uniform = [2.0, 12.0] }")


def test_a_chosen_draw_is_drawn_apart_from_the_pick() -> None:
    chosen = "sky.sun_elevation_deg = { choice = [{ uniform = [5.0, 60.0] }] }"
    assert all(_sun(seed) != _sun(seed, chosen) for seed in range(20))


def test_a_draw_resolves_in_a_list_and_in_a_drawn_table(tmp_path: Path) -> None:
    path = variant(
        tmp_path,
        "[[objects]]\n"
        'asset = { choice = ["yacht", "cargo_ship"] }\n'
        "range_m = 500.0\n"
        "bearing_deg = 0.0\n"
        "drift = { choice = [{ sway_m = { uniform = [1.0, 2.0] }, surge_m = 1.0, "
        "period_s = 30.0 }] }\n",
    )
    assets, sways = set(), set()
    for seed in range(20):
        (ship,) = load(path, [f"seed = {seed}"]).objects
        assert ship.drift is not None
        assets.add(ship.asset)
        sways.add(ship.drift.sway_m)
    assert assets == {"yacht", "cargo_ship"}
    assert all(1.0 <= sway <= 2.0 for sway in sways)
    assert len(sways) == 20


def test_a_draw_replaces_the_table_it_meets() -> None:
    load(
        BASELINE,
        ['sky = { choice = [{ hdri = "sunflowers" }, { sun_elevation_deg = 10.0 }] }'],
    )


def test_a_table_over_a_choice_of_tables_goes_into_every_option() -> None:
    skies = 'sky = { choice = [{ hdri = "sunflowers" }, { sun_elevation_deg = 10.0 }] }'
    for seed in range(8):
        drawn = load(BASELINE, [skies, f"seed = {seed}"]).sky
        fixed = load(
            BASELINE, [skies, f"seed = {seed}", "sky.visibility_km = 10.0"]
        ).sky
        assert fixed == drawn.model_copy(update={"visibility_km": 10.0})


def test_a_preset_in_a_choice_is_a_preset_of_the_field() -> None:
    drawn = load(BASELINE, ['rigs.bow = { choice = [{ preset = "port" }] }']).rigs
    assert drawn == {"bow": load(SCENARIOS / "port-pod.toml").rigs["port"]}


def test_a_preset_in_a_choice_of_cameras_is_a_camera_preset() -> None:
    cameras = '{ choice = [{ eo = { preset = "eo_4k_49deg" } }] }'
    drawn = load(BASELINE, [f"rigs.bow.cameras = {cameras}"]).rigs["bow"]
    assert drawn.cameras["eo"].width_px == 3840


def test_the_schema_never_offers_a_drawn_seed() -> None:
    assert "$ref" not in str(json_schema()["properties"]["seed"])


def test_the_seed_is_never_drawn() -> None:
    with pytest.raises(ValueError, match="seed cannot be drawn"):
        load(BASELINE, ["seed = { choice = [1, 2] }"])


def test_a_draw_is_validated_as_the_field_it_lands_in() -> None:
    with pytest.raises(ValidationError, match="sun_elevation_deg"):
        load(BASELINE, ["sky.sun_elevation_deg = { uniform = [100.0, 120.0] }"])


@pytest.mark.parametrize("path", [BASELINE, HERO, RANDOMIZED])
def test_a_scenario_points_at_the_committed_schema(path: Path) -> None:
    """The `#:schema` line is a comment, so nothing else would ever notice it rot."""
    line = path.read_text().splitlines()[0]
    assert line.startswith("#:schema ")
    assert (path.parent / line.removeprefix("#:schema ")).resolve() == SCHEMA


@pytest.mark.parametrize(
    "overrides", runpy.run_path(str(HERO.with_suffix(".py")))["SKIES"].values()
)
def test_every_readme_hero_sky_loads_without_a_warning(overrides: list[str]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load(HERO, overrides)


def test_the_randomized_example_loads() -> None:
    load(RANDOMIZED)


def test_every_shipped_preset_parses() -> None:
    """A preset directory is named after the block it serves, and holds valid TOML."""
    presets = sorted(CFG_DIR.rglob("*.toml"))
    assert {path.parent.name for path in presets} == {"rigs", "cameras", "objects"}
    for preset in presets:
        tomllib.load(preset.open("rb"))


@pytest.mark.parametrize("stops", [-127.0, 127.0])
def test_a_compensation_float32_cannot_hold_is_rejected(stops: float) -> None:
    with pytest.raises(ValidationError, match="exposure_compensation_ev"):
        Outputs(exposure_compensation_ev=stops)


EO = Camera(band="eo", hfov_deg=45.0, width_px=8, height_px=8)


@pytest.mark.parametrize("name", ["../escaped", "/tmp/absolute", "sub/dir"])
def test_a_camera_cannot_be_path_text(name: str) -> None:
    with pytest.raises(ValidationError, match="pattern"):
        Rig(height_m=12.0, cameras={name: EO})


def test_a_camera_is_named_for_its_rig_and_itself() -> None:
    rigs = {
        "port": Rig(height_m=12.0, cameras={"eo": EO}),
        "starboard": Rig(height_m=12.0, cameras={"eo": EO}),
    }
    assert [m.name for m in Scenario(rigs=rigs).mounts] == ["port_eo", "starboard_eo"]


def test_two_cameras_cannot_share_a_name() -> None:
    """They would share a datablock and overwrite each other's render."""
    rigs = {
        "a_b": Rig(height_m=12.0, cameras={"c": EO}),
        "a": Rig(height_m=12.0, cameras={"b_c": EO}),
    }
    with pytest.raises(ValidationError, match="share a name"):
        Scenario(rigs=rigs)


def test_sea_temperature_bounds_are_the_tables_span() -> None:
    bounds = Sea.model_json_schema()["properties"]["t_sea_k"]
    _, temperatures, _ = lwir._table()
    assert (bounds["minimum"], bounds["maximum"]) == (temperatures[0], temperatures[-1])


@pytest.mark.parametrize(
    ("duration_s", "fps", "times_s"),
    [(0.0, 10, [0.0]), (0.3, 10, [0.0, 0.1, 0.2]), (1.0, 2, [0.0, 0.5])],
)
def test_a_frame_is_at_its_index_over_the_rate(duration_s, fps, times_s) -> None:
    assert Outputs(duration_s=duration_s, fps=fps).times_s == pytest.approx(times_s)


def test_a_loop_rounds_each_period_to_a_whole_fraction_of_the_clip() -> None:
    loop = Outputs(duration_s=30.0, loop=True)
    periods = [loop.period_s(p) for p in (7.0, 9.0, 30.0, 100.0)]
    assert periods == pytest.approx([7.5, 10.0, 30.0, 30.0])
    assert Outputs(duration_s=30.0).period_s(9.0) == 9.0


@pytest.mark.parametrize(
    ("name", "overrides", "match"),
    [
        ("baseline.toml", ["outputs.loop = true"], "duration_s > 0"),
        ("underway.toml", ["outputs.loop = true"], "drift"),
        (
            "drifting.toml",
            [
                'targets = { asset = "container_ship", count = 2, range_m = 900.0,'
                " bearing_deg = [-5.0, 5.0], speed_mps = 1.0 }"
            ],
            "drift",
        ),
        ("drifting.toml", ["outputs.duration_s = 8"], "ownship.roll's 9.0 s"),
        ("drifting.toml", ["outputs.duration_s = 20"], "container_ship drift"),
        ("port-pod-loop.toml", ["outputs.duration_s = 30"], "yacht orbit's 40.0 s"),
        (
            "drifting.toml",
            ["sea.swell = { height_m = 1.0, period_s = 40.0 }"],
            "sea.swell's 40.0 s",
        ),
    ],
)
def test_a_loop_that_cannot_close_is_an_error(name, overrides, match) -> None:
    with pytest.raises(ValidationError, match=match):
        load(SCENARIOS / name, overrides)


def test_an_orbit_sets_the_course_itself() -> None:
    with pytest.raises(ValidationError, match=r"drop \['heading_deg'\]"):
        Object(
            asset="yacht",
            range_m=200.0,
            bearing_deg=0.0,
            heading_deg=0.0,
            orbit=Orbit(period_s=60.0),
        )


def test_a_loop_too_short_for_its_waves_says_so() -> None:
    with pytest.warns(UserWarning, match="shifts the waves' frequencies"):
        load(SCENARIOS / "drifting.toml", ["sea.wind_speed_mps = 20"])
    with pytest.warns(UserWarning, match="shifts the waves' frequencies"):
        load(
            SCENARIOS / "drifting.toml",
            ["sea.swell = { height_m = 1.0, period_s = 13.0 }"],
        )


def test_a_ring_takes_a_list_of_assets_in_turn() -> None:
    ring = Targets(
        asset=["yacht", "cargo_ship"], count=3, range_m=900.0, bearing_deg=(0.0, 90.0)
    )

    assert [asset for asset, _, _ in ring.poses()] == ["yacht", "cargo_ship", "yacht"]


def test_a_ring_needs_an_asset() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        Targets(asset=[], count=3, range_m=900.0, bearing_deg=(0.0, 90.0))


def test_over_the_visibility_a_dark_target_keeps_two_percent_contrast() -> None:
    sky = Sky(visibility_km=23.0)
    assert math.exp(-sky.extinction_per_m * 23_000) == pytest.approx(0.02)
    assert Sky(visibility_km=None).extinction_per_m == 0.0


def test_the_air_takes_its_atmosphere_s_temperature_unless_set() -> None:
    assert Sky().t_air_k == lwir.SURFACE_AIR_K[lwir.ATMOSPHERE]
    assert Sky(atmosphere="tropical").t_air_k == lwir.SURFACE_AIR_K["tropical"]
    assert Sky(atmosphere="tropical", t_air_k=280.0).t_air_k == 280.0


def test_a_removed_aerosol_density_names_the_visibility() -> None:
    with pytest.raises(ValidationError, match="visibility_km"):
        Sky.model_validate({"aerosol_density": 2.0})


def test_the_sea_takes_its_atmosphere_s_temperature_unless_set() -> None:
    base = load(BASELINE).model_dump()
    sea = {k: v for k, v in base["sea"].items() if k != "t_sea_k"}
    tropical = {**base, "sea": sea, "sky": {**base["sky"], "atmosphere": "tropical"}}
    assert (
        Scenario.model_validate(tropical).sea.t_sea_k == lwir.SURFACE_SEA_K["tropical"]
    )
    by_hand = {**tropical, "sea": {**sea, "t_sea_k": 290.0}}
    assert Scenario.model_validate(by_hand).sea.t_sea_k == 290.0


def test_an_hdri_keeps_its_own_sky_over_an_inherited_one() -> None:
    photo = skies.library()["kloofendal_48d_partly_cloudy"]
    with pytest.warns(UserWarning, match="sun_elevation_deg is ignored"):
        sky = Sky.model_validate(
            {"hdri": "kloofendal_48d_partly_cloudy", "sun_elevation_deg": 5.0}
        )
    assert sky.sun_elevation_deg == photo.sun_elevation_deg


def test_an_hdri_without_a_disc_warms_no_sunlit_side() -> None:
    sky = Sky.model_validate({"hdri": "overcast_soil"})
    assert sky.sun_elevation_deg is None
    assert sky.solar_gain_k == 0.0


def test_an_unknown_hdri_names_the_library() -> None:
    with pytest.raises(ValidationError, match="overcast_soil"):
        Sky.model_validate({"hdri": "no_such_sky"})


def test_an_unknown_asset_names_the_manifest() -> None:
    with pytest.raises(ValidationError, match="yacht"):
        Object.model_validate(
            {"asset": "no_such_hull", "range_m": 1.0, "bearing_deg": 0.0}
        )


def test_the_sky_texture_needs_a_sun() -> None:
    with pytest.raises(ValidationError, match="only for an hdri"):
        Sky.model_validate({"sun_elevation_deg": None})


def test_a_dumped_hdri_sky_validates_again_without_a_warning() -> None:
    sky = Sky.model_validate({"hdri": "kloofendal_48d_partly_cloudy"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Sky.model_validate(sky.model_dump()) == sky
