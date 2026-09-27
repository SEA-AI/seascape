"""Loader rules and the shipped presets. No Blender."""

import json
import tomllib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from seascape import lwir
from seascape.config import (
    CFG_DIR,
    Band,
    Camera,
    Object,
    Orbit,
    Outputs,
    Pod,
    Rig,
    Samples,
    Scenario,
    Sea,
    load,
)

SCENARIOS = Path(__file__).parents[1] / "scenarios"
BASELINE = SCENARIOS / "baseline.toml"
SCHEMA = Path(__file__).parents[1] / "schema" / "scenario.json"


# One camera in one pod; overriding `rig.pods` replaces the whole list.
ONE_POD = '[[rig.pods]]\nname = "bow"\nyaw_deg = 0.0\n\n[[rig.pods.cameras]]\n'


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
    kinds = [mount.camera.kind for mount in baseline.rig.mounts]

    assert kinds == ["eo", "ir"]


def test_the_baseline_carries_no_scenario_a_variant_would_inherit(baseline) -> None:
    """`extends` copies whatever is here; a variant never asked for a ring."""
    assert (baseline.ownship.asset, baseline.targets) == (None, None)


def test_a_preset_supplies_optics_and_the_block_supplies_the_mount(twin_pod) -> None:
    eo, ir = twin_pod.rig.mounts[0], twin_pod.rig.mounts[-1]

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
    assert (eo.pod.name, eo.nominal_bearing_deg) == ("port", -100.0)


def test_a_nominal_bearing_is_its_pod_plus_its_fan(twin_pod) -> None:
    port = twin_pod.rig.pods[0]

    assert port.yaw_deg == -60.0
    assert [camera.yaw_deg for camera in port.cameras] == [-40.0, 0.0, 40.0, 50.0]
    assert [mount.nominal_bearing_deg for mount in twin_pod.rig.mounts][:4] == [
        -100.0,
        -60.0,
        -20.0,
        -10.0,
    ]


def test_the_installed_rig_takes_its_height_from_the_preset(twin_pod) -> None:
    assert twin_pod.rig.height_m == 51.8


def test_samples_covers_every_band() -> None:
    """`scene._output` reads this with `getattr(samples, band)`, so a band added to the
    Literal without a field here fails mid-build rather than in validation."""
    assert set(Samples.model_fields) == set(get_args(Band.__value__))


def test_an_override_is_the_toml_line_it_would_be_written_as(baseline) -> None:
    scenario = load(BASELINE, ["rig.pitch_deg = -5"])

    assert scenario.rig.pitch_deg == -5.0
    assert scenario.rig.height_m == baseline.rig.height_m


def test_an_override_resolves_presets_like_a_line_in_the_file() -> None:
    scenario = load(SCENARIOS / "twin-pod.toml", ['rig.pods = [{ preset = "port" }]'])
    assert [pod.name for pod in scenario.rig.pods] == ["port"]
    assert len(scenario.rig.mounts) == 4


def test_an_override_merges_a_table_rather_than_replacing_it() -> None:
    scenario = load(BASELINE, ["outputs.samples.eo = 8"])

    assert (scenario.outputs.samples.eo, scenario.outputs.samples.ir) == (8, 64)


def test_overrides_apply_in_order() -> None:
    scenario = load(BASELINE, ["rig.pitch_deg = -5", "rig.pitch_deg = -10"])

    assert scenario.rig.pitch_deg == -10.0


@pytest.mark.parametrize(
    ("override", "error"),
    [
        pytest.param("rig.tlit_deg = -5", ValidationError, id="misspelt-key"),
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
            ONE_POD + 'preset = "eo_4k_49deg"\nhfov_deg = 10.0\n',
        )
    )
    camera = scenario.rig.mounts[0].camera
    assert camera.hfov_deg == 10.0
    assert camera.width_px == 3840  # untouched by the block


def test_a_preset_outranks_an_inherited_value(tmp_path) -> None:
    """Expanding after the parent merge inverts this, and nothing else notices."""
    (tmp_path / "single.toml").write_text(
        'height_m = 2.0\n\n[[pods]]\nname = "bow"\nyaw_deg = 0.0\n\n'
        '[[pods.cameras]]\npreset = "ir_vga_24deg"\n'
    )
    scenario = load(variant(tmp_path, '[rig]\npreset = "./single.toml"\n'))
    assert scenario.rig.height_m == 2.0
    assert [mount.camera.kind for mount in scenario.rig.mounts] == ["ir"]


def test_tables_merge_and_lists_replace(tmp_path, baseline) -> None:
    scenario = load(
        variant(
            tmp_path,
            "[sea]\nwind_speed_mps = 3.0\n\n" + ONE_POD + 'preset = "ir_vga_24deg"\n',
        )
    )
    assert scenario.sea.wind_speed_mps == 3.0
    assert scenario.sea.t_sea_k == baseline.sea.t_sea_k
    assert scenario.rig.height_m == baseline.rig.height_m
    assert len(scenario.rig.mounts) == 1


@pytest.mark.parametrize("name", ["./mine.toml", "mine.toml"])
def test_preset_can_be_a_path(tmp_path, name) -> None:
    (tmp_path / "mine.toml").write_text('kind = "eo"\nhfov_deg = 12.0\n')
    scenario = load(
        variant(
            tmp_path,
            ONE_POD + f'preset = "{name}"\nwidth_px = 1920\nheight_px = 1080\n',
        )
    )
    assert scenario.rig.mounts[0].camera.hfov_deg == 12.0


@pytest.mark.parametrize(
    ("preset", "error", "match"),
    [
        ("not_a_real_preset", FileNotFoundError, "not_a_real_preset.toml"),
        (12, TypeError, "must be a string"),
    ],
)
def test_bad_presets_fail_loudly(tmp_path, preset, error, match) -> None:
    with pytest.raises(error, match=match):
        load(variant(tmp_path, f"[rig]\npreset = {json.dumps(preset)}\n"))


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
        load(variant(tmp_path, "[rig]\nheight_metres = 22.0\n"))


@pytest.mark.parametrize("t_sea_k", [260.0, 400.0])
def test_sea_temperature_is_bounded_at_the_config_boundary(tmp_path, t_sea_k) -> None:
    with pytest.raises(ValidationError, match="t_sea_k"):
        load(variant(tmp_path, f"[sea]\nt_sea_k = {t_sea_k}\n"))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_numbers_are_rejected(tmp_path, value) -> None:
    """`yaw_deg` carries no bound, so nothing else would catch one."""
    with pytest.raises(ValidationError, match="yaw_deg"):
        load(
            variant(
                tmp_path,
                f'[[rig.pods]]\nname = "bow"\nyaw_deg = {value}\n\n'
                '[[rig.pods.cameras]]\npreset = "ir_vga_24deg"\n',
            )
        )


def test_committed_schema_matches_the_models() -> None:
    """Editors validate against the committed file; a stale one is worse than none."""
    assert json.loads(SCHEMA.read_text()) == Scenario.model_json_schema(), (
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


def test_baseline_points_at_the_committed_schema() -> None:
    """The `#:schema` line is a comment, so nothing else would ever notice it rot."""
    line = BASELINE.read_text().splitlines()[0]
    assert line.startswith("#:schema ")
    assert (BASELINE.parent / line.removeprefix("#:schema ")).resolve() == SCHEMA


def test_every_shipped_preset_parses() -> None:
    """A preset directory is named after the block it serves, and holds valid TOML."""
    presets = sorted(CFG_DIR.rglob("*.toml"))
    assert {path.parent.name for path in presets} == {
        "rig",
        "pods",
        "cameras",
        "objects",
    }
    for preset in presets:
        tomllib.load(preset.open("rb"))


@pytest.mark.parametrize("exposure_ev", [-50.0, 100.0])
def test_an_exposure_blender_would_clamp_is_rejected(exposure_ev: float) -> None:
    with pytest.raises(ValidationError, match="exposure_ev"):
        Outputs(exposure_ev=exposure_ev)


@pytest.mark.parametrize("name", ["../escaped", "/tmp/absolute", "sub/dir"])
def test_a_pod_cannot_be_path_text(name: str) -> None:
    with pytest.raises(ValidationError, match="name"):
        Pod(
            name=name,
            yaw_deg=0.0,
            cameras=[Camera(kind="eo", hfov_deg=60.0, width_px=8, height_px=8)],
        )


def test_two_cameras_cannot_share_a_name() -> None:
    """They would share a datablock and overwrite each other's render."""
    camera = {"kind": "eo", "hfov_deg": 45.0, "width_px": 8, "height_px": 8}
    with pytest.raises(ValidationError, match="share a name"):
        Rig(
            height_m=12.0,
            pods=[
                Pod(name="bow", yaw_deg=0.0, cameras=[Camera(**camera, name="fwd")]),
                Pod(name="mast", yaw_deg=0.0, cameras=[Camera(**camera, name="fwd")]),
            ],
        )


def test_two_pods_cannot_share_a_name() -> None:
    """Mount names miss it: cameras of different bands still differ."""
    eo = Camera(kind="eo", hfov_deg=45.0, width_px=8, height_px=8)
    ir = Camera(kind="ir", hfov_deg=24.0, width_px=8, height_px=8)

    with pytest.raises(ValidationError, match="two pods share a name"):
        Rig(
            height_m=12.0,
            pods=[
                Pod(name="port", yaw_deg=-60.0, offset_x_m=-20.0, cameras=[eo]),
                Pod(name="port", yaw_deg=+60.0, offset_x_m=+20.0, cameras=[ir]),
            ],
        )


def test_an_authored_name_wins_over_the_derived_one() -> None:
    named = Camera(kind="eo", hfov_deg=45.0, width_px=8, height_px=8, name="EO_PORT_C")
    plain = Camera(kind="eo", hfov_deg=45.0, width_px=8, height_px=8)

    rig = Rig(
        height_m=12.0, pods=[Pod(name="port", yaw_deg=0.0, cameras=[named, plain])]
    )

    assert [mount.name for mount in rig.mounts] == ["EO_PORT_C", "port_eo_1"]


def test_the_same_camera_on_two_pods_is_fine() -> None:
    camera = Camera(kind="eo", yaw_deg=0.0, hfov_deg=45.0, width_px=8, height_px=8)

    rig = Rig(
        height_m=12.0,
        pods=[
            Pod(name="port", yaw_deg=-60.0, cameras=[camera]),
            Pod(name="starboard", yaw_deg=+60.0, cameras=[camera]),
        ],
    )

    assert [mount.name for mount in rig.mounts] == ["port_eo_0", "starboard_eo_0"]


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
