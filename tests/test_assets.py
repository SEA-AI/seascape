"""Manifest parsing, and the digest gate in front of the cache."""

import hashlib
import re
import tomllib
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import pytest
from click.testing import CliRunner

from seascape import assets, cli, skies
from seascape.config import CFG_DIR

BODY = b"not really a mesh"
DIGEST = hashlib.sha256(BODY).hexdigest()


@pytest.fixture
def one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A one-entry manifest served over `file://`, cached under `tmp_path`."""
    source = tmp_path / "source.fbx"
    source.write_bytes(BODY)
    manifest = tmp_path / "assets.toml"
    manifest.write_text(
        f'[ship]\nkind = "hull"\ndescription = "A hull."\ncategory = "ship"\n'
        f'url = "{source.as_uri()}"\n'
        f'sha256 = "{DIGEST}"\nlength_m = 1.0\ndraught_m = 0.1\ntriangles = 1\n'
        'texture_px = []\nlicence = "CC0-1.0"\nattribution = "nobody"\n'
    )
    monkeypatch.setattr(assets, "MANIFEST", manifest)
    monkeypatch.setattr(assets, "CACHE", tmp_path / "cache")
    assets.manifest.cache_clear()
    yield source
    assets.manifest.cache_clear()


def test_every_object_preset_names_an_asset() -> None:
    """A preset is validated only when a scenario uses it."""
    for preset in sorted((CFG_DIR / "objects").glob("*.toml")):
        with preset.open("rb") as handle:
            assert tomllib.load(handle)["asset"] in assets.manifest(), preset


def test_every_url_ends_in_a_bare_extension() -> None:
    """The cached filename takes its suffix off the URL, so the URL must end in one."""
    for name, asset in assets.manifest().items():
        assert re.fullmatch(r"\.[a-z0-9]+", PurePosixPath(asset.url).suffix), name


def test_fetch_downloads_once(one: Path) -> None:
    first = assets.fetch("ship")
    assert first.read_bytes() == BODY
    assert first.suffix == ".fbx"

    one.unlink()
    assert assets.fetch("ship") == first


def test_fetch_rejects_a_corrupt_cache(one: Path) -> None:
    assets.fetch("ship").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="manifest says"):
        assets.fetch("ship")


def test_fetch_rejects_bytes_that_miss_the_digest(one: Path) -> None:
    one.write_bytes(b"a different mesh")
    with pytest.raises(ValueError, match="manifest says"):
        assets.fetch("ship")
    assert not (assets.CACHE / "ship.fbx").exists()
    (part,) = assets.CACHE.glob("*.part")
    assert part.read_bytes() == b"a different mesh"


def test_the_listing_names_every_mesh_and_sky() -> None:
    listing = CliRunner().invoke(cli.main, ["assets", "list"])
    assert listing.exit_code == 0
    for name in [*assets.manifest(), *skies.library()]:
        assert name in listing.output, name


def test_fetch_puts_a_mesh_s_files_where_it_names_them(one: Path) -> None:
    with assets.MANIFEST.open("a") as manifest:
        manifest.write(
            f'[ship.files."textures/skin.jpg"]\nurl = "{one.as_uri()}"\n'
            f'sha256 = "{DIGEST}"\n'
        )
    mesh = assets.fetch("ship")
    assert mesh == assets.CACHE / "ship" / "source.fbx"
    assert (mesh.parent / "textures" / "skin.jpg").read_bytes() == BODY


def test_a_file_outside_the_mesh_s_folder_is_refused() -> None:
    entry = dict(assets.manifest()["yacht"])
    entry["files"] = {"../../.bashrc": {"url": "x", "sha256": DIGEST}}
    with pytest.raises(ValueError, match="outside the mesh's folder"):
        assets.Hull.model_validate(entry)
