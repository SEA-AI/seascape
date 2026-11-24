import json
import subprocess
from pathlib import Path

import pytest
from PIL import ExifTags

from seascape.outputs import provenance


def test_the_exif_names_how_a_picture_was_made() -> None:
    exif = provenance.exif("open-sea", 7, ["sea.wind_speed_mps = 3"], "bow_eo")
    assert json.loads(exif[ExifTags.Base.ImageDescription]) == {
        "scenario": "open-sea",
        "seed": 7,
        "overrides": ["sea.wind_speed_mps = 3"],
        "camera": "bow_eo",
    }


def test_a_long_request_leaves_its_overrides_out_of_the_exif() -> None:
    exif = provenance.exif("open-sea", 7, ['"' * 1000] * 50)
    assert json.loads(exif[ExifTags.Base.ImageDescription]) == {
        "scenario": "open-sea",
        "seed": 7,
        "overrides_omitted": True,
    }
    assert len(exif.tobytes()) < 65_535


def test_an_install_without_git_names_the_image_s_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provenance, "__file__", str(tmp_path / "provenance.py"))
    monkeypatch.setenv("SEASCAPE_COMMIT", "abc123")
    assert provenance.commit() == "abc123"
    monkeypatch.setenv("SEASCAPE_COMMIT", "")
    assert provenance.commit() is None


def test_a_git_hook_s_repository_does_not_stand_in_for_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here = provenance.commit()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / ".git" / "index"))

    assert provenance.commit() == here
