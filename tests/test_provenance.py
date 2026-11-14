import json

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
