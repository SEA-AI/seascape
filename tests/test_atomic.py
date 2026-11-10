from pathlib import Path

import pytest

from seascape import atomic


def test_a_clean_block_replaces_the_file_and_leaves_no_scratch(tmp_path: Path) -> None:
    path = tmp_path / "cache.bin"
    path.write_bytes(b"old")
    with atomic.replacing(path) as part:
        part.write_bytes(b"new")
        assert path.read_bytes() == b"old"
    assert path.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [path]


def test_a_failed_block_keeps_the_file_and_leaves_no_scratch(tmp_path: Path) -> None:
    path = tmp_path / "cache.bin"
    path.write_bytes(b"old")
    with pytest.raises(OSError, match="disk full"), atomic.replacing(path) as part:
        part.write_bytes(b"half")
        raise OSError("disk full")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def test_the_scratch_file_keeps_the_suffix(tmp_path: Path) -> None:
    with atomic.replacing(tmp_path / "mask.png") as part:
        part.write_bytes(b"x")
        assert part.suffix == ".png"
