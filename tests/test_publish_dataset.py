import json
import subprocess
import sys
from pathlib import Path

UPLOAD = Path(".claude/skills/publish-dataset/scripts/upload.py")


def test_a_folder_without_its_scenario_is_refused(tmp_path: Path) -> None:
    # A --variants root: merged labels, no one scenario.
    info = {"commit": "abc123", "version": "0.2.0"}
    (tmp_path / "labels.json").write_text(json.dumps({"info": info}))

    command = [sys.executable, UPLOAD, "sync", tmp_path]
    refused = subprocess.run(command, check=False, capture_output=True, text=True)

    assert refused.returncode == 1
    assert f"no {tmp_path / 'scenario.toml'}" in refused.stderr
