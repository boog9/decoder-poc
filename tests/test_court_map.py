# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for :mod:`tools.map_court_by_name`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_map_court_by_name(tmp_path: Path) -> None:
    """Ensure the mapping script produces a non-empty JSON."""

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_000001.png").write_text("x", encoding="utf-8")
    (frames_dir / "000002.jpg").write_text("x", encoding="utf-8")
    (frames_dir / "frame_000003.jpeg").write_text("x", encoding="utf-8")

    court_data = [
        {"frame": 1, "meta": "a"},
        {"file": "000002.jpg", "meta": "b"},
        {"frame": 3, "meta": "c"},
    ]
    court_json = tmp_path / "court.json"
    court_json.write_text(json.dumps(court_data), encoding="utf-8")

    out_json = tmp_path / "court_by_name.json"
    env = os.environ | {
        "FRAMES_DIR": str(frames_dir),
        "COURT_JSON": str(court_json),
        "OUT_JSON": str(out_json),
    }

    script = Path(__file__).resolve().parents[1] / "tools" / "map_court_by_name.py"
    subprocess.run(
        [sys.executable, str(script)], env=env, check=True, cwd=str(tmp_path)
    )

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data, "Output JSON should not be empty"
    assert "frame_000001.png" in data
    assert "000002.jpg" in data
    assert any(k.startswith("frame_000003") for k in data)
