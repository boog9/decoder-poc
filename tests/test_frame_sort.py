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
"""Tests for numeric frame sorting utilities."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cv2_stub = types.SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, LINE_AA=0)
sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))

from src.utils.draw_helpers import load_frames  # noqa: E402
import src.track_objects as tobj  # noqa: E402
from src.track_objects import _load_detections_grouped  # noqa: E402
tobj.logger = types.SimpleNamespace(warning=lambda *a, **k: None)


def test_load_frames_numeric(tmp_path: Path) -> None:
    (tmp_path / "frame_10.png").write_text("a")
    (tmp_path / "frame_2.png").write_text("a")
    (tmp_path / "frame_1.png").write_text("a")
    frames = load_frames(tmp_path)
    assert [f.name for f in frames] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_detection_loading_warns(monkeypatch, tmp_path: Path, caplog) -> None:
    data = [
        {"frame": "frame_10.png", "class": 0, "bbox": [0, 0, 1, 1], "score": 0.9},
        {"frame": "frame_2.png", "class": 0, "bbox": [0, 0, 1, 1], "score": 0.9},
    ]
    jpath = tmp_path / "dets.json"
    jpath.write_text(json.dumps(data))
    grouped = _load_detections_grouped(jpath, 0.1)
    assert list(grouped.keys()) == [2, 10]
