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
"""Tests for :mod:`src.track_objects`."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sys
import types


class _DummyTqdm:
    def __enter__(self) -> "_DummyTqdm":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


sys.modules.setdefault("tqdm", types.ModuleType("tqdm")).tqdm = _DummyTqdm
loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
    remove=lambda *a, **k: None,
    add=lambda *a, **k: None,
)
sys.modules.setdefault("loguru", loguru_mod)
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.optimize", types.ModuleType("scipy.optimize")).linear_sum_assignment = lambda *a, **k: ([], [])
np_mod = types.ModuleType("numpy")
np_mod.array = lambda a, dtype=None: a
np_mod.asarray = lambda a, dtype=None: a
np_mod.concatenate = lambda arrs, axis=0: sum(arrs, [])
np_mod.float32 = "float32"
np_mod.int32 = int
sys.modules.setdefault("numpy", np_mod)
torch_mod = types.ModuleType("torch")
torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
torch_mod.no_grad = contextlib.nullcontext
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.cuda", torch_mod.cuda)
sys.modules.setdefault("yolox", types.ModuleType("yolox"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.track_objects as tob  # noqa: E402
from src.utils.classes import CLASS_NAME_TO_ID  # noqa: E402


def test_load_detections_skips_null_class(tmp_path: Path) -> None:
    data = [
        {"frame": "frame_000001.png", "class": None, "bbox": [0, 0, 1, 1], "score": 1.0},
        {"frame": "frame_000001.png", "class": 0, "bbox": [0, 0, 1, 1], "score": 1.0},
    ]
    path = tmp_path / "dets.json"
    path.write_text(json.dumps(data))
    frames = tob._load_detections_grouped(path, 0.0)
    assert frames == {1: {0: [{"bbox": [0.0, 0.0, 1.0, 1.0], "score": 1.0}]}}


def test_court_class_id_and_skip(tmp_path: Path) -> None:
    assert CLASS_NAME_TO_ID["tennis_court"] == 100
    data = [
        {
            "frame": "frame_000001.png",
            "class": 100,
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "score": 1.0,
        }
    ]
    path = tmp_path / "court.json"
    path.write_text(json.dumps(data))
    frames = tob._load_detections_grouped(path, 0.0)
    assert frames == {}


def test_load_detections_grouped_nested(tmp_path: Path) -> None:
    data = [
        {
            "frame": "frame_000001.png",
            "detections": [
                {"class": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
                {
                    "class": "sports ball",
                    "bbox": [5, 5, 8, 8],
                    "score": 0.8,
                },
                {
                    "class": 100,
                    "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "score": 1.0,
                },
                {"class": None, "bbox": [1, 1, 2, 2], "score": 0.5},
            ],
        }
    ]
    j = tmp_path / "nested.json"
    j.write_text(json.dumps(data))
    frames = tob._load_detections_grouped(j, 0.0)
    assert 1 in frames
    cls_map = frames[1]
    assert CLASS_NAME_TO_ID["person"] in cls_map
    assert CLASS_NAME_TO_ID["sports ball"] in cls_map
    # ensure only bbox detections retained
    assert all("bbox" in d for v in cls_map.values() for d in v)
