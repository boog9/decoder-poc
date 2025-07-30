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
"""Ensure JSON files use integer class IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(remove=lambda *a, **k: None, add=lambda *a, **k: None)
sys.modules.setdefault("loguru", loguru_mod)

from src.utils.classes import CLASS_NAME_TO_ID


@pytest.mark.parametrize("fname", ["detections.json", "tracks.json"])
def test_class_fields_are_int(tmp_path: Path, fname: str) -> None:
    data = [
        {"frame": 1, "class": CLASS_NAME_TO_ID["person"], "bbox": [0, 0, 1, 1]},
        {"frame": 2, "class": CLASS_NAME_TO_ID["ball"], "bbox": [0, 0, 1, 1]},
    ]
    path = tmp_path / fname
    path.write_text(json.dumps(data))

    with path.open() as fh:
        loaded = json.load(fh)

    assert all(isinstance(d.get("class"), int) for d in loaded)
    assert {d["class"] for d in loaded} <= {0, 32}
