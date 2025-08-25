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
"""Tests for small helpers used in multi-scale detection."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detect_objects import _is_box_detection


def test_is_box_detection() -> None:
    """Validate that only dicts with bbox and score are considered boxes."""

    assert _is_box_detection({"bbox": [0, 0, 1, 1], "score": 0.9, "class": 0})
    assert not _is_box_detection({"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "score": 1.0})
    assert not _is_box_detection({"bbox": [0, 0, 1, 1]})
    assert not _is_box_detection({"score": 0.5})
    assert not _is_box_detection(None)
