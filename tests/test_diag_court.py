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
"""Tests for :mod:`tools.diag_court`."""

from __future__ import annotations

from pathlib import Path

from tools.diag_court import analyze


def test_analyze_identity(tmp_path: Path) -> None:
    data = [
        {
            "frame": "f1",
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    ]
    entries, rmses, dets = analyze(data)
    assert entries[0][0] == "f1"
    assert rmses[0] == 0.0
    assert dets[0] == 1.0
