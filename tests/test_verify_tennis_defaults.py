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
"""Tests for :mod:`tools.verify_tennis_defaults`."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verify_tennis_defaults import _metrics  # noqa: E402


def test_metrics_computation(tmp_path: Path) -> None:
    data = [
        {"frame": 1, "class": 0, "track_id": 1},
        {"frame": 1, "class": 32, "track_id": 5},
        {"frame": 2, "class": 32, "track_id": 5},
        {"frame": 2, "class": 0, "track_id": 2},
    ]
    players, frac_ball, avg_ball = _metrics(data)
    assert players == 2
    assert frac_ball == 1.0
    assert avg_ball == 2.0
