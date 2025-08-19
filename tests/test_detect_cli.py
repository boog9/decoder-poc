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
"""Tests for CLI alias support in detect_objects."""

from __future__ import annotations

from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace())

from src.detect_objects import parse_args  # noqa: E402


def _base_args() -> list[str]:
    """Return required base CLI arguments."""

    return ["detect", "--frames-dir", "f", "--output-json", "o"]


def test_person_conf_alias() -> None:
    """Verify that person confidence alias flags parse correctly."""

    ns = parse_args(_base_args() + ["--person-conf", "0.7"])
    assert ns.person_conf == 0.7

    ns_alias = parse_args(_base_args() + ["--conf-person", "0.8"])
    assert ns_alias.person_conf == 0.8


def test_ball_conf_alias() -> None:
    """Verify that ball confidence alias flags parse correctly."""

    ns = parse_args(_base_args() + ["--ball-conf", "0.2"])
    assert ns.ball_conf == 0.2

    ns_alias = parse_args(_base_args() + ["--conf-ball", "0.3"])
    assert ns_alias.ball_conf == 0.3
