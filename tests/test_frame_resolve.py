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
"""Tests for frame path resolution utility."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(remove=lambda *a, **k: None, add=lambda *a, **k: None)
sys.modules.setdefault("loguru", loguru_mod)

from src.track_objects import _resolve_frame_path


def test_resolve_frame_path_supports_multiple_patterns(tmp_path: Path) -> None:
    # create frame_000001.png
    p = tmp_path / "frame_000001.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    got = _resolve_frame_path(tmp_path, 1)
    assert got is not None and got.name == "frame_000001.png"

    # create 000001.jpg and check priority (either is acceptable)
    (tmp_path / "000001.jpg").write_bytes(b"\xff\xd8\xff")
    got2 = _resolve_frame_path(tmp_path, 1)
    assert got2 is not None
    assert got2.name in {"000001.jpg", "frame_000001.png"}
