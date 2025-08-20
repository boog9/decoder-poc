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
"""Tests for :mod:`src.court_detector`."""

from __future__ import annotations

from pathlib import Path
import sys
import types
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.court_detector as cd


def test_detect_court_returns_polygon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(2):
        (frames / f"frame_{i+1:06d}.png").write_bytes(b"")

    dummy_img = types.SimpleNamespace(size=(100, 60))

    class DummyCtx:
        def __enter__(self):
            return dummy_img

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cd.Image, "open", lambda p: DummyCtx())

    result = cd.detect_court(frames)
    assert len(result) == 2
    names = {r["frame"] for r in result}
    assert all(f"frame_{i:06d}.png" in names for i in range(1, 3))
    assert all("polygon" in r and len(r["polygon"]) >= 4 for r in result)
    assert all(r.get("class") == 100 for r in result)
