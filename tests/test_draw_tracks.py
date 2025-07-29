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
"""Tests for :mod:`src.draw_tracks`."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import src.draw_tracks as dt  # noqa: E402


class DummyCV2:
    def __init__(self) -> None:
        self.rectangles: list[tuple] = []
        self.written: list[Path] = []
        self.circles: list[tuple] = []

    @staticmethod
    def imread(path: str):
        return types.SimpleNamespace(shape=(1, 1, 3), tobytes=lambda: b"0")

    def rectangle(self, img, pt1, pt2, color, thickness=1):
        self.rectangles.append((pt1, pt2, color, thickness))

    def circle(self, img, center, radius, color, thickness=-1):
        self.circles.append((center, radius, color))

    def getTextSize(self, text, font, fs, thick):
        return (len(text) * 6, 10), 0

    def putText(self, img, text, org, font, fs, color, thick, lineType=None):
        pass

    def imwrite(self, path: str, img) -> bool:
        Path(path).write_bytes(b"img")
        self.written.append(Path(path))
        return True

    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16


def _setup_cv2(monkeypatch: pytest.MonkeyPatch) -> DummyCV2:
    dummy = DummyCV2()
    cv2_mod = types.ModuleType("cv2")
    cv2_mod.imread = dummy.imread
    cv2_mod.rectangle = dummy.rectangle
    cv2_mod.circle = dummy.circle
    cv2_mod.getTextSize = dummy.getTextSize
    cv2_mod.putText = dummy.putText
    cv2_mod.imwrite = dummy.imwrite
    cv2_mod.FONT_HERSHEY_SIMPLEX = 0
    cv2_mod.LINE_AA = 16
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)
    monkeypatch.setattr(dt, "cv2", cv2_mod)
    return dummy


class DummyPopen:
    def __init__(self, cmd, stdin=None):
        self.cmd = cmd
        self.stdin = types.SimpleNamespace(write=lambda b: None, close=lambda: None)
        self._out = Path(cmd[-1])

    def wait(self):
        self._out.write_bytes(b"video")


def test_visualize_tracks_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _setup_cv2(monkeypatch)
    monkeypatch.setattr(dt, "logger", types.SimpleNamespace(info=lambda *a, **k: None))

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 4):
        (frames / f"frame_{i:06d}.png").write_bytes(b"\x00")

    tracks = [
        {"frame": 1, "class": "person", "track_id": 5, "bbox": [0, 0, 2, 2], "score": 0.9},
        {"frame": 2, "class": "person", "track_id": 5, "bbox": [1, 1, 3, 3], "score": 0.8},
        {"frame": 3, "class": "person", "track_id": 7, "bbox": [2, 2, 4, 4], "score": 0.7},
    ]
    tj = tmp_path / "tracks.json"
    tj.write_text(json.dumps(tracks))

    out_dir = tmp_path / "out"
    dt.visualize_tracks(frames, tj, out_dir, None, label=True, palette="track", thickness=2, fps=30.0)

    assert len(dummy.written) == 3
    for p in dummy.written:
        assert p.exists()


def test_visualize_tracks_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _setup_cv2(monkeypatch)
    monkeypatch.setattr(dt, "logger", types.SimpleNamespace(info=lambda *a, **k: None))
    monkeypatch.setattr(dt.subprocess, "Popen", DummyPopen)

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 4):
        (frames / f"frame_{i:06d}.png").write_bytes(b"\x00")

    tracks = [
        {"frame": 1, "class": "person", "track_id": 5, "bbox": [0, 0, 2, 2], "score": 0.9},
        {"frame": 2, "class": "person", "track_id": 5, "bbox": [1, 1, 3, 3], "score": 0.8},
    ]
    tj = tmp_path / "tracks.json"
    tj.write_text(json.dumps(tracks))

    out_video = tmp_path / "out.mp4"
    dt.visualize_tracks(frames, tj, None, out_video, palette="track", fps=25.0)

    assert out_video.exists() and out_video.stat().st_size > 0


def test_frame_index_shift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _setup_cv2(monkeypatch)
    monkeypatch.setattr(
        dt,
        "logger",
        types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 3):
        (frames / f"frame_{i:06d}.png").write_bytes(b"\x00")

    tracks = [
        {"frame": 0, "track_id": 1, "bbox": [0, 0, 1, 1]},
        {"frame": 1, "track_id": 1, "bbox": [1, 1, 2, 2]},
    ]
    tj = tmp_path / "tracks.json"
    tj.write_text(json.dumps(tracks))

    out_dir = tmp_path / "out"
    dt.visualize_tracks(frames, tj, out_dir, None)

    assert len(dummy.rectangles) == 2
