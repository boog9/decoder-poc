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
"""Tests for :mod:`src.draw_roi`."""

from __future__ import annotations

from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

cv2_dummy = types.ModuleType("cv2")
cv2_dummy.imread = lambda path: [[0]]
cv2_dummy.rectangle = lambda img, pt1, pt2, color, thickness=1: None
cv2_dummy.imwrite = lambda path, img: True
sys.modules.setdefault("cv2", cv2_dummy)

import src.draw_roi as dr  # noqa: E402


class DummyCV2:
    def __init__(self) -> None:
        self.rectangles: list[tuple] = []
        self.written: list[Path] = []

    @staticmethod
    def imread(path: str):
        return [[0]]

    def rectangle(self, img, pt1, pt2, color, thickness=1):
        self.rectangles.append((pt1, pt2, color, thickness))

    def imwrite(self, path: str, img) -> bool:
        Path(path).write_bytes(b"img")
        self.written.append(Path(path))
        return True


def _setup_cv2(monkeypatch: pytest.MonkeyPatch) -> DummyCV2:
    dummy = DummyCV2()
    cv2_mod = types.ModuleType("cv2")
    cv2_mod.imread = dummy.imread
    cv2_mod.rectangle = dummy.rectangle
    cv2_mod.imwrite = dummy.imwrite
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)
    monkeypatch.setattr(dr, "cv2", cv2_mod)
    return dummy


def test_parse_args_defaults() -> None:
    args = dr.parse_args(
        ["--frames-dir", "frames", "--detections-json", "det.json", "--output-dir", "out"]
    )
    assert args.img_size == 640
    assert args.color == "red"


def test_draw_rois_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_cv2 = _setup_cv2(monkeypatch)

    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img.jpg").write_bytes(b"\x00")

    det_json = tmp_path / "det.json"
    det_json.write_text('[{"frame": "img.jpg", "detections": [{"bbox": [1, 1, 5, 5]}]}]')

    out_dir = tmp_path / "out"
    dr.draw_rois(frames, det_json, out_dir, 640, color="blue")

    out_img = out_dir / "img.jpg"
    assert out_img.exists()
    assert dummy_cv2.rectangles


def test_draw_rois_sanitizes_bbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_cv2(monkeypatch)

    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img.jpg").write_bytes(b"\x00")

    det_json = tmp_path / "det.json"
    det_json.write_text('[{"frame": "img.jpg", "detections": [{"bbox": [5, 5, 1, 1]}]}]')

    out_dir = tmp_path / "out"
    dr.draw_rois(frames, det_json, out_dir, 640)

    assert (out_dir / "img.jpg").exists()


def test_backproject_bbox() -> None:
    bbox = dr._backproject_bbox((1.0, 1.0, 5.0, 5.0), 0.5, 1.0, 1.0, 10, 10)
    assert bbox == (0, 0, 8, 8)
