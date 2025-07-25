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
cv2_dummy.putText = lambda img, text, org, font, fs, color, thick, lineType=None: None
cv2_dummy.getTextSize = lambda text, font, fs, thick: ((len(text)*6, 10), 0)
cv2_dummy.imwrite = lambda path, img: True
cv2_dummy.FONT_HERSHEY_SIMPLEX = 0
cv2_dummy.LINE_AA = 16
sys.modules.setdefault("cv2", cv2_dummy)

import src.draw_roi as dr  # noqa: E402


class DummyCV2:
    def __init__(self) -> None:
        self.rectangles: list[tuple] = []
        self.written: list[Path] = []
        self.texts: list[tuple] = []

    @staticmethod
    def imread(path: str):
        return [[0]]

    def rectangle(self, img, pt1, pt2, color, thickness=1):
        self.rectangles.append((pt1, pt2, color, thickness))

    def putText(self, img, text, org, font, fs, color, thick, lineType=None):
        self.texts.append((text, org, color))

    def getTextSize(self, text, font, fs, thick):
        return (len(text) * 6, 10), 0

    def imwrite(self, path: str, img) -> bool:
        Path(path).write_bytes(b"img")
        self.written.append(Path(path))
        return True


def _setup_cv2(monkeypatch: pytest.MonkeyPatch) -> DummyCV2:
    dummy = DummyCV2()
    cv2_mod = types.ModuleType("cv2")
    cv2_mod.imread = dummy.imread
    cv2_mod.rectangle = dummy.rectangle
    cv2_mod.putText = dummy.putText
    cv2_mod.getTextSize = dummy.getTextSize
    cv2_mod.imwrite = dummy.imwrite
    cv2_mod.FONT_HERSHEY_SIMPLEX = 0
    cv2_mod.LINE_AA = 16
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)
    monkeypatch.setattr(dr, "cv2", cv2_mod)
    return dummy


def test_parse_args_defaults() -> None:
    args = dr.parse_args(
        ["--frames-dir", "frames", "--detections-json", "det.json", "--output-dir", "out"]
    )
    assert args.img_size == 640
    assert args.color == "red"
    assert args.label is False


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


def test_draw_rois_label_coloring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = _setup_cv2(monkeypatch)
    monkeypatch.setattr(dr, "_preprocess_params", lambda img, size: (1.0, 0.0, 0.0, 10, 10))

    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img.jpg").write_bytes(b"\x00")

    det_json = tmp_path / "det.json"
    det_json.write_text(
        '[{"frame": "img.jpg", "detections": [{"bbox": [1, 1, 5, 5], "class": 0, "score": 0.9}]}]'
    )

    out_dir = tmp_path / "out"
    dr.draw_rois(frames, det_json, out_dir, 640, label=True)

    assert dummy.rectangles
    expected = dr._label_color(0)
    assert dummy.rectangles[0][2] == expected
    assert dummy.texts
    assert "person:90.0%" in dummy.texts[0][0]
