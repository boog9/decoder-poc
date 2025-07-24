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


class DummyImage:
    def __init__(self, path: str | None = None):
        self.path = Path(path) if path else None

    def convert(self, mode: str):
        return self

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"img")

    @staticmethod
    def open(path: str | Path):
        return DummyImage(str(path))

    @staticmethod
    def new(mode: str, size: tuple[int, int]):
        return DummyImage()


class DummyDraw:
    def __init__(self, img: DummyImage):
        self.img = img

    def rectangle(self, *args, **kwargs):
        pass


pil_mod = types.ModuleType("PIL")
pil_mod.__path__ = []  # treat as package
pil_image_mod = types.ModuleType("PIL.Image")
pil_image_mod.open = DummyImage.open
pil_image_mod.new = DummyImage.new
pil_image_mod.Image = DummyImage
pil_draw_mod = types.ModuleType("PIL.ImageDraw")
pil_draw_mod.Draw = DummyDraw
pil_mod.Image = DummyImage
pil_mod.ImageDraw = pil_draw_mod
sys.modules["PIL"] = pil_mod
sys.modules["PIL.Image"] = pil_image_mod
sys.modules["PIL.ImageDraw"] = pil_draw_mod

from PIL import Image  # type: ignore

import src.draw_roi as dr


def test_parse_args_defaults() -> None:
    args = dr.parse_args([
        "--frames-dir",
        "frames",
        "--detections-json",
        "det.json",
        "--output-dir",
        "out",
    ])
    assert args.color == "red"


def test_draw_rois_writes_files(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    img_path = frames / "img1.jpg"
    Image.new("RGB", (10, 10)).save(img_path)
    det_json = tmp_path / "det.json"
    det_json.write_text('[{"frame": "img1.jpg", "detections": [{"bbox": [1, 1, 5, 5]}]}]')

    out_dir = tmp_path / "out"
    dr.draw_rois(frames, det_json, out_dir, color="blue")

    out_img = out_dir / "img1.jpg"
    assert out_img.exists(), "Annotated image was not created"


def test_draw_rois_sanitizes_bbox(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    img_path = frames / "img1.jpg"
    Image.new("RGB", (10, 10)).save(img_path)
    det_json = tmp_path / "det.json"
    det_json.write_text('[{"frame": "img1.jpg", "detections": [{"bbox": [5, 5, 1, 1]}]}]')

    out_dir = tmp_path / "out"
    # Should not raise even though bbox coordinates are reversed
    dr.draw_rois(frames, det_json, out_dir)

    out_img = out_dir / "img1.jpg"
    assert out_img.exists(), "Annotated image was not created"
