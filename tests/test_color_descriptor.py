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
"""Tests for HSV colour descriptor utilities."""

from __future__ import annotations

from pathlib import Path
import types
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(remove=lambda *a, **k: None, add=lambda *a, **k: None)
sys.modules.setdefault("loguru", loguru_mod)

torch_mod = types.ModuleType("torch")
torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_mod)

PIL = pytest.importorskip("PIL")
from PIL import Image

from src.track_objects import _hsv_hist, _reuse_id


def test_hsv_hist_top_half() -> None:
    img = Image.new("RGB", (10, 10), (0, 0, 255))
    for y in range(5):
        for x in range(10):
            img.putpixel((x, y), (255, 0, 0))
    hist = _hsv_hist(img, [0, 0, 10, 10])
    assert abs(sum(hist) - 1.0) < 1e-6
    red_bin = sum(hist[:64])
    assert red_bin > 0.8


def test_reuse_id_color_weight() -> None:
    hist1 = [1.0] + [0.0] * 511
    cache = [{"id": 1, "bbox": [0, 0, 10, 10], "frame": 0, "hist": hist1}]
    hist2 = hist1.copy()
    assert _reuse_id(cache, [0, 0, 10, 10], hist2, 1, 10, 0.5) == 1
    cache = [{"id": 1, "bbox": [0, 0, 10, 10], "frame": 0, "hist": hist1}]
    hist3 = [0.0] * 512
    hist3[100] = 1.0
    assert _reuse_id(cache, [0, 0, 10, 10], hist3, 1, 10, 0.5) is None


def test_hsv_hist_handles_rgba() -> None:
    img = Image.new("RGBA", (8, 8), (0, 255, 0, 128))
    img2 = img.convert("RGB")
    hist = _hsv_hist(img2, [0, 0, 8, 8])
    assert abs(sum(hist) - 1.0) < 1e-6
