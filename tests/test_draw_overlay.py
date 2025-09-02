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
"""Tests for :mod:`src.draw_overlay`."""

from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.draw_overlay import draw_rois, load_roi_json


def test_load_roi_json_valid(tmp_path: Path) -> None:
    data = {
        "polygons": [
            {
                "name": "A",
                "points": [[0, 0], [10, 0], [10, 10]],
                "color": "#FF0000",
                "thickness": 3,
                "fill_alpha": 0.5,
            }
        ]
    }
    json_path = tmp_path / "roi.json"
    json_path.write_text(__import__("json").dumps(data))

    rois = load_roi_json(json_path)
    assert len(rois) == 1
    roi = rois[0]
    assert roi["color"] == (0, 0, 255)
    assert roi["thickness"] == 3
    assert roi["fill_alpha"] == 0.5


def test_draw_rois_draws() -> None:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rois = [
        {
            "name": "square",
            "points": np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.int32),
            "color": (0, 255, 0),
            "thickness": 2,
            "fill_alpha": 1.0,
        }
    ]
    draw_rois(img, rois)
    # check a pixel inside the square changed to green
    assert img[50, 50, 1] == 255
