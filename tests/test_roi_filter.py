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
"""Tests for ROI filtering during detection."""

from __future__ import annotations

from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DummyPoint:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class DummyPolygon:
    def __init__(self, pts: list[tuple[float, float]]) -> None:
        xs, ys = zip(*pts)
        self.x0, self.y0 = min(xs), min(ys)
        self.x1, self.y1 = max(xs), max(ys)

    def buffer(self, margin: float) -> "DummyPolygon":
        return DummyPolygon(
            [
                (self.x0 - margin, self.y0 - margin),
                (self.x1 + margin, self.y1 + margin),
            ]
        )

    def contains(self, point: DummyPoint) -> bool:
        return self.x0 <= point.x <= self.x1 and self.y0 <= point.y <= self.y1

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


geometry = types.SimpleNamespace(Polygon=DummyPolygon, Point=DummyPoint)
shapely_stub = types.SimpleNamespace(geometry=geometry)
sys.modules.setdefault("shapely", shapely_stub)
sys.modules.setdefault("shapely.geometry", geometry)
sys.modules.setdefault("torch", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.SimpleNamespace(ndarray=object))
from shapely.geometry import Polygon  # type: ignore  # noqa: E402
from src.detect_objects import _filter_by_roi  # noqa: E402


def test_filter_by_roi_keeps_inside() -> None:
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    entries = [
        {
            "frame": "frame_1.png",
            "detections": [
                {"bbox": [10, 10, 20, 20], "class": 0},
                {"bbox": [150, 150, 160, 160], "class": 0},
            ],
        }
    ]
    filtered = _filter_by_roi(entries, poly, 0, False)
    dets = filtered[0]["detections"]
    assert len(dets) == 1 and dets[0]["bbox"] == [10, 10, 20, 20]
