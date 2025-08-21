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
"""Tests for loading ROI from court.json style list."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import types

# Stub for shapely.geometry to avoid dependency on Shapely


class DummyPoint:
    """Simple point representation."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class DummyPolygon:
    """Minimal polygon supporting bounds and containment."""

    def __init__(self, pts: list[list[float]]) -> None:
        self.pts = pts
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.x0, self.x1 = min(xs), max(xs)
        self.y0, self.y1 = min(ys), max(ys)
        self.exterior = types.SimpleNamespace(coords=pts)

    def buffer(self, margin: float) -> "DummyPolygon":
        return DummyPolygon(
            [
                [self.x0 - margin, self.y0 - margin],
                [self.x1 + margin, self.y0 - margin],
                [self.x1 + margin, self.y1 + margin],
                [self.x0 - margin, self.y1 + margin],
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

from shapely.geometry import Polygon  # type: ignore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detect_objects import _load_roi  # noqa: E402


def test_load_roi_from_list(tmp_path: Path) -> None:
    data = [
        {"frame": "frame_000001.png", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"frame": "frame_000002.png", "polygon": [[1, 1], [9, 1], [9, 9], [1, 9]]},
    ]
    p = tmp_path / "court.json"
    p.write_text(json.dumps(data))
    poly = _load_roi(p)
    assert isinstance(poly, Polygon)
    assert list(poly.exterior.coords)[0][0] == 0
