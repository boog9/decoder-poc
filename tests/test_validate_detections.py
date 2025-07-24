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
"""Tests for :mod:`src.validate_detections`."""

from __future__ import annotations

from pathlib import Path

import src.validate_detections as vd


class DummyImage:
    def __init__(self, size: tuple[int, int] = (10, 10)) -> None:
        self.size = size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    @staticmethod
    def open(path: Path):  # type: ignore[override]
        return DummyImage()


def test_sanity_check_counts(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "f1.jpg").write_bytes(b"\x00")

    det_json = tmp_path / "det.json"
    det_json.write_text(
        "[{'frame': 'f1.jpg', 'detections': [{'bbox': [0,0,5,5], 'score': 0.2},"
        " {'bbox': [-1,0,2,2], 'score': 0.9}]}]".replace("'", '"')
    )

    monkeypatch.setattr(vd, "Image", DummyImage)
    bad, low = vd.sanity_check(det_json, frames, score_thr=0.3)
    assert bad == 1
    assert low == 1


def test_score_area_stats(tmp_path: Path) -> None:
    det_json = tmp_path / "det.json"
    det_json.write_text(
        "[{'frame': 'f', 'detections': [{'bbox': [0,0,2,2], 'score': 0.5},"
        " {'bbox': [0,0,3,3], 'score': 0.9}]}]".replace("'", '"')
    )
    scores, areas = vd.score_area_stats(det_json)
    assert len(scores) == 5
    assert len(areas) == 5
    assert abs(scores[-1] - 0.9) < 0.01
    assert abs(areas[-1] - 9.0) < 0.01
