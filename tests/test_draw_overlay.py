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

from __future__ import annotations

from pathlib import Path
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

cv2_dummy = types.ModuleType('cv2')
cv2_dummy.imread = lambda *a, **k: None
cv2_dummy.imwrite = lambda *a, **k: True
cv2_dummy.rectangle = lambda *a, **k: None
cv2_dummy.putText = lambda *a, **k: None
cv2_dummy.FONT_HERSHEY_SIMPLEX = 0
cv2_dummy.LINE_AA = 16
sys.modules.setdefault('cv2', cv2_dummy)

import pytest

import src.draw_overlay as dov  # noqa: E402


def test_load_detections_supports_schemes(tmp_path: Path) -> None:
    nested = tmp_path / "nested.json"
    nested.write_text(
        '[{"frame": "frame_000001.png", "detections": [{"bbox": [0,0,1,1]}]}]'
    )
    flat = tmp_path / "flat.json"
    flat.write_text('[{"frame": "frame_000001.png", "bbox": [0,0,1,1]}]')

    det_nested = dov._load_detections(nested)
    det_flat = dov._load_detections(flat)
    assert set(det_nested.keys()) == {'frame_000001.png'}
    assert set(det_flat.keys()) == {'frame_000001.png'}
    assert len(det_nested['frame_000001.png']) == 1
    assert len(det_flat['frame_000001.png']) == 1


def test_resolve_frame_path_by_various_keys(tmp_path: Path) -> None:
    frame = tmp_path / 'frame_000001.png'
    frame.write_bytes(b'0')

    p1, idx1 = dov._resolve_frame_path(tmp_path, 'frame_000001.png')
    p2, idx2 = dov._resolve_frame_path(tmp_path, 1)
    p3, idx3 = dov._resolve_frame_path(tmp_path, 'foo001bar')
    assert p1 == frame and idx1 == 1
    assert p2 == frame and idx2 == 1
    assert p3 == frame and idx3 == 1


def test_load_tracks_populates_track_id(tmp_path: Path) -> None:
    flat = tmp_path / "flat.json"
    flat.write_text('[{"frame":"frame_000001.png","bbox":[0,0,10,10]}]')
    fm = dov._load_tracks(flat)
    assert set(fm.keys()) == {"frame_000001.png"}
    assert "track_id" in fm["frame_000001.png"][0]
    assert fm["frame_000001.png"][0]["track_id"] is None


def test_parse_only_class_mixed() -> None:
    names, ids = dov._parse_class_filter("person, 32, sports ball,100")
    assert "person" in names and "sports ball" in names
    assert 32 in ids and 100 in ids


def test_validate_track_ids_raises(tmp_path: Path) -> None:
    tj = tmp_path / "tracks.json"
    tj.write_text(
        '[{"frame":1,"bbox":[0,0,1,1],"track_id":1},{"frame":2,"bbox":[0,0,1,1],"track_id":2}]'
    )
    fm = dov._load_tracks(tj)
    with pytest.raises(ValueError):
        dov._validate_track_ids(fm)


def test_track_color_stability() -> None:
    dov.PALETTE_SEED = 42
    assert dov._track_color(7) == dov._track_color(7)


def test_track_color_consistent_over_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 6):
        (frames / f"frame_{i:06d}.png").write_bytes(b"0")
    tracks = []
    for i in range(1, 6):
        tracks.append({"frame": i, "bbox": [0, 0, 2, 2], "score": 0.9, "class": 0, "track_id": 5})
        tracks.append({"frame": i, "bbox": [2, 2, 4, 4], "score": 0.8, "class": 0, "track_id": 7})
    tj = tmp_path / "t.json"
    import json
    tj.write_text(json.dumps(tracks))

    expected5 = dov._track_color(5)
    expected7 = dov._track_color(7)

    class DummyCV2:
        def __init__(self) -> None:
            self.rect_colors: list[tuple[int, int, int]] = []

        def imread(self, path: str, flag=None):
            return types.SimpleNamespace(shape=(10, 10, 3))

        def rectangle(self, img, pt1, pt2, color, thickness):
            self.rect_colors.append(color)

        def putText(self, *a, **k):
            pass

        def imwrite(self, path: str, img) -> bool:
            return True

        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        IMREAD_COLOR = 1

    dummy = DummyCV2()
    monkeypatch.setattr(dov, "cv2", dummy)

    fm = dov._load_tracks(tj)
    out = tmp_path / "out"
    res = dov._draw_overlay(
        frames,
        fm,
        out,
        False,
        False,
        0.0,
        set(),
        set(),
        1,
        0.5,
        0,
        -1,
        0,
        "track",
    )
    assert res == 5
    assert dummy.rect_colors[0] == expected5
    assert dummy.rect_colors[1] == expected7
    assert dummy.rect_colors[2] == expected5
