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
from typing import List
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

cv2_dummy = types.ModuleType('cv2')
cv2_dummy.imread = lambda *a, **k: None
cv2_dummy.imwrite = lambda *a, **k: True
cv2_dummy.rectangle = lambda *a, **k: None
cv2_dummy.putText = lambda *a, **k: None
cv2_dummy.polylines = lambda *a, **k: None
cv2_dummy.FONT_HERSHEY_SIMPLEX = 0
cv2_dummy.LINE_AA = 16
sys.modules.setdefault('cv2', cv2_dummy)

# Import canonical court model for tests without heavy dependencies
services_pkg = types.ModuleType("services")
court_pkg = types.ModuleType("services.court_detector")
court_pkg.__path__ = [
    str(Path(__file__).resolve().parents[1] / "services" / "court_detector")
]
services_pkg.court_detector = court_pkg
sys.modules.setdefault("services", services_pkg)
sys.modules.setdefault("services.court_detector", court_pkg)
from services.court_detector.court_reference_tcd import CANONICAL_LINES

# Stub for shapely.geometry to avoid heavy dependency
class DummyPolygon:
    def __init__(self, pts: list[list[float]]) -> None:
        self.exterior = types.SimpleNamespace(coords=pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))
        self.area = (self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1])


geometry = types.SimpleNamespace(Polygon=DummyPolygon)
shapely_stub = types.SimpleNamespace(geometry=geometry)
sys.modules.setdefault("shapely", shapely_stub)
sys.modules.setdefault("shapely.geometry", geometry)
from shapely.geometry import Polygon  # type: ignore  # noqa: E402

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


def test_load_roi_from_dict(tmp_path: Path) -> None:
    roi = tmp_path / "roi.json"
    roi.write_text('{"polygon": [[0,0], [1,0], [1,1], [0,1]], "lines": {"l":[[0,0],[1,1]]}}')
    poly, lines = dov._load_roi(roi)
    assert isinstance(poly, Polygon)
    assert tuple(poly.exterior.coords[0]) == (0, 0)
    assert "l" in lines


def test_load_roi_from_list(tmp_path: Path) -> None:
    roi = tmp_path / "court.json"
    roi.write_text(
        '[{"polygon": [[0,0], [1,0], [1,1], [0,1]], "lines": {"l": [[0,0],[1,1]]}}, '
        '{"polygon": [[1,1], [2,1], [2,2], [1,2]]}]'
    )
    poly, lines = dov._load_roi(roi)
    assert isinstance(poly, Polygon)
    assert tuple(poly.exterior.coords[0]) == (0, 0)
    assert "l" in lines


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
        True,
        True,
        None,
        None,
        False,
        {},
    )
    assert res == 5
    assert dummy.rect_colors[0] == expected5
    assert dummy.rect_colors[1] == expected7
    assert dummy.rect_colors[2] == expected5


def test_draw_overlay_renders_court_polygon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000001.png").write_bytes(b"0")
    frame_map = {
        "frame_000001.png": [
            {
                "class": 100,
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        ]
    }

    class DummyCV2:
        def __init__(self) -> None:
            self.poly_count = 0

        def imread(self, path: str, flag=None):
            return types.SimpleNamespace(shape=(10, 10, 3))

        def imwrite(self, path: str, img) -> bool:
            return True

        def polylines(self, img, pts, is_closed, color, thickness):
            self.poly_count += 1

        def line(self, *a, **k):
            pass

        def rectangle(self, *a, **k):
            pass

        def putText(self, *a, **k):
            pass

        def getPerspectiveTransform(self, src, dst):  # type: ignore[override]
            return np.eye(3, dtype=float)

        def perspectiveTransform(self, pts, h):  # type: ignore[override]
            return pts

        def fillPoly(self, *a, **k):  # type: ignore[override]
            pass

        def addWeighted(self, *a, **k):  # type: ignore[override]
            pass

        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        IMREAD_COLOR = 1

    dummy = DummyCV2()
    monkeypatch.setattr(dov, "cv2", dummy)

    out = tmp_path / "out"
    res = dov._draw_overlay(
        frames,
        frame_map,
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
        "class",
        True,
        True,
        None,
        None,
        False,
        {},
    )
    assert res == 1
    assert dummy.poly_count == 1


def test_draw_overlay_renders_court_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000001.png").write_bytes(b"0")
    frame_map = {
        "frame_000001.png": [
            {
                "class": 100,
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "lines": {"service_center": [[0, 0], [1, 1]]},
                "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
        ]
    }

    import importlib
    import sys
    sys.modules.pop("numpy", None)
    real_np = importlib.import_module("numpy")
    monkeypatch.setattr(dov, "np", real_np)

    class DummyCV2:
        def __init__(self) -> None:
            self.polylines_calls: List[int] = []
            self.line_calls: List[int] = []

        def imread(self, path: str, flag=None):
            return types.SimpleNamespace(shape=(10, 10, 3))

        def imwrite(self, path: str, img) -> bool:
            return True

        def polylines(self, img, pts, is_closed, color, thickness):
            self.polylines_calls.append(thickness)

        def line(self, img, pt1, pt2, color, thickness, *a, **k):
            self.line_calls.append(thickness)

        def rectangle(self, *a, **k):
            pass

        def putText(self, *a, **k):
            pass

        def getPerspectiveTransform(self, src, dst):
            return None

        def perspectiveTransform(self, pts, h):
            return pts

        def fillPoly(self, *a, **k):
            pass

        def addWeighted(self, *a, **k):
            pass

        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        IMREAD_COLOR = 1

    dummy = DummyCV2()
    monkeypatch.setattr(dov, "cv2", dummy)

    out = tmp_path / "out"
    res = dov._draw_overlay(
        frames,
        frame_map,
        out,
        False,
        False,
        0.0,
        set(),
        set(),
        2,
        0.5,
        0,
        -1,
        0,
        "class",
        True,
        True,
        None,
        None,
        False,
        {},
    )
    assert res == 1
    assert dummy.polylines_calls == [2]
    assert dummy.line_calls  # at least one line drawn


def test_draw_overlay_skips_court_polygon_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000001.png").write_bytes(b"0")
    frame_map = {"frame_000001.png": [{"class": 100, "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}
    frame_map["frame_000001.png"][0]["homography"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    class DummyCV2:
        def __init__(self) -> None:
            self.poly_count = 0

        def imread(self, path: str, flag=None):
            return types.SimpleNamespace(shape=(10, 10, 3))

        def imwrite(self, path: str, img) -> bool:
            return True

        def polylines(self, img, pts, is_closed, color, thickness):
            self.poly_count += 1

        def line(self, *a, **k):
            pass

        def rectangle(self, *a, **k):
            pass

        def putText(self, *a, **k):
            pass

        def getPerspectiveTransform(self, src, dst):  # type: ignore[override]
            return np.eye(3, dtype=float)

        def perspectiveTransform(self, pts, h):  # type: ignore[override]
            return pts

        def fillPoly(self, *a, **k):  # type: ignore[override]
            pass

        def addWeighted(self, *a, **k):  # type: ignore[override]
            pass

        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        IMREAD_COLOR = 1

    dummy = DummyCV2()
    monkeypatch.setattr(dov, "cv2", dummy)
    warnings: list = []
    monkeypatch.setattr(dov.LOGGER, "warning", lambda *a, **k: warnings.append(a))

    out = tmp_path / "out"
    res = dov._draw_overlay(
        frames,
        frame_map,
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
        "class",
        False,
        True,
        None,
        None,
        False,
        {},
    )
    assert res == 1
    assert dummy.poly_count == 0
    assert not warnings


def test_export_mp4_skips_crf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = tmp_path / "frame_000001.png"
    frame.write_bytes(b"0")
    calls: list[list[str]] = []

    def fake_try(cmd: list[str]) -> tuple[int, str, str]:
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"0")
        return 0, "", ""

    monkeypatch.setattr(dov, "_ffmpeg_try", fake_try)
    mp4 = tmp_path / "out.mp4"
    dov._export_mp4(tmp_path, mp4, 25, -1)
    assert calls
    assert "-crf" not in calls[0]
    assert "-x264-params" not in calls[0]
    assert "-qp" not in calls[0]
    assert mp4.exists()


def test_export_mp4_fallback_no_crf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = tmp_path / "frame_000001.png"
    frame.write_bytes(b"0")
    calls: list[list[str]] = []
    responses = [
        (1, "", "Unknown encoder 'libx264'"),
        (1, "", "Unknown encoder 'h264_nvenc'"),
        (0, "", ""),
    ]

    def fake_try(cmd: list[str]) -> tuple[int, str, str]:
        calls.append(cmd)
        code, out, err = responses.pop(0)
        if code == 0:
            Path(cmd[-1]).write_bytes(b"0")
        return code, out, err

    monkeypatch.setattr(dov, "_ffmpeg_try", fake_try)
    mp4 = tmp_path / "out.mp4"
    dov._export_mp4(tmp_path, mp4, 25, 23)
    assert mp4.exists()
    assert calls
    cmd = calls[-1]
    assert "mpeg4" in cmd
    assert "-qscale:v" in cmd
    assert not any(opt in cmd for opt in ["-crf", "-x264-params", "-qp"])

def test_draw_overlay_uses_roi_by_frame_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000001.png").write_bytes(b"0")
    frame_map: dict[str, list[dict]] = {"frame_000001.png": []}
    roi_map = {
        "1": {
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
        }
    }

    class DummyCV2:
        def __init__(self) -> None:
            self.poly_count = 0
        def imread(self, path: str, flag=None):
            import types
            return types.SimpleNamespace(shape=(10, 10, 3))
        def imwrite(self, path: str, img) -> bool:
            return True
        def polylines(self, *a, **k): self.poly_count += 1
        def rectangle(self, *a, **k): pass
        def putText(self, *a, **k): pass
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        IMREAD_COLOR = 1

    import src.draw_overlay as dov
    dummy = DummyCV2()
    monkeypatch.setattr(dov, "cv2", dummy)

    out = tmp_path / "out"
    res = dov._draw_overlay(
        frames,
        frame_map,
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
        "class",
        True,
        False,
        None,
        roi_map,
        False,
        {},
        False,
    )
    assert res == 1
    assert dummy.poly_count >= 1


def test_compute_court_lines_identity() -> None:
    import importlib
    import sys

    sys.modules.pop("numpy", None)
    real_np = importlib.import_module("numpy")
    dov.np = real_np  # type: ignore[attr-defined]
    sys.modules.pop("cv2", None)
    import cv2 as real_cv2
    dov.cv2 = real_cv2  # type: ignore[attr-defined]
    lines = dov._compute_court_lines([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert "line_0" in lines
    assert lines["line_0"][0] == list(CANONICAL_LINES[0][0])


def test_draw_overlay_detect_mode_writes_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000001.png").write_bytes(b"0")
    det_map = {"frame_000001.png": [{"bbox": [0, 0, 2, 2], "score": 0.75, "class": 0}]}

    texts: list[str] = []

    class DummyCV2:
        def imread(self, path: str, flag=None):
            return types.SimpleNamespace(shape=(10, 10, 3))

        def imwrite(self, path: str, img) -> bool:
            return True

        def rectangle(self, *a, **k):
            pass

        def putText(self, img, text, org, font, fs, color, th, lt):
            texts.append(text)

        def polylines(self, *a, **k):
            pass

        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        IMREAD_COLOR = 1

    monkeypatch.setattr(dov, "cv2", DummyCV2())

    out = tmp_path / "out"
    res = dov._draw_overlay(
        frames,
        det_map,
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
        "detect",
        False,
        False,
        None,
        None,
        False,
        {},
        False,
    )
    assert res == 1
    assert any("0.75" in t for t in texts)


def test_draw_overlay_placeholder_star(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_000001.png").write_bytes(b"0")
    frame_map: dict[str, list[dict]] = {"frame_000001.png": []}
    roi_map = {
        "frame_000001.png": {
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "placeholder": True,
        }
    }

    texts: list[str] = []

    class DummyCV2:
        def imread(self, path: str, flag=None):
            return types.SimpleNamespace(shape=(10, 10, 3))

        def imwrite(self, path: str, img) -> bool:
            return True

        def polylines(self, *a, **k):
            pass

        def rectangle(self, *a, **k):
            pass

        def putText(self, img, text, org, font, fs, color, th, lt):
            texts.append(text)

        def line(self, *a, **k):
            pass

        def perspectiveTransform(self, pts, H):  # type: ignore[override]
            return pts

        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        IMREAD_COLOR = 1

    monkeypatch.setattr(dov, "cv2", DummyCV2())

    out = tmp_path / "out"
    res = dov._draw_overlay(
        frames,
        frame_map,
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
        "class",
        True,
        False,
        None,
        roi_map,
        False,
        {},
        False,
    )
    assert res == 1
    assert "*" in texts


def test_flexible_bool_parsing(tmp_path: Path) -> None:
    """Boolean flags accept both explicit and negated forms."""

    base = ["--frames-dir", str(tmp_path), "--output-dir", str(tmp_path)]

    assert dov.parse_args(base + ["--draw-court=false"]).draw_court is False
    assert dov.parse_args(base + ["--draw-court", "false"]).draw_court is False
    assert dov.parse_args(base + ["--draw-court"]).draw_court is True
    assert dov.parse_args(base + ["--no-draw-court"]).draw_court is False

    assert (
        dov.parse_args(base + ["--draw-court-axes"]).draw_court_axes is True
    )
    assert (
        dov.parse_args(base + ["--draw-court-axes=false"]).draw_court_axes is False
    )

    assert dov.parse_args(base + ["--only-court", "true"]).only_court is True
    assert dov.parse_args(base + ["--no-only-court"]).only_court is False

    defaults = dov.parse_args(base)
    assert defaults.draw_court is True
    assert defaults.draw_court_lines is True
    assert defaults.draw_court_axes is False
    assert defaults.only_court is False
