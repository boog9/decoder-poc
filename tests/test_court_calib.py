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
"""Tests for :mod:`src.court_calib`."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.court_calib as cc


def test_calibrate_court_interpolates_homography(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(5):
        (frames / f"frame_{i:06d}.png").write_bytes(b"")

    dummy_img = types.SimpleNamespace(size=(0, 0))
    sizes = {frames / f"frame_{i:06d}.png": (100 + i * 10, 80) for i in range(5)}

    class DummyCtx:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> types.SimpleNamespace:
            dummy_img.size = sizes[self.path]
            return dummy_img

        def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover
            return False

    monkeypatch.setattr(cc.Image, "open", lambda p: DummyCtx(p))
    monkeypatch.setattr(
        cc,
        "logger",
        types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )
    monkeypatch.setattr(cc, "verify_torch_ckpt", lambda path: None)

    def fake_detect(img, device, weights, min_score):
        w, _ = img.size
        tx = w - 100
        h = [[1.0, 0.0, float(tx)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        poly = [[float(tx), 0.0], [float(tx + 1), 0.0], [float(tx + 1), 1.0], [float(tx), 1.0]]
        return {"polygon": poly, "lines": {}, "homography": h, "score": 0.9}

    monkeypatch.setattr(cc.cd, "detect_single_frame", fake_detect)

    res = cc.calibrate_court(
        frames,
        device="cpu",
        weights=Path("w"),
        min_score=0.5,
        stride=2,
        allow_placeholder=False,
    )
    assert len(res) == 5
    # Translation for frame1 should be halfway between frame0 (0) and frame2 (20)
    assert abs(res[1]["homography"][0][2] - 10.0) < 1e-6
    assert abs(res[1]["polygon"][0][0] - 10.0) < 1e-6
    assert sum(not r["placeholder"] for r in res) > 0
    assert all(r["placeholder"] is False for r in res)


def test_parse_args_aliases() -> None:
    """Ensure CLI aliases map to canonical arguments."""

    args = cc.parse_args(
        [
            "--frames-dir",
            "frames",
            "--output-json",
            "out.json",
            "--sample-rate",
            "7",
            "--stabilize",
            "ema",
            "--weights",
            "w.pth",
        ]
    )
    assert args.out_json == Path("out.json")
    assert args.stride == 7
