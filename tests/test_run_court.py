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
"""Tests for :mod:`services.court_detector.run_court`."""

import json
from pathlib import Path

import numpy as np
import pytest

if not hasattr(np, "ndarray"):
    pytest.skip("numpy not available", allow_module_level=True)

import cv2
import torch

from services.court_detector.run_court import process_frames, build_argparser
from services.court_detector.tcd_model import BallTrackerNet


if not hasattr(torch, "zeros"):
    pytest.skip("torch not available", allow_module_level=True)
try:  # ensure tensor creation works
    torch.zeros(1)
except Exception:  # pragma: no cover - environment limitation
    pytest.skip("incomplete torch implementation", allow_module_level=True)


def _write_frames(dir_path: Path) -> None:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(dir_path / "frame_a.jpg"), img)
    cv2.imwrite(str(dir_path / "frame_b.png"), img)
    cv2.imwrite(str(dir_path / "frame_c.png"), img)
    # overlay file that must be ignored
    cv2.imwrite(str(dir_path / "frame_c.png.heat.png"), img)


def test_process_frames(tmp_path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frames(frames_dir)

    weights_path = tmp_path / "w.pth"
    torch.save(BallTrackerNet().state_dict(), weights_path)

    out_json = tmp_path / "out.json"
    process_frames(str(frames_dir), str(weights_path), str(out_json))

    data = json.loads(out_json.read_text())
    assert len(data) == 3
    assert all("placeholder" in item for item in data)


def test_argparser_aliases() -> None:
    parser = build_argparser()
    ns = parser.parse_args([
        "--frames-dir",
        "in",
        "--out-json",
        "x.json",
        "--stride",
        "2",
    ])
    assert ns.output_json == "x.json"
    assert ns.sample_rate == 2
