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
"""Integration tests for :mod:`services.court_detector.run_court`."""

import json

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
try:
    torch.zeros(1)
except Exception:  # pragma: no cover
    pytest.skip("incomplete torch implementation", allow_module_level=True)


def test_process_frames(tmp_path) -> None:
    """Processing should create output JSON with placeholders."""

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(3):
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), img)

    model = BallTrackerNet()
    weights_path = tmp_path / "w.pth"
    torch.save(model.state_dict(), weights_path)

    out_json = tmp_path / "out.json"
    process_frames(str(frames_dir), str(weights_path), str(out_json))

    data = json.loads(out_json.read_text())
    assert len(data) == 3
    assert all(item["placeholder"] for item in data)


def test_process_frames_accepts_various_formats(tmp_path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # write jpg and png with generic names
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "a.jpg"), img)
    cv2.imwrite(str(frames_dir / "b.png"), img)

    weights_path = tmp_path / "w.pth"
    torch.save(BallTrackerNet().state_dict(), weights_path)

    out_json = tmp_path / "out.json"
    process_frames(str(frames_dir), str(weights_path), str(out_json))
    data = json.loads(out_json.read_text())
    assert len(data) == 2


def test_natural_sorting(tmp_path) -> None:
    """Frames should be processed in natural numeric order."""

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    order = ["frame_1.png", "frame_10.png", "frame_2.png"]
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    for name in order:
        cv2.imwrite(str(frames_dir / name), img)

    weights_path = tmp_path / "w.pth"
    torch.save(BallTrackerNet().state_dict(), weights_path)

    out_json = tmp_path / "out.json"
    process_frames(str(frames_dir), str(weights_path), str(out_json))
    data = json.loads(out_json.read_text())
    assert [d["frame"] for d in data] == ["frame_1.png", "frame_2.png", "frame_10.png"]


def test_dump_kps_json(tmp_path) -> None:
    """Keypoints JSON should be written when requested."""

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000001.png"), img)

    weights_path = tmp_path / "w.pth"
    torch.save(BallTrackerNet().state_dict(), weights_path)

    out_json = tmp_path / "out.json"
    kp_json = tmp_path / "kps.json"
    process_frames(
        str(frames_dir),
        str(weights_path),
        str(out_json),
        kp_json_path=str(kp_json),
    )

    data = json.loads(kp_json.read_text())
    assert len(data) == 1
    assert len(data[0]["kps"]) == 14


def test_no_frames_exit_code(tmp_path) -> None:
    """Missing frames should exit with code 2."""

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    weights_path = tmp_path / "w.pth"
    torch.save(BallTrackerNet().state_dict(), weights_path)
    out_json = tmp_path / "out.json"

    with pytest.raises(SystemExit) as exc:
        process_frames(str(frames_dir), str(weights_path), str(out_json))
    assert exc.value.code == 2


def test_argparser_aliases() -> None:
    parser = build_argparser()
    ns = parser.parse_args(
        [
            "--frames-dir",
            "in",
            "--out-json",
            "x.json",
            "--stride",
            "2",
        ]
    )
    assert ns.output_json == "x.json"
    assert ns.sample_rate == 2


def test_argparser_kp_json() -> None:
    """Parser should accept --dump-kps-json option."""

    parser = build_argparser()
    ns = parser.parse_args(
        ["--frames-dir", "in", "--output-json", "out.json", "--dump-kps-json", "kps.json"]
    )
    assert ns.kp_json_path == "kps.json"
