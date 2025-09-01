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
"""Tests for symlink staging in :mod:`src.draw_overlay`."""

from __future__ import annotations

import os
import shutil
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

cv2_dummy = types.ModuleType("cv2")
cv2_dummy.imread = lambda *a, **k: None
cv2_dummy.imwrite = lambda *a, **k: True
cv2_dummy.rectangle = lambda *a, **k: None
cv2_dummy.putText = lambda *a, **k: None
cv2_dummy.polylines = lambda *a, **k: None
cv2_dummy.FONT_HERSHEY_SIMPLEX = 0
cv2_dummy.LINE_AA = 16
sys.modules.setdefault("cv2", cv2_dummy)

geometry = types.SimpleNamespace(Polygon=None)
shapely_stub = types.SimpleNamespace(geometry=geometry)
sys.modules.setdefault("shapely", shapely_stub)
sys.modules.setdefault("shapely.geometry", geometry)

import pytest

import src.draw_overlay as dov


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDATx\x9cc\xf8\x0f\x00\x01"
    b"\x01\x01\x00\x18\xdd\x8f\xe1\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_stage_symlink_relative(tmp_path: Path) -> None:
    """Ensure staged symlinks are relative and optional MP4 export works."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    (out_dir / "frame_000001.png").write_bytes(PNG_BYTES)
    (out_dir / "frame_000002.png").write_bytes(PNG_BYTES)

    stage = dov._stage_frames(out_dir)
    link = stage / "000001.png"
    assert link.is_symlink()
    assert os.readlink(link) == "../frame_000001.png"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4_path = out_dir / "video.mp4"
        dov._export_mp4(out_dir, mp4_path, fps=30, crf=18)
        assert mp4_path.exists()
