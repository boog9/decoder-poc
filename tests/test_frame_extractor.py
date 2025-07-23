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
"""Tests for frame_extractor module."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import frame_extractor


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_frames(tmp_path: Path) -> None:
    """Ensure frames are extracted from a short generated video."""
    video = tmp_path / "test.mp4"
    output_dir = tmp_path / "frames"

    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=128x128:rate=30",
            str(video),
            "-loglevel",
            "error",
        ],
        check=True,
    )

    frame_extractor.extract_frames(video, output_dir, fps=10)

    extracted = list(output_dir.glob("*.jpg"))
    assert extracted, "No frames were extracted"

