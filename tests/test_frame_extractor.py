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
"""Tests for :mod:`src.frame_extractor`."""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import src.frame_extractor as fe
from src import frame_extractor


def fake_popen(expected_cmd: List[str]):
    """Create a fake ``Popen`` callable returning dummy progress."""

    class FakeProc:
        def __init__(self):
            self.stdout = io.StringIO("frame=1\nframe=2\n")
            self.returncode = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def wait(self) -> None:
            return None

    def _popen(cmd, stdout=None, stderr=None, text=None, **kwargs):
        assert cmd[0] == "ffmpeg"
        return FakeProc()

    return _popen


def test_build_ffmpeg_command(tmp_path):
    cmd = fe.build_ffmpeg_command("video.mp4", str(tmp_path), 30)
    assert cmd[0] == "ffmpeg"
    assert "fps=30" in " ".join(cmd)
    assert str(tmp_path) in cmd[-3]


def test_extract_frames_invokes_ffmpeg(tmp_path, monkeypatch):
    video = tmp_path / "vid.mp4"
    video.write_text("dummy")
    outdir = tmp_path / "frames"
    expected_cmd = fe.build_ffmpeg_command(str(video), str(outdir), 30)

    monkeypatch.setattr(fe.subprocess, "Popen", fake_popen(expected_cmd))
    fe.extract_frames(video, outdir, 30)

    assert outdir.exists()


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
