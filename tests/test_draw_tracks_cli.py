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
"""CLI option validation tests for :mod:`src.draw_tracks`."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import src.draw_tracks as dt


def test_cli_requires_one_output(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "f.png").write_bytes(b"0")
    tj = tmp_path / "tracks.json"
    tj.write_text("[]")

    runner = CliRunner()
    res = runner.invoke(
        dt.cli,
        [
            "--frames-dir",
            str(frames),
            "--tracks-json",
            str(tj),
            "--output-dir",
            str(tmp_path / "out"),
            "--output-video",
            str(tmp_path / "out.mp4"),
        ],
    )
    assert res.exit_code != 0
    assert "Specify exactly one" in res.output

    res = runner.invoke(
        dt.cli,
        [
            "--frames-dir",
            str(frames),
            "--tracks-json",
            str(tj),
        ],
    )
    assert res.exit_code != 0
    assert "Specify exactly one" in res.output
