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
"""Tests for :mod:`src.detect.runner`."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detect.ms_scheduler import PassConfig
from src.detect.runner import DetectionRunner


def _dummy(frames: list[Path], scale: int):
    return [[{"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "class_id": 0}] for _ in frames]


def test_roi_offsets(tmp_path: Path) -> None:
    frame = tmp_path / "f.png"
    frame.touch()
    runner = DetectionRunner(_dummy)
    cfg = PassConfig(name="roi", scale=100, type="roi", roi=(10, 20, 30, 40))
    out = runner.run([frame], cfg)
    det = out[frame][0]
    assert det["bbox"] == [11.0, 22.0, 13.0, 24.0]
