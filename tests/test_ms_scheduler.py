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
"""Tests for :mod:`src.detect.ms_scheduler`."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detect.ms_scheduler import MSScheduler


def test_build_passes_all() -> None:
    sched = MSScheduler(scales=[640, 1280], tiling="far2x2@0.1", roi_follow="ball:win=320")
    passes = sched.build(has_homography=True)
    names = [p.name for p in passes]
    assert names == ["base", "hi", "tile", "roi"]
