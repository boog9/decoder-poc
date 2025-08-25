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
"""Ensure frame-wise merges do not leak detections across frames."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detect.merge import Detection as MSD, merge_detections
from src.utils.classes import CLASS_NAME_TO_ID


def test_framewise_merge_no_cross_leak() -> None:
    """Detections from different frames must be merged independently."""

    pid = CLASS_NAME_TO_ID["person"]

    f0_base = [MSD([0, 0, 10, 10], 0.9, pid, "base")]
    f0_hi = [MSD([1, 1, 11, 11], 0.8, pid, "hi")]

    f1_base = [MSD([100, 100, 110, 110], 0.9, pid, "base")]
    f1_hi = [MSD([101, 101, 111, 111], 0.8, pid, "hi")]

    m0 = merge_detections([f0_base, f0_hi], iou={pid: 0.5})
    m1 = merge_detections([f1_base, f1_hi], iou={pid: 0.5})

    assert len(m0) >= 1
    assert len(m1) >= 1
    assert m0[0].bbox[0] < 50
    assert m1[0].bbox[0] > 50
