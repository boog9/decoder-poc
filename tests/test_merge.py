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
"""Tests for :mod:`src.detect.merge`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detect.merge import Detection, merge_detections
from src.utils.classes import CLASS_NAME_TO_ID


def test_wbf_and_topk() -> None:
    ball_id = CLASS_NAME_TO_ID["sports ball"]
    d1 = Detection(bbox=[0.0, 0.0, 10.0, 10.0], score=0.9, class_id=ball_id, source="base")
    d2 = Detection(bbox=[1.0, 1.0, 11.0, 11.0], score=0.8, class_id=ball_id, source="hi")
    merged = merge_detections(
        [[d1], [d2]],
        iou={ball_id: 0.5},
        bonuses={"hi": {ball_id: 0.1}},
        topk={ball_id: 1},
    )
    assert len(merged) == 1
    assert merged[0].score > 0.8
    assert merged[0].bbox[0] > 0.0
