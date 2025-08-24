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
"""Stub tennis court detector used for tests and offline mode."""

from __future__ import annotations

from typing import Dict, List


def _stub_detect(w: int, h: int) -> Dict[str, object]:
    """Return a deterministic full-frame court geometry.

    Args:
        w: Frame width in pixels.
        h: Frame height in pixels.

    Returns:
        Dictionary with polygon, lines, homography and score.
    """

    polygon = [
        [0.0, 0.0],
        [float(max(0, w - 1)), 0.0],
        [float(max(0, w - 1)), float(max(0, h - 1))],
        [0.0, float(max(0, h - 1))],
    ]
    homography: List[List[float]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return {
        "polygon": polygon,
        "lines": {},
        "homography": homography,
        "score": 1.0,
    }

