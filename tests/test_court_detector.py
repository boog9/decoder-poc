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
"""Tests for :mod:`src.court_detector` stub."""

from __future__ import annotations

import src.court_detector as cd


def test_stub_detect_returns_full_frame() -> None:
    """Ensure stub detector returns full-frame polygon and identity homography."""

    w, h = 320, 200
    res = cd._stub_detect(w, h)
    assert res["score"] == 1.0
    assert res["polygon"][0] == [0.0, 0.0]
    assert res["polygon"][2] == [float(w - 1), float(h - 1)]
    H = res["homography"]
    assert H[0][0] == 1.0 and H[1][1] == 1.0 and H[2][2] == 1.0
    assert cd._stub_detect(0, 0)["polygon"][0] == [0.0, 0.0]

