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
"""Tests for :mod:`services.court_detector.postproc`."""

import numpy as np
import pytest

if not hasattr(np, "ndarray"):
    pytest.skip("numpy not available", allow_module_level=True)

from services.court_detector.postproc import heat_to_peak_xy, refine_kps_if_needed


def test_heat_to_peak_xy() -> None:
    """Peak finder should return coordinates of the maximum."""

    heat = np.zeros((10, 10), dtype=np.uint8)
    heat[2, 3] = 200
    x, y = heat_to_peak_xy(heat)
    assert (x, y) == (3, 2)


def test_refine_kps_if_needed_noop() -> None:
    """Refinement stub should leave coordinates unchanged."""

    img = np.zeros((5, 5, 3), dtype=np.uint8)
    assert refine_kps_if_needed(img, 1, 2, k_idx=0) == (1, 2)
