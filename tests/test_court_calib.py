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
"""Tests for helper functions in :mod:`src.court_calib`."""

from __future__ import annotations

import sys
from pathlib import Path

# Replace dummy numpy from conftest with real implementation for cv2.
sys.modules.pop("numpy", None)
import numpy as np  # type: ignore  # noqa: E402

# Ensure modules under repository are importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from court_calib import identity_h, lerp_poly, scale_poly


def test_scale_poly() -> None:
    """Polygon is scaled from 640×360 to arbitrary size."""

    poly = np.array([[0, 0], [640, 0], [640, 360], [0, 360]], dtype=np.float32)
    scaled = scale_poly(poly, 1280, 720)
    assert scaled[2] == [1280.0, 720.0]


def test_identity_h() -> None:
    """Identity homography has ones on the diagonal."""

    H = identity_h()
    assert H[0][0] == 1.0 and H[1][1] == 1.0 and H[2][2] == 1.0


def test_lerp_poly() -> None:
    """Linear interpolation between polygons uses factor t."""

    p0 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    p1 = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    pj = lerp_poly(p0, p1, 0.5)
    expected = np.array(
        [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]], dtype=np.float32
    )
    assert np.allclose(pj, expected)
