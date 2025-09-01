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
"""Tests for :mod:`services.court_detector.homography_tcd`."""

from __future__ import annotations

import numpy as np
import cv2

from services.court_detector.homography_tcd import (
    HomographyEMA,
    compute_H_from_poly,
    order_poly,
    rmse,
)


def test_compute_homography_roundtrip() -> None:
    poly = [[10, 10], [20, 10], [20, 20], [10, 20]]
    H = compute_H_from_poly(poly)
    ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(ref, H).reshape(-1, 2)
    assert np.allclose(proj, np.array(order_poly(poly), np.float32), atol=1e-4)


def test_homography_ema_updates() -> None:
    ema = HomographyEMA(alpha=0.5, h_thr=0.5)
    poly1 = [[0, 0], [2, 0], [2, 2], [0, 2]]
    rec1 = ema.update(poly1)
    assert not rec1["placeholder"]
    poly2 = [[0, 0], [4, 0], [4, 4], [0, 4]]
    rec2 = ema.update(poly2)
    assert rec2["polygon"] != poly2  # EMA applied
    assert not rec2["placeholder"]
    assert rmse(rec1["polygon"], rec2["polygon"]) > 0
