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
"""Homography utilities for TCD court detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import numpy as np


def order_poly(poly: Iterable[Iterable[float]]) -> np.ndarray:
    """Order four points in TL, TR, BR, BL sequence."""

    P = np.asarray(list(poly), np.float32)
    s = P.sum(1)
    d = np.diff(P, axis=1).ravel()
    tl, br, tr, bl = np.argmin(s), np.argmax(s), np.argmin(d), np.argmax(d)
    return np.array([P[tl], P[tr], P[br], P[bl]], np.float32)


def compute_H_from_poly(poly: Iterable[Iterable[float]]) -> np.ndarray:
    """Compute canonical→image homography for ``poly``."""

    ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    img = order_poly(poly)
    return cv2.getPerspectiveTransform(ref, img)


def rmse(a: Iterable[Iterable[float]], b: Iterable[Iterable[float]]) -> float:
    """Return root-mean-square error between point sets ``a`` and ``b``."""

    A, B = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.sqrt(np.mean((A - B) ** 2)))


@dataclass
class HomographyEMA:
    """Maintain EMA-stabilized polygon and homography."""

    alpha: float = 0.2
    h_thr: float = 5.0
    _poly: Optional[np.ndarray] = None
    _H: Optional[np.ndarray] = None

    def update(self, poly: List[List[float]]) -> dict:
        """Update with new polygon and return record for ``court.json``."""

        P = order_poly(poly)
        if self._poly is None:
            self._poly = P
        else:
            self._poly = self.alpha * P + (1.0 - self.alpha) * self._poly

        H = compute_H_from_poly(self._poly)
        ref = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        proj = cv2.perspectiveTransform(ref.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = rmse(proj, self._poly)
        placeholder = err > self.h_thr or not np.isfinite(H).all()
        if placeholder:
            H_out = self._H
        else:
            self._H = H
            H_out = H
        rec = {
            "polygon": self._poly.tolist(),
            "homography": H_out.tolist() if H_out is not None else [],
            "placeholder": placeholder,
        }
        return rec


__all__ = ["order_poly", "compute_H_from_poly", "rmse", "HomographyEMA"]
