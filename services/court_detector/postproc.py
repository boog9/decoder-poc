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
"""Post-processing utilities for tennis court keypoints."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional


def heat_to_peak_xy(
    heat_uint8: np.ndarray, low_thresh: int = 170, max_radius: int = 25
) -> Tuple[Optional[int], Optional[int]]:
    """Return the (x, y) coordinate of the dominant peak in a heatmap.

    Args:
        heat_uint8: Single-channel heatmap as ``uint8``.
        low_thresh: Values below this threshold are set to zero before peak
            extraction.
        max_radius: Unused placeholder for possible local refinement.

    Returns:
        ``(x, y)`` coordinate of the peak or ``(None, None)`` if no peak is
        detected.
    """

    _, thr = cv2.threshold(heat_uint8, low_thresh, 255, cv2.THRESH_TOZERO)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(thr)
    if max_val <= 0:
        return (None, None)
    return (max_loc[0], max_loc[1])


def refine_kps_if_needed(
    orig_bgr: np.ndarray,
    x: Optional[int],
    y: Optional[int],
    disable_idx: Tuple[int, int, int] = (8, 12, 9),
    k_idx: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Optional keypoint refinement hook.

    Current implementation is a stub that returns the input coordinates unless
    the keypoint index is disabled.
    """

    if x is None or y is None:
        return (x, y)
    if k_idx in disable_idx:
        return (x, y)
    return (x, y)


__all__ = ["heat_to_peak_xy", "refine_kps_if_needed"]
