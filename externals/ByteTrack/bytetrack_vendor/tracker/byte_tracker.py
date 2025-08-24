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
"""Light-weight, IoU-based tracker used for tests.

This module provides a very small subset of the real ByteTrack interface so
that unit tests can exercise the tracking pipeline without the heavy external
dependency.  The implementation keeps a list of active tracks and assigns a
stable ``track_id`` to detections across frames using greedy IoU matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two ``tlwh`` boxes."""

    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class STrack:
    """Simple track state."""

    tlwh: np.ndarray
    score: float
    cls: int | None
    track_id: int
    missed: int = 0


class BYTETracker:
    """Very small tracker with persistent IDs.

    Parameters mirror the real ByteTrack constructor but are ignored except for
    ``track_thresh`` and ``match_thresh`` which influence association.
    """

    def __init__(
        self,
        track_thresh: float = 0.3,
        track_buffer: int = 30,
        match_thresh: float = 0.3,
        frame_rate: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracks: List[STrack] = []
        self.next_id = 1

    def update(
        self,
        tlwhs,
        scores,
        classes,
        frame_id,
        homography=None,
        assoc_w_iou: float = 1.0,
        assoc_w_plane: float = 0.0,
        plane_thresh: float = 0.25,
        court_dims=None,
        img_size: Tuple[float, float] | None = None,
    ) -> List[STrack]:
        """Update tracker state with new detections.

        Supports optional plane-aware association when ``homography`` is
        provided. The association cost combines IoU and normalised distance in
        the projected plane.
        """

        detections = [
            (np.asarray(tlwh, dtype=np.float64), float(score), cls)
            for tlwh, score, cls in zip(tlwhs, scores, classes)
            if score >= self.track_thresh
        ]

        assigned = set()
        results: List[STrack] = []

        def _proj(x: float, y: float) -> tuple[float, float]:
            if homography is None:
                return x, y
            denom = homography[2][0] * x + homography[2][1] * y + homography[2][2]
            if abs(denom) < 1e-6:
                return x, y
            u = (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / denom
            v = (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / denom
            return u, v

        def _center(box) -> tuple[float, float]:
            x, y, w, h = box
            return x + w / 2.0, y + h / 2.0

        s = max(assoc_w_iou + assoc_w_plane, 1e-6)
        w_iou, w_plane = assoc_w_iou / s, assoc_w_plane / s

        plane_diag = None
        if homography is not None and court_dims:
            cw, ch = court_dims
            plane_diag = max((cw ** 2 + ch ** 2) ** 0.5, 1e-6)

        for track in self.tracks:
            best_cost = float("inf")
            best_idx: int | None = None
            t_center = _center(track.tlwh)
            t_plane = _proj(*t_center)
            for i, (tlwh, score, cls) in enumerate(detections):
                if i in assigned:
                    continue
                iou = _iou(track.tlwh, tlwh)
                if iou < self.match_thresh:
                    continue
                d_center = _center(tlwh)
                d_plane = _proj(*d_center)
                if homography is None or court_dims is None:
                    if img_size is not None:
                        fw, fh = img_size
                        diag = (fw ** 2 + fh ** 2) ** 0.5
                    else:
                        img_w = max(track.tlwh[0] + track.tlwh[2], tlwh[0] + tlwh[2])
                        img_h = max(track.tlwh[1] + track.tlwh[3], tlwh[1] + tlwh[3])
                        diag = (img_w ** 2 + img_h ** 2) ** 0.5 or 1.0
                else:
                    diag = plane_diag
                dist = ((t_plane[0] - d_plane[0]) ** 2 + (t_plane[1] - d_plane[1]) ** 2) ** 0.5
                dist_norm = dist / max(diag, 1e-6)
                if dist_norm > 1.0:
                    dist_norm = 1.0
                if homography is not None and court_dims and dist_norm > plane_thresh:
                    continue
                cost = w_iou * (1 - iou) + w_plane * dist_norm
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            if best_idx is not None:
                tlwh, score, cls = detections[best_idx]
                track.tlwh = tlwh
                track.score = score
                track.cls = cls
                track.missed = 0
                assigned.add(best_idx)
                results.append(track)
            else:
                track.missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.track_buffer]

        for i, (tlwh, score, cls) in enumerate(detections):
            if i in assigned:
                continue
            track = STrack(tlwh=tlwh, score=score, cls=cls, track_id=self.next_id)
            self.next_id += 1
            self.tracks.append(track)
            results.append(track)

        return results
