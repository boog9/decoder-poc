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
from typing import List

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

    def update(self, tlwhs, scores, classes, frame_id):
        """Update tracker state with new detections."""

        detections = [
            (np.asarray(tlwh, dtype=np.float64), float(score), cls)
            for tlwh, score, cls in zip(tlwhs, scores, classes)
            if score >= self.track_thresh
        ]

        assigned = set()
        results: List[STrack] = []
        # Associate detections to existing tracks.
        for track in self.tracks:
            best_iou = 0.0
            best_idx: int | None = None
            for i, (tlwh, score, cls) in enumerate(detections):
                if i in assigned:
                    continue
                iou = _iou(track.tlwh, tlwh)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx is not None and best_iou >= self.match_thresh:
                tlwh, score, cls = detections[best_idx]
                track.tlwh = tlwh
                track.score = score
                track.cls = cls
                track.missed = 0
                assigned.add(best_idx)
                results.append(track)
            else:
                track.missed += 1

        # Remove tracks that have been missed for too long.
        self.tracks = [t for t in self.tracks if t.missed <= self.track_buffer]

        # Create new tracks for unmatched detections.
        for i, (tlwh, score, cls) in enumerate(detections):
            if i in assigned:
                continue
            track = STrack(tlwh=tlwh, score=score, cls=cls, track_id=self.next_id)
            self.next_id += 1
            self.tracks.append(track)
            results.append(track)

        return results
