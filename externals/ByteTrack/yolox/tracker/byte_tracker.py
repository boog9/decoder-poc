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
"""Minimal stub of ByteTrack tracker for testing."""

from __future__ import annotations

import numpy as np


class STrack:
    """Simple track object."""

    def __init__(self, tlwh: list[float]):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.score = 0.0
        self.track_id = -1

    @property
    def tlwh(self) -> np.ndarray:
        """Return the track bounding box as ``(x, y, w, h)``."""
        return self._tlwh


class BYTETracker:
    """Dummy BYTETracker implementation."""

    def __init__(self, *args, **kwargs) -> None:
        self.tracks: list[STrack] = []
        self.next_id = 1

    def update(self, tlwhs, scores, classes, frame_id):
        out = []
        for tlwh, score in zip(tlwhs, scores):
            track = STrack(tlwh)
            track.score = float(score)
            track.track_id = self.next_id
            self.next_id += 1
            out.append(track)
        self.tracks.extend(out)
        return out
