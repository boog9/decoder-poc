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
"""Multi-scale scheduler for detection passes."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


@dataclass
class PassConfig:
    """Configuration for a single detection pass."""

    name: str
    scale: int
    type: str = "full"
    grid: Optional[tuple[int, int]] = None
    overlap: float | None = None
    roi: Optional[tuple[int, int, int, int]] = None


class MSScheduler:
    """Build a list of detection passes based on CLI options."""

    def __init__(
        self, scales: List[int], tiling: str | None = None, roi_follow: str | None = None
    ) -> None:
        self.scales = scales
        self.tiling = tiling
        self.roi_follow = roi_follow

    def build(self, has_homography: bool = False) -> List[PassConfig]:
        """Return a list of :class:`PassConfig` objects."""

        passes: List[PassConfig] = []
        if not self.scales:
            raise ValueError("at least one scale is required")
        passes.append(PassConfig(name="base", scale=self.scales[0]))
        if len(self.scales) > 1:
            passes.append(PassConfig(name="hi", scale=self.scales[1]))
        if self.tiling and has_homography:
            grid_part, overlap_part = self.tiling.split("@")
            match = re.search(r"(\d+)x(\d+)", grid_part)
            if match:
                gx, gy = int(match.group(1)), int(match.group(2))
                overlap = float(overlap_part)
                passes.append(
                    PassConfig(
                        name="tile",
                        scale=self.scales[0],
                        type="tile",
                        grid=(gx, gy),
                        overlap=overlap,
                    )
                )
        if self.roi_follow:
            match = re.search(r"win=(\d+)", self.roi_follow)
            win = int(match.group(1)) if match else self.scales[0]
            passes.append(PassConfig(name="roi", scale=win, type="roi"))
        return passes
