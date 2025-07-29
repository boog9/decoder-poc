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
"""Common drawing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_FONT_LINE = cv2.LINE_AA


def load_frames(frames_dir: Path, max_frames: int | None = None) -> List[Path]:
    """Return sorted image paths from ``frames_dir``.

    Only files with extensions from :data:`IMAGE_EXT` are returned.
    """
    frames = sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXT
    )
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def get_font() -> Tuple[int, float, int]:
    """Return font, scale and line type for text drawing."""

    return _FONT, _FONT_SCALE, _FONT_LINE


__all__ = ["IMAGE_EXT", "load_frames", "get_font"]
