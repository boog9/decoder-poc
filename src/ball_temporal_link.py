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
"""Temporal linking utilities for tennis ball detections.

This module fills short gaps in ``sports ball`` detections by linear
interpolation and removes stationary false positives (e.g. scoreboard icons).
"""

from __future__ import annotations

from typing import List

import re


def _extract_frame_id(val) -> int | None:
    match = re.findall(r"\d+", str(val))
    return int(match[-1]) if match else None


def _center(bbox: List[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def link_ball_detections(entries: List[dict], gap_max: int = 5) -> None:
    """In-place interpolation of missing ball detections.

    Parameters
    ----------
    entries:
        Detection results in nested format ``[{"frame": str, "detections": [...]}, ...]``.
    gap_max:
        Maximum gap in frames to interpolate.  # tennis tuning
    """

    balls: List[tuple[int, dict]] = []
    for entry in entries:
        frame_id = _extract_frame_id(entry.get("frame")) or 0
        for det in entry.get("detections", []):
            if det.get("class") == _class_id_ball():
                balls.append((frame_id, det))

    balls.sort(key=lambda x: x[0])

    # Interpolate gaps
    for idx in range(len(balls) - 1, 0, -1):
        f0, d0 = balls[idx - 1]
        f1, d1 = balls[idx]
        gap = f1 - f0
        if 1 < gap <= gap_max:  # tennis tuning
            c0 = _center(d0["bbox"])
            c1 = _center(d1["bbox"])
            for step in range(1, gap):
                nx = c0[0] + (c1[0] - c0[0]) * step / gap
                ny = c0[1] + (c1[1] - c0[1]) * step / gap
                w = d0["bbox"][2] - d0["bbox"][0]
                h = d0["bbox"][3] - d0["bbox"][1]
                bbox = [nx - w / 2, ny - h / 2, nx + w / 2, ny + h / 2]
                balls.insert(
                    idx,
                    (
                        f0 + step,
                        {
                            "bbox": bbox,
                            "score": 0.25,
                            "class": d0["class"],
                            "interpolated": True,
                        },
                    ),
                )

    # Remove stationary sequences (>5 frames with <1px movement)
    # tennis tuning: filters scoreboard or lamp artifacts
    filtered: List[tuple[int, dict]] = []
    last_c = None
    stationary = 0
    for f, det in balls:
        c = _center(det["bbox"])
        if last_c and (abs(c[0] - last_c[0]) <= 1 and abs(c[1] - last_c[1]) <= 1):
            stationary += 1
        else:
            stationary = 0
        last_c = c
        if stationary >= 5:
            continue
        filtered.append((f, det))

    # Rebuild nested entries
    per_frame: dict[int, List[dict]] = {}
    for f, det in filtered:
        per_frame.setdefault(f, []).append(det)
    for entry in entries:
        fid = _extract_frame_id(entry.get("frame")) or 0
        entry["detections"].extend(per_frame.get(fid, []))


def _class_id_ball() -> int:
    from .utils.classes import CLASS_NAME_TO_ID

    return CLASS_NAME_TO_ID["sports ball"]


__all__ = ["link_ball_detections"]
