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
"""Runner executing detection passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List

from .ms_scheduler import PassConfig


class DetectionRunner:
    """Execute detection passes and map boxes to global coordinates."""

    def __init__(self, detect_func: Callable[[List[Path], int], List[List[Dict[str, Any]]]]):
        self.detect_func = detect_func

    def run(self, frames: List[Path], cfg: PassConfig) -> Dict[Path, List[Dict[str, Any]]]:
        if cfg.type == "full":
            outputs = self.detect_func(frames, cfg.scale)
            return {f: d for f, d in zip(frames, outputs)}
        if cfg.type == "roi" and cfg.roi is not None:
            x1, y1, _, _ = cfg.roi
            outputs = self.detect_func(frames, cfg.scale)
            adjusted: Dict[Path, List[Dict[str, Any]]] = {}
            for f, dets in zip(frames, outputs):
                adj: List[Dict[str, Any]] = []
                for d in dets:
                    bx = d["bbox"]
                    adj_box = [bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1]
                    nd = dict(d)
                    nd["bbox"] = adj_box
                    adj.append(nd)
                adjusted[f] = adj
            return adjusted
        if cfg.type == "tile" and cfg.grid is not None:
            gx, gy = cfg.grid
            tile_w = cfg.scale // gx
            tile_h = cfg.scale // gy
            outputs = self.detect_func(frames, cfg.scale)
            adjusted: Dict[Path, List[Dict[str, Any]]] = {f: [] for f in frames}
            for y in range(gy):
                for x in range(gx):
                    x_off = x * tile_w
                    y_off = y * tile_h
                    for f, dets in zip(frames, outputs):
                        for d in dets:
                            bx = d["bbox"]
                            adj_box = [bx[0] + x_off, bx[1] + y_off, bx[2] + x_off, bx[3] + y_off]
                            nd = dict(d)
                            nd["bbox"] = adj_box
                            adjusted[f].append(nd)
            return adjusted
        outputs = self.detect_func(frames, cfg.scale)
        return {f: d for f, d in zip(frames, outputs)}
