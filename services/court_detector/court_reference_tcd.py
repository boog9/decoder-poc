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
"""Canonical tennis court model in normalized coordinates."""

from __future__ import annotations

from typing import List, Tuple

# Normalized service line positions based on 23.77m court length and
# 6.40m service box offset from the net.
_service = 6.40 / 23.77
_south = 0.5 - _service
_north = 0.5 + _service
_mark = 0.1 / 23.77

CANONICAL_LINES: List[List[Tuple[float, float]]] = [
    [(0.0, 0.0), (1.0, 0.0)],  # south baseline
    [(0.0, 1.0), (1.0, 1.0)],  # north baseline
    [(0.0, 0.0), (0.0, 1.0)],  # west sideline
    [(1.0, 0.0), (1.0, 1.0)],  # east sideline
    [(0.0, 0.5), (1.0, 0.5)],  # net
    [(0.125, 0.0), (0.125, 1.0)],  # singles left
    [(0.875, 0.0), (0.875, 1.0)],  # singles right
    [(0.0, _south), (1.0, _south)],  # south service line
    [(0.0, _north), (1.0, _north)],  # north service line
    [(0.5, _south), (0.5, _north)],  # center service line
    [(0.5, 0.0), (0.5, _mark)],  # south center mark
    [(0.5, 1.0), (0.5, 1.0 - _mark)],  # north center mark
]

__all__ = ["CANONICAL_LINES"]
