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
"""Compatibility wrapper for :mod:`src.draw_overlay` in track mode."""

from __future__ import annotations

import logging
import sys
from typing import Iterable

from . import draw_overlay

LOGGER = logging.getLogger("draw_tracks")


def main(argv: Iterable[str] | None = None) -> int:
    """Forward to :func:`draw_overlay.main` in ``track`` mode.

    Args:
        argv: Optional iterable of CLI arguments.

    Returns:
        Exit status from :func:`draw_overlay.main`.
    """

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    LOGGER.warning(
        "`src.draw_tracks` is deprecated, forwarding to `src.draw_overlay --mode track`..."
    )
    args = list(argv) if argv is not None else sys.argv[1:]
    return draw_overlay.main(["--mode", "track", *args])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
