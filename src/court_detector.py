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
"""Simple tennis court detector placeholder.

This module provides a lightweight placeholder implementation of a tennis
court detector. Each frame is processed independently and the polygon is
approximated by the full image extent.

Example usage:

```bash
python -m src.court_detector --frames-dir frames/ --output-json court.json
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image


def detect_court(
    frames_dir: Path,
    *,
    device: str = "auto",
    use_homography: bool = False,
    refine_kps: bool = False,
    weights: Path | None = None,
) -> List[Dict[str, Any]]:
    """Detect tennis court polygons for frames in ``frames_dir``.

    Parameters
    ----------
    frames_dir:
        Directory containing frame images.
    device:
        Inert parameter selecting computation device.
    use_homography:
        Placeholder flag for homography refinement.
    refine_kps:
        Placeholder flag for keypoint refinement.
    weights:
        Optional path to model weights (unused).

    Returns
    -------
    list of dict
        Each item contains ``frame``, ``class`` and ``polygon`` keys.
    """

    frames = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    result: List[Dict[str, Any]] = []
    for frame in frames:
        try:
            with Image.open(frame) as img:
                w, h = img.size
        except Exception:
            continue
        polygon = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        result.append({"frame": frame.name, "class": 100, "polygon": polygon})
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True, help="Input frames directory")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--use-homography",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable homography refinement (placeholder)",
    )
    parser.add_argument(
        "--refine-kps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable keypoint refinement (placeholder)",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Model weights (unused)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    data = detect_court(
        args.frames_dir,
        device=args.device,
        use_homography=args.use_homography,
        refine_kps=args.refine_kps,
        weights=args.weights,
    )
    with args.output_json.open("w") as fh:
        json.dump(data, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
