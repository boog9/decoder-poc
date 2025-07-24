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
"""Draw ROI detections on frame images."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw

LOGGER = logging.getLogger(__name__)



def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing input frame images.",
    )
    parser.add_argument(
        "--detections-json",
        type=Path,
        required=True,
        help="JSON file with detection data from detect_objects",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where annotated frames will be written",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="red",
        help="Rectangle outline color (default: red)",
    )
    return parser.parse_args(argv)



def _load_detections(path: Path) -> List[dict]:
    """Load detection list from ``path``."""
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Invalid detections format")
    return data


def _sanitize_bbox(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Ensure ``bbox`` coordinates are top-left to bottom-right.

    Args:
        bbox: Bounding box as ``[x1, y1, x2, y2]``.

    Returns:
        Sanitized bounding box tuple ``(x1, y1, x2, y2)`` with coordinates
        sorted so ``x2 >= x1`` and ``y2 >= y1``.
    """

    x1, y1, x2, y2 = bbox
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2



def draw_rois(frames_dir: Path, detections_json: Path, output_dir: Path, color: str = "red") -> None:
    """Overlay detection ROIs on frames and save to ``output_dir``.

    Args:
        frames_dir: Directory of frame images.
        detections_json: JSON file with detection results.
        output_dir: Destination for annotated images.
        color: Outline color for rectangles.
    """
    detections = _load_detections(detections_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in detections:
        frame_name = entry.get("frame")
        rois = [det.get("bbox") for det in entry.get("detections", [])]
        if not frame_name:
            LOGGER.debug("Skipping detection entry without frame")
            continue
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            LOGGER.warning("Frame not found: %s", frame_path)
            continue
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for bbox in rois:
            if not isinstance(bbox, list) or len(bbox) != 4:
                LOGGER.debug("Invalid bbox %s in %s", bbox, frame_name)
                continue
            x1, y1, x2, y2 = _sanitize_bbox(bbox)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        out_path = output_dir / frame_name
        img.save(out_path)
        LOGGER.debug("Wrote %s", out_path)



def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        draw_rois(args.frames_dir, args.detections_json, args.output_dir, args.color)
    except Exception as exc:  # pragma: no cover - top level
        LOGGER.error("Failed to draw ROIs: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
