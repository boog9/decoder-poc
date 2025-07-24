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
"""Utilities for validating detection JSON files."""

from __future__ import annotations

import json
from pathlib import Path
import argparse
from typing import Iterable, Tuple

import logging
from PIL import Image

LOGGER = logging.getLogger(__name__)


def sanity_check(
    json_path: Path,
    frames_dir: Path,
    score_thr: float = 0.3,
) -> Tuple[int, int]:
    """Validate bounding boxes and confidence scores.

    Args:
        json_path: Path to the detections JSON file.
        frames_dir: Directory containing frame images.
        score_thr: Threshold below which detections are considered low
            confidence.

    Returns:
        Tuple ``(bad_boxes, low_conf)`` with counts of invalid boxes and low
        confidence detections.
    """
    bad_boxes = 0
    low_conf = 0
    with json_path.open() as fh:
        data = json.load(fh)

    for obj in data:
        frame_path = frames_dir / obj["frame"]
        if not frame_path.exists():
            continue
        with Image.open(frame_path) as img:
            w, h = img.size
        for det in obj.get("detections", []):
            x0, y0, x1, y1 = det.get("bbox", [0, 0, 0, 0])
            score = det.get("score", 0.0)
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                bad_boxes += 1
            if score < score_thr:
                low_conf += 1
    return bad_boxes, low_conf


def _quantiles(values: list[float]) -> list[float]:
    """Return min, 25%%, median, 75%% and max for ``values``."""
    if not values:
        return [0.0] * 5
    values = sorted(values)
    n = len(values)
    def _q(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return values[lo] * (1 - frac) + values[hi] * frac
    return [_q(p) for p in [0, 0.25, 0.5, 0.75, 1]]


def score_area_stats(json_path: Path) -> Tuple[list[float], list[float]]:
    """Compute quantiles for detection scores and areas."""
    with json_path.open() as fh:
        data = json.load(fh)

    scores = []
    areas = []
    for fr in data:
        for det in fr.get("detections", []):
            scores.append(det.get("score", 0.0))
            x0, y0, x1, y1 = det.get("bbox", [0, 0, 0, 0])
            areas.append(max(0.0, (x1 - x0) * (y1 - y0)))
    score_q = _quantiles(scores)
    area_q = _quantiles(areas)
    return score_q, area_q


__all__ = ["sanity_check", "score_area_stats"]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    chk = sub.add_parser("sanity-check", help="Validate bounding boxes")
    chk.add_argument("--detections", type=Path, required=True)
    chk.add_argument("--frames-dir", type=Path, required=True)
    chk.add_argument("--score-thr", type=float, default=0.3)

    st = sub.add_parser("stats", help="Show score/area quantiles")
    st.add_argument("--detections", type=Path, required=True)

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.cmd == "sanity-check":
        bad, low = sanity_check(args.detections, args.frames_dir, args.score_thr)
        LOGGER.info("Invalid boxes: %d", bad)
        LOGGER.info("Low confidence: %d", low)
    elif args.cmd == "stats":
        scores, areas = score_area_stats(args.detections)
        LOGGER.info("Quantiles score: %s", [round(s, 3) for s in scores])
        LOGGER.info("Quantiles area : %s", [round(a, 1) for a in areas])


if __name__ == "__main__":  # pragma: no cover
    main()
