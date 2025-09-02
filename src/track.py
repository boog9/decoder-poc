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
"""Simple ByteTrack wrapper for linking detections into tracks."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from loguru import logger

try:  # pragma: no cover - external dependency
    from bytetrack_vendor.tracker.byte_tracker import BYTETracker
except Exception:  # pragma: no cover - missing optional dependency
    BYTETracker = None  # type: ignore


def _load_detections(path: Path) -> List[Dict]:
    """Load detections from ``path``."""
    return json.loads(path.read_text())


def _fake_track_id() -> int:
    """Return a monotonically increasing track identifier."""
    i = 0
    while True:
        i += 1
        yield i


def track(detections_json: Path, output_json: Path, fps: int, min_score: float) -> None:
    """Assign track IDs using ByteTrack or a simple fallback."""
    data = _load_detections(detections_json)
    id_gen = _fake_track_id()
    tracker = BYTETracker(frame_rate=fps) if BYTETracker else None
    results: List[Dict] = []
    for frame in data:
        tracks = []
        if tracker:
            tlwhs, scores, classes = [], [], []
            for det in frame.get("detections", []):
                if det.get("score", 0.0) >= min_score:
                    x1, y1, x2, y2 = det["bbox"]
                    tlwhs.append((x1, y1, x2 - x1, y2 - y1))
                    scores.append(det["score"])
                    classes.append(det.get("class"))
            outputs = tracker.update(tlwhs, scores, classes, frame["frame"])
            for trk in outputs:
                x1, y1, w, h = trk.tlwh
                tracks.append(
                    {
                        "id": int(trk.track_id),
                        "bbox": [float(x1), float(y1), float(x1 + w), float(y1 + h)],
                        "class": int(trk.cls) if trk.cls is not None else None,
                        "score": float(trk.score),
                    }
                )
        else:
            for det in frame.get("detections", []):
                if det.get("score", 0.0) >= min_score:
                    tracks.append({**det, "id": next(id_gen)})
        results.append({"frame": frame["frame"], "tracks": tracks})
    output_json.write_text(json.dumps(results), encoding="utf-8")
    logger.info("wrote %s", output_json)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    trk = sub.add_parser("track", help="build tracks from detections")
    trk.add_argument("--detections-json", type=Path, required=True)
    trk.add_argument("--output-json", type=Path, required=True)
    trk.add_argument("--fps", type=int, default=30)
    trk.add_argument("--min-score", type=float, default=0.10)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.cmd == "track":
        track(args.detections_json, args.output_json, args.fps, args.min_score)
    else:  # pragma: no cover - argparse guarantees
        raise SystemExit("unknown command")


if __name__ == "__main__":  # pragma: no cover
    main()
