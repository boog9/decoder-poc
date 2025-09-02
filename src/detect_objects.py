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
"""Simple CLI for player and ball detection."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger("detect")


def list_frames(frames_dir: Path) -> List[Path]:
    """Return sorted list of image frames."""
    exts = {".jpg", ".jpeg", ".png"}
    frames = [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]
    if not frames:
        raise FileNotFoundError(f"no frames found in {frames_dir}")
    return frames


def detect(frames_dir: Path, output_json: Path) -> None:
    """Write empty detections for all frames.

    This is a placeholder implementation that produces an empty detection list
    for every frame in ``frames_dir``.
    """
    frames = list_frames(frames_dir)
    data = [{"frame": f.name, "detections": []} for f in frames]
    output_json.write_text(json.dumps(data), encoding="utf-8")
    LOGGER.info("wrote %s", output_json)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    det = sub.add_parser("detect", help="run detection on a frame directory")
    det.add_argument("--frames-dir", type=Path, required=True, help="input frames directory")
    det.add_argument("--output-json", type=Path, required=True, help="path to write detections JSON")
    det.add_argument("--img-size", type=int, default=1536, help="model input size")
    det.add_argument("--p-conf", type=float, default=0.30, help="person confidence threshold")
    det.add_argument("--b-conf", type=float, default=0.05, help="ball confidence threshold")
    det.add_argument("--p-nms", type=float, default=0.60, help="person NMS threshold")
    det.add_argument("--b-nms", type=float, default=0.70, help="ball NMS threshold")
    det.add_argument("--two-pass", action="store_true", help="run two-pass filtering")
    det.add_argument("--nms-class-aware", action="store_true", help="use class-aware NMS")
    det.add_argument("--multi-scale", choices=["on", "off"], default="on", help="multi-scale mode")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.cmd == "detect":
        detect(args.frames_dir, args.output_json)
    else:  # pragma: no cover - argparse guarantees
        raise SystemExit("unknown command")


if __name__ == "__main__":  # pragma: no cover
    main()
