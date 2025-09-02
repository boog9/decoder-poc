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
"""Overlay detections or tracks on video frames."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import cv2

LOGGER = logging.getLogger("draw_overlay")


def _load_frames(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    frames = [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]
    if not frames:
        raise FileNotFoundError(f"no frames found in {frames_dir}")
    return frames


def _draw_boxes(img, boxes: List[Dict], show_id: bool, show_label: bool) -> None:
    for obj in boxes:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        parts = []
        if show_label and "class" in obj:
            parts.append(str(obj["class"]))
        if show_id and "id" in obj:
            parts.append(f"#{obj['id']}")
        if parts:
            cv2.putText(img, " ".join(parts), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def draw_detect(frames_dir: Path, detections_json: Path, output_dir: Path, show_label: bool) -> None:
    frames = _load_frames(frames_dir)
    detections = {item["frame"]: item["detections"] for item in json.loads(detections_json.read_text())}
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img = cv2.imread(str(frame))
        _draw_boxes(img, detections.get(frame.name, []), show_id=False, show_label=show_label)
        cv2.imwrite(str(output_dir / frame.name), img)


def draw_track(frames_dir: Path, tracks_json: Path, output_dir: Path, show_id: bool, show_label: bool) -> None:
    frames = _load_frames(frames_dir)
    tracks = {item["frame"]: item.get("tracks", []) for item in json.loads(tracks_json.read_text())}
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img = cv2.imread(str(frame))
        _draw_boxes(img, tracks.get(frame.name, []), show_id=show_id, show_label=show_label)
        cv2.imwrite(str(output_dir / frame.name), img)


def export_mp4(frames_dir: Path, output_mp4: Path, fps: int, crf: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        str(output_mp4),
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["detect", "track"], required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--detections-json", type=Path)
    parser.add_argument("--tracks-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-mp4", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--id", action="store_true", help="draw track ids")
    parser.add_argument("--label", action="store_true", help="draw class labels")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    if args.mode == "detect" and args.detections_json:
        draw_detect(args.frames_dir, args.detections_json, args.output_dir, args.label)
    elif args.mode == "track" and args.tracks_json:
        draw_track(args.frames_dir, args.tracks_json, args.output_dir, args.id, args.label)
    else:
        raise SystemExit("invalid arguments")
    if args.export_mp4:
        export_mp4(args.output_dir, args.export_mp4, args.fps, args.crf)


if __name__ == "__main__":  # pragma: no cover
    main()
