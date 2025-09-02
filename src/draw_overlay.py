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
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("draw_overlay")


def _load_frames(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    frames = [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]
    if not frames:
        raise FileNotFoundError(f"no frames found in {frames_dir}")
    return frames


def _hex_to_bgr(color: str) -> Tuple[int, int, int]:
    """Convert a ``#RRGGBB`` string into a BGR tuple for OpenCV.

    Args:
        color: Hex color string.

    Returns:
        Tuple representing BGR color. Defaults to yellow for invalid input.
    """

    if not isinstance(color, str):
        return (0, 255, 255)
    s = color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return (0, 255, 255)
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def load_roi_json(path: Path) -> List[Dict]:
    """Load ROI polygons from a JSON description.

    The file structure is expected to be::

        {
          "polygons": [
            {
              "name": "ROI-1",
              "points": [[120, 80], [1280, 80], [1280, 640], [120, 640]],
              "color": "#00FF00",
              "thickness": 2,
              "fill_alpha": 0.15
            }
          ]
        }

    Args:
        path: Path to the JSON file.

    Returns:
        Sanitized list of ROI dictionaries ready for drawing.
    """

    try:
        data = json.loads(path.read_text())
        polys = data.get("polygons", [])
        sanitized: List[Dict] = []
        for poly in polys:
            pts = poly.get("points", [])
            if not isinstance(pts, list) or len(pts) < 3:
                continue
            name = poly.get("name")
            color = _hex_to_bgr(poly.get("color", "#FFFF00"))
            thickness = int(poly.get("thickness", 2))
            fill_alpha = float(poly.get("fill_alpha", 0.0))
            fill_alpha = max(0.0, min(1.0, fill_alpha))
            ipts = np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)
            sanitized.append(
                {
                    "name": name,
                    "points": ipts,
                    "color": color,
                    "thickness": thickness,
                    "fill_alpha": fill_alpha,
                }
            )
        return sanitized
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to load ROI json '%s': %s", path, exc)
        return []


def draw_rois(img, rois: List[Dict]) -> None:
    """Draw ROI polygons on an image.

    Args:
        img: Target image (modified in place).
        rois: ROI description returned by :func:`load_roi_json`.
    """

    if not rois:
        return
    overlay = img.copy()
    for roi in rois:
        pts = roi["points"].reshape((-1, 1, 2))
        color = roi["color"]
        thickness = roi["thickness"]
        alpha = roi["fill_alpha"]
        if alpha > 0:
            cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, color, thickness, lineType=cv2.LINE_AA)
        if roi.get("name"):
            x, y = int(pts[0, 0, 0]), int(pts[0, 0, 1])
            cv2.putText(
                overlay,
                str(roi["name"]),
                (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.addWeighted(overlay, 1.0, img, 0.0, 0, dst=img)


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


def draw_detect(
    frames_dir: Path,
    detections_json: Path,
    output_dir: Path,
    show_label: bool,
    rois: Optional[List[Dict]] = None,
) -> None:
    frames = _load_frames(frames_dir)
    detections = {
        item["frame"]: item["detections"]
        for item in json.loads(detections_json.read_text())
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img = cv2.imread(str(frame))
        if rois:
            draw_rois(img, rois)
        _draw_boxes(img, detections.get(frame.name, []), show_id=False, show_label=show_label)
        cv2.imwrite(str(output_dir / frame.name), img)


def draw_track(
    frames_dir: Path,
    tracks_json: Path,
    output_dir: Path,
    show_id: bool,
    show_label: bool,
    rois: Optional[List[Dict]] = None,
) -> None:
    frames = _load_frames(frames_dir)
    tracks = {
        item["frame"]: item.get("tracks", [])
        for item in json.loads(tracks_json.read_text())
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img = cv2.imread(str(frame))
        if rois:
            draw_rois(img, rois)
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
    parser.add_argument(
        "--roi-json",
        type=Path,
        help="Optional ROI polygons JSON for visualization only (no effect on detection/tracking)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rois: List[Dict] = []
    if args.roi_json:
        rois = load_roi_json(args.roi_json)
        LOGGER.info("Loaded %d ROI polygons for visualization", len(rois))

    if args.mode == "detect" and args.detections_json:
        draw_detect(args.frames_dir, args.detections_json, args.output_dir, args.label, rois)
    elif args.mode == "track" and args.tracks_json:
        draw_track(
            args.frames_dir,
            args.tracks_json,
            args.output_dir,
            args.id,
            args.label,
            rois,
        )
    else:
        raise SystemExit("invalid arguments")
    if args.export_mp4:
        export_mp4(args.output_dir, args.export_mp4, args.fps, args.crf)


if __name__ == "__main__":  # pragma: no cover
    main()
