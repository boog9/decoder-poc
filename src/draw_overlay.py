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
"""Command line interface for drawing detection or tracking overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger("draw_overlay")
COCO: Dict[int, str] = {0: "person", 32: "sports ball"}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _class_label(value: int | str) -> str:
    """Return human readable class label."""

    try:
        cid = int(value)
    except (TypeError, ValueError):
        return str(value)
    return COCO.get(cid, str(cid))


def _class_color(value: int | str) -> Tuple[int, int, int]:
    """Generate a stable BGR color for a class label."""

    digest = hashlib.md5(str(value).encode()).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def _track_color(track_id: Optional[int]) -> Tuple[int, int, int]:
    """Generate a stable color for a track identifier."""

    digest = hashlib.md5(str(track_id).encode()).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def _load_records(json_path: Path, keys: Tuple[str, ...]) -> Dict[str, List[dict]]:
    """Load a JSON file in nested or flat schema.

    Args:
        json_path: Path to JSON file.
        keys: Possible keys holding the per-frame list (e.g. ``("detections",)``).

    Returns:
        Mapping of frame key to list of detection dictionaries.
    """

    with json_path.open() as fh:
        data = json.load(fh)
    result: Dict[str, List[dict]] = defaultdict(list)
    if not isinstance(data, list):
        raise ValueError("Invalid JSON structure: expected a list")
    if data and isinstance(data[0], dict) and "frame" in data[0] and any(
        k in data[0] for k in keys
    ):
        # Nested per-frame schema
        for rec in data:
            frame = rec.get("frame")
            items = None
            for k in keys:
                if isinstance(rec.get(k), list):
                    items = rec.get(k)
                    break
            if frame is None or items is None:
                continue
            for det in items:
                result[str(frame)].append(det)
    else:
        # Flat list
        for det in data:
            frame = det.get("frame")
            if frame is None:
                continue
            result[str(frame)].append(det)
    return result


def _load_detections(json_path: Path) -> Dict[str, List[dict]]:
    """Load detections from ``json_path`` into frame mapping."""

    return _load_records(json_path, ("detections",))


def _load_tracks(json_path: Path) -> Dict[str, List[dict]]:
    """Load tracks from ``json_path`` into frame mapping.

    Missing ``track_id`` fields are populated with ``None``.
    """

    frame_map = _load_records(json_path, ("tracks", "detections"))
    for dets in frame_map.values():
        for det in dets:
            if "track_id" not in det:
                for key in ("track_id", "id", "track"):
                    if key in det:
                        det["track_id"] = det[key]
                        break
            if "track_id" not in det:
                det["track_id"] = None
    return frame_map


def _resolve_frame_path(frames_dir: Path, frame_key: str | int) -> Tuple[Optional[Path], Optional[int]]:
    """Resolve ``frame_key`` to an existing image path and numeric index."""

    if isinstance(frame_key, int):
        idx = frame_key
        for ext in ("png", "jpg", "jpeg"):
            candidate = frames_dir / f"frame_{idx:06d}.{ext}"
            if candidate.exists():
                return candidate, idx
        return None, idx

    path = frames_dir / str(frame_key)
    if path.exists():
        match = re.search(r"(\d+)(?!.*\d)", path.name)
        idx = int(match.group(1)) if match else None
        return path, idx

    match = re.search(r"(\d+)(?!.*\d)", str(frame_key))
    if match:
        idx = int(match.group(1))
        for ext in ("png", "jpg", "jpeg"):
            candidate = frames_dir / f"frame_{idx:06d}.{ext}"
            if candidate.exists():
                return candidate, idx
        return None, idx
    return None, None


def _imread_safe(path: Path) -> Optional[np.ndarray]:
    """Read image from ``path`` using OpenCV with Pillow fallback."""

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        return np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
    except Exception:  # pragma: no cover - pillow failure
        return None


def _parse_class_filter(value: Optional[str]) -> Tuple[set[str], set[int]]:
    """Parse ``--only-class`` filter value."""

    names: set[str] = set()
    ids: set[int] = set()
    if not value:
        return names, ids
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.add(int(item))
        except ValueError:
            names.add(item)
    return names, ids


# ---------------------------------------------------------------------------
# Drawing logic
# ---------------------------------------------------------------------------

def _draw_overlay(
    frames_dir: Path,
    frame_map: Dict[str, List[dict]],
    output_dir: Path,
    label: bool,
    show_id: bool,
    score_min: float,
    allowed_names: set[str],
    allowed_ids: set[int],
    thickness: int,
    font_scale: float,
    start: int,
    end: int,
    max_frames: int,
    mode: str,
) -> int:
    """Draw overlays for ``frame_map``.

    Returns:
        Number of frames written.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    items: List[Tuple[int, Path, List[dict]]] = []
    for frame_key, dets in frame_map.items():
        path, idx = _resolve_frame_path(frames_dir, frame_key)
        if path is None or idx is None:
            LOGGER.warning("Skipping frame %s: cannot resolve", frame_key)
            continue
        items.append((idx, path, dets))
    items.sort(key=lambda x: x[0])

    written = 0
    for idx, path, dets in items:
        # guard rails for range selection
        if idx < start:
            continue
        if end != -1 and idx > end:
            continue
        if max_frames and written >= max_frames:
            break

        img = _imread_safe(path)
        if img is None:
            LOGGER.warning("Failed to read %s", path)
            continue
        h, w = img.shape[:2]
        for det in dets:
            score = float(det.get("score", 0.0))
            if score < score_min:
                continue
            cls_val = det.get("class", "")
            cname = _class_label(cls_val)
            cid: Optional[int]
            try:
                cid = int(cls_val)
            except (TypeError, ValueError):
                cid = None
            if (allowed_names and cname not in allowed_names) or (
                allowed_ids and (cid is None or cid not in allowed_ids)
            ):
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                LOGGER.warning("Invalid bbox for frame %s", path.name)
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                LOGGER.warning("Degenerate bbox for frame %s", path.name)
                continue
            if mode == "track":
                color = _track_color(det.get("track_id"))
            else:
                color = _class_color(cls_val)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            text_bits: List[str] = []
            if label:
                txt = cname
                if det.get("score") is not None:
                    txt += f" {score:.2f}"
                text_bits.append(txt)
            if show_id and mode == "track":
                tid = det.get("track_id")
                if tid is not None:
                    text_bits.append(f"#{tid}")
            if text_bits:
                text = " ".join(text_bits)
                cv2.putText(
                    img,
                    text,
                    (x1, max(0, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    max(1, thickness - 1),
                    cv2.LINE_AA,
                )
        out_path = output_dir / path.name
        if cv2.imwrite(str(out_path), img):
            written += 1
            if written % 10 == 0 or written == 1:
                LOGGER.info("frame %s saved to %s", path.name, out_path)
    LOGGER.info("Saved %d frame(s) to %s", written, output_dir)
    return written


def _export_mp4(output_dir: Path, mp4_path: Path, fps: int, crf: int) -> None:
    """Export frames in ``output_dir`` to an MP4 using ffmpeg image2 demuxer.

    To guarantee ordering, stage symlinks as 000001.png, 000002.png, ...
    """

    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(f for f in output_dir.iterdir() if f.suffix.lower() in exts)
    if not files:
        LOGGER.warning("No frames to export at %s", output_dir)
        return
    tmp = output_dir / "_mp4_stage"
    if tmp.exists():
        for g in tmp.iterdir():
            try:
                g.unlink()
            except Exception:
                pass
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(files, start=1):
        link = tmp / f"{i:06d}.png"
        try:
            link.symlink_to(f)
        except (OSError, NotImplementedError):
            try:
                link.hardlink_to(f)
            except Exception:
                import shutil
                shutil.copyfile(f, link)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-f",
        "image2",
        "-i",
        str(tmp / "%06d.png"),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True, help="Input frames directory")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save annotated frames"
    )
    parser.add_argument("--detections-json", type=Path, help="Detections JSON file")
    parser.add_argument("--tracks-json", type=Path, help="Tracks JSON file")
    parser.add_argument(
        "--mode",
        choices=["auto", "detect", "track"],
        default="auto",
        help="Overlay mode (default: auto)",
    )
    parser.add_argument("--label", action="store_true", help="Draw class labels and scores")
    parser.add_argument("--id", action="store_true", help="Draw track IDs")
    parser.add_argument(
        "--score-min", type=float, default=0.0, help="Minimum score threshold"
    )
    parser.add_argument(
        "--only-class",
        type=str,
        help="Comma separated list of class names or IDs to keep",
    )
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box thickness")
    parser.add_argument("--font-scale", type=float, default=0.5, help="Text font scale")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit number of frames")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (-1 = last)")
    parser.add_argument("--export-mp4", type=Path, help="Optional MP4 export path")
    parser.add_argument("--fps", type=int, default=25, help="FPS for MP4 export")
    parser.add_argument("--crf", type=int, default=23, help="CRF for MP4 export")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)

    if args.mode == "auto":
        if args.tracks_json and args.tracks_json.exists():
            mode = "track"
        elif args.detections_json and args.detections_json.exists():
            mode = "detect"
        else:
            LOGGER.error("No detections or tracks JSON provided")
            return 1
    else:
        mode = args.mode

    if mode == "detect" and not args.detections_json:
        LOGGER.error("--detections-json required for detect mode")
        return 1
    if mode == "track" and not args.tracks_json:
        LOGGER.error("--tracks-json required for track mode")
        return 1

    if mode == "detect":
        frame_map = _load_detections(args.detections_json)
    else:
        frame_map = _load_tracks(args.tracks_json)

    allowed_names, allowed_ids = _parse_class_filter(args.only_class)
    written = _draw_overlay(
        args.frames_dir,
        frame_map,
        args.output_dir,
        args.label,
        args.id,
        args.score_min,
        allowed_names,
        allowed_ids,
        args.thickness,
        args.font_scale,
        args.start,
        args.end,
        args.max_frames,
        mode,
    )

    if written and args.export_mp4:
        try:
            _export_mp4(args.output_dir, args.export_mp4, args.fps, args.crf)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - ffmpeg failure
            LOGGER.error("ffmpeg failed: %s", exc)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
