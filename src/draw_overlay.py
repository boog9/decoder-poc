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
import os
import re
import shlex
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shapely.geometry import Polygon  # tennis tuning

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - yaml is optional
    yaml = None  # type: ignore

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger("draw_overlay")
CLASS_MAP: Dict[int, str] = {0: "person", 32: "sports ball", 100: "tennis_court"}
PALETTE_SEED = 0
COURT_CLASS_ID = 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _class_label(value: int | str) -> str:
    """Return human readable class label."""

    try:
        cid = int(value)
    except (TypeError, ValueError):
        return str(value)
    return CLASS_MAP.get(cid, str(cid))


def _hash_color(key: str) -> Tuple[int, int, int]:
    """Return a deterministic BGR color for ``key``."""

    digest = hashlib.md5(f"{PALETTE_SEED}:{key}".encode()).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def _class_color(value: int | str) -> Tuple[int, int, int]:
    """Generate a stable BGR color for a class label."""

    return _hash_color(str(value))


def _track_color(track_id: Optional[int]) -> Tuple[int, int, int]:
    """Generate a stable color for a track identifier."""

    return _hash_color(str(track_id))


def _roi_is_frame(poly: "Polygon") -> bool:
    """Return True if ``poly`` spans the entire frame."""  # tennis tuning

    minx, miny, maxx, maxy = poly.bounds
    area = (maxx - minx) * (maxy - miny)
    return (
        abs(poly.area - area) < 1e-3
        and minx <= 1
        and miny <= 1
        and (maxx >= 1000 or maxy >= 1000)
    )


def _apply_h(pts: List[List[float]], h: List[List[float]]) -> List[List[float]]:
    """Project points via homography matrix.

    Args:
        pts: Points in world coordinates.
        h: 3x3 homography matrix.

    Returns:
        Transformed points in pixel coordinates.
    """

    H = np.array(h, dtype=float)
    arr = np.array(pts, dtype=float)
    ones = np.ones((arr.shape[0], 1))
    homog = np.hstack([arr, ones])
    prod = homog @ H.T
    prod /= prod[:, 2:3] + 1e-9
    return prod[:, :2].tolist()


def _compute_court_lines(h: List[List[float]]) -> Dict[str, List[List[float]]]:
    """Compute standard ITF tennis court lines using ``homography``.

    Args:
        h: 3x3 homography mapping court coordinates (metres) to pixels.

    Returns:
        Mapping of line name to list of two endpoints in pixel coordinates.
    """

    width = 10.97
    length = 23.77
    half_width = width / 2.0
    net_y = length / 2.0
    service_offset = 6.40
    mark_len = 0.1
    south_y = net_y - service_offset
    north_y = net_y + service_offset
    layout = {
        "baseline_south": [[0.0, 0.0], [width, 0.0]],
        "baseline_north": [[0.0, length], [width, length]],
        "service_south": [[0.0, south_y], [width, south_y]],
        "service_north": [[0.0, north_y], [width, north_y]],
        "service_center": [[half_width, south_y], [half_width, north_y]],
        "center_mark_south": [[half_width, 0.0], [half_width, mark_len]],
        "center_mark_north": [[half_width, length], [half_width, length - mark_len]],
    }
    return {name: _apply_h(pts, h) for name, pts in layout.items()}


def _load_class_map(path: Path) -> Dict[int, str]:
    """Load a mapping of class IDs to names from JSON or YAML."""

    with path.open() as fh:
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:  # pragma: no cover - optional
                raise RuntimeError("PyYAML is required to load YAML class maps")
            data = yaml.safe_load(fh)
        else:
            data = json.load(fh)
    mapping: Dict[int, str] = {}
    for k, v in (data or {}).items():
        try:
            mapping[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return mapping


def _validate_track_ids(frame_map: Dict[str, List[dict]]) -> None:
    """Ensure at least one ``track_id`` repeats across frames."""

    ids = [
        det.get("track_id")
        for dets in frame_map.values()
        for det in dets
        if det.get("track_id") is not None
    ]
    if ids and len(ids) == len(set(ids)):
        raise ValueError(
            "Looks like track IDs are not persistent across frames. "
            "Colors will flicker. Please fix track export."
        )


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
    if (
        data
        and isinstance(data[0], dict)
        and "frame" in data[0]
        and any(k in data[0] for k in keys)
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


# tennis tuning: tolerant ROI loader (dict or list)
def _load_roi(path: Path) -> Tuple["Polygon", Dict[str, List[List[float]]]]:
    """Load a court ROI polygon and optional lines from JSON file.

    Accepts:
      - dict with 'polygon'/'roi' and optional 'lines'
      - list of dicts with per-frame polygons; takes the first available polygon
    """

    with path.open() as fh:
        data = json.load(fh)

    lines: Dict[str, List[List[float]]] = {}
    pts: List[List[float]] | None = None

    if isinstance(data, dict):
        pts = data.get("polygon") or data.get("roi")
        lines = data.get("lines") or {}
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                cand = item.get("polygon") or item.get("roi")
                if cand:
                    pts = cand
                    lines = item.get("lines") or {}
                    break
    else:
        raise TypeError("ROI JSON must be a dict or a list")

    if not pts:
        raise ValueError("ROI JSON missing 'polygon'/'roi'")

    return Polygon(pts), lines


def _load_roi_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-frame ROI data from ``court.json``-style files.

    Args:
        path: Path to JSON file.

    Returns:
        Mapping of frame names to dictionaries with optional keys
        ``polygon``, ``lines``, ``homography`` and ``placeholder``.
    """

    with path.open() as fh:
        data = json.load(fh)

    mapping: Dict[str, Dict[str, Any]] = {}

    def _add_aliases(item: Dict[str, Any]) -> None:
        # Основний ключ — числовий індекс кадру
        if "frame" in item and item["frame"] is not None:
            try:
                idx = int(item["frame"])
                mapping[str(idx)] = item
                # Додаємо типові імена файлів для кадру
                mapping[f"frame_{idx:06d}.png"] = item
                mapping[f"frame_{idx:06d}.jpg"] = item
                mapping[f"frame_{idx:06d}.jpeg"] = item
            except (TypeError, ValueError):
                pass
        # Додатково: якщо явно вказано filename/path/name — теж мапимо
        for k in ("file", "name", "path"):
            v = item.get(k)
            if isinstance(v, str) and v:
                mapping[v] = item

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            _add_aliases(item)
            # Фолбек: якщо немає жодного ключа, класти за зростанням
            if not any(x in item for x in ("frame", "file", "name", "path")):
                mapping[str(len(mapping))] = item
    elif isinstance(data, dict):
        _add_aliases(data)
        if not any(x in data for x in ("frame", "file", "name", "path")):
            mapping["0"] = data
    else:
        raise TypeError("ROI JSON must be a dict or a list")
    return mapping


def _get_frame_roi(idx: int, path: Path, roi_map: Dict[str, Dict[str, Any]] | None) -> Optional[Dict[str, Any]]:
    """Підібрати ROI для кадру за кількома можливими ключами."""

    if not roi_map:
        return None
    candidates = [
        path.name,
        f"frame_{idx:06d}.png",
        f"frame_{idx:06d}.jpg",
        f"frame_{idx:06d}.jpeg",
        str(idx),
    ]
    for key in candidates:
        if key in roi_map:
            return roi_map[key]
    return None


def _resolve_frame_path(
    frames_dir: Path, frame_key: str | int
) -> Tuple[Optional[Path], Optional[int]]:
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


def _draw_axes(img: np.ndarray, h: List[List[float]], thickness: int) -> None:
    """Draw small coordinate axes in opposite court corners."""

    axes = [(0.0, 0.0), (10.97, 23.77)]
    for base in axes:
        p0, px, py = _apply_h(
            [base, (base[0] + 1.0, base[1]), (base[0], base[1] + 1.0)], h
        )
        x0, y0 = map(int, p0)
        x1, y1 = map(int, px)
        x2, y2 = map(int, py)
        if hasattr(cv2, "arrowedLine"):
            cv2.arrowedLine(
                img, (x0, y0), (x1, y1), (0, 0, 255), max(1, thickness // 2), tipLength=0.2
            )
            cv2.arrowedLine(
                img, (x0, y0), (x2, y2), (0, 255, 0), max(1, thickness // 2), tipLength=0.2
            )
        else:  # pragma: no cover - minimal OpenCV builds
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), max(1, thickness // 2))
            cv2.line(img, (x0, y0), (x2, y2), (0, 255, 0), max(1, thickness // 2))


# ---------------------------------------------------------------------------
# Drawing logic
# ---------------------------------------------------------------------------


def _draw_overlay(
    frames_dir: Path,
    frame_map: Dict[str, List[dict]],
    output_dir: Path,
    label: bool,
    show_id: bool,
    confidence_thr: float,
    allowed_names: set[str],
    allowed_ids: set[int],
    thickness: int,
    font_scale: float,
    start: int,
    end: int,
    max_frames: int,
    mode: str,
    draw_court: bool,
    draw_court_lines: bool,
    roi_poly: Polygon | None,
    roi_lines: Dict[str, List[List[float]]] | None,
    roi_map: Dict[str, Dict[str, Any]] | None,
    only_court: bool,
    primary_map: Dict[int, int],
    draw_court_axes: bool = False,
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
        frame_roi = _get_frame_roi(idx, path, roi_map)
        frame_poly = roi_poly
        frame_lines = roi_lines
        placeholder = False
        homography = None
        if frame_roi:
            if frame_roi.get("polygon"):
                frame_poly = Polygon(frame_roi["polygon"])  # type: ignore[arg-type]
            if frame_roi.get("lines"):
                frame_lines = frame_roi.get("lines")
            homography = frame_roi.get("homography")
            placeholder = bool(frame_roi.get("placeholder"))

        if draw_court_lines and (not frame_lines) and homography:
            frame_lines = _compute_court_lines(homography)

        if only_court and (frame_poly is None or _roi_is_frame(frame_poly)):
            for det in dets:
                if det.get("class") == COURT_CLASS_ID and det.get("polygon") and draw_court:
                    pts = np.array(
                        [[int(x), int(y)] for x, y in det["polygon"]], dtype=np.int32
                    )
                    cv2.polylines(img, [pts], True, (0, 255, 0), thickness)
                    if draw_court_lines and det.get("lines"):
                        for line in det["lines"].values():
                            lpts = np.array(
                                [[int(x), int(y)] for x, y in line], dtype=np.int32
                            )
                            cv2.polylines(
                                img, [lpts], False, (0, 255, 0), max(1, thickness // 2)
                            )
                    break
            if placeholder:
                cv2.putText(
                    img,
                    "*",
                    (5, int(20 * font_scale) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 2.0,
                    (0, 0, 255),
                    max(1, thickness),
                    cv2.LINE_AA,
                )
            _imwrite(output_dir / path.name, img)
            written += 1
            continue  # tennis tuning
        if draw_court and frame_poly is not None and not _roi_is_frame(frame_poly):
            pts = np.array(
                [[int(x), int(y)] for x, y in frame_poly.exterior.coords], dtype=np.int32
            )
            cv2.polylines(img, [pts], True, (0, 255, 0), thickness)
            if draw_court_lines and frame_lines:
                for line in frame_lines.values():
                    lpts = np.array(
                        [[int(x), int(y)] for x, y in line], dtype=np.int32
                    )
                    cv2.polylines(
                        img, [lpts], False, (0, 255, 0), max(1, thickness // 2)
                    )
            if draw_court_axes and homography:
                _draw_axes(img, homography, thickness)
            if only_court:
                if placeholder:
                    cv2.putText(
                        img,
                        "*",
                        (5, int(20 * font_scale) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale * 2.0,
                        (0, 0, 255),
                        max(1, thickness),
                        cv2.LINE_AA,
                    )
                _imwrite(output_dir / path.name, img)
                written += 1
                continue  # tennis tuning
        for det in dets:
            if det.get("class") == COURT_CLASS_ID and det.get("polygon"):
                if draw_court:
                    pts = np.array(
                        [[int(x), int(y)] for x, y in det["polygon"]], dtype=np.int32
                    )
                    cv2.polylines(img, [pts], True, (0, 255, 0), thickness)
                if draw_court_lines and det.get("lines"):
                    for line in det["lines"].values():
                        lpts = np.array(
                            [[int(x), int(y)] for x, y in line], dtype=np.int32
                        )
                        cv2.polylines(
                            img, [lpts], False, (0, 255, 0), max(1, thickness // 2)
                        )
                if draw_court_axes and det.get("homography"):
                    _draw_axes(img, det["homography"], thickness)
                continue
            score = float(det.get("score", 0.0))
            if score < confidence_thr:
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
            disp_tid: Optional[int] = None
            if mode == "track":
                tid = det.get("track_id")
                disp_tid = primary_map.get(tid, tid)
                color = _track_color(disp_tid)
            else:
                color = _class_color(cls_val)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            text_bits: List[str] = []
            if show_id and mode == "track":
                if disp_tid is not None:
                    text_bits.append(f"#{disp_tid}")
            if mode == "detect":
                text_bits.append(f"{score:.2f}")
            elif label:
                text_bits.append(cname)
                if det.get("score") is not None:
                    text_bits.append(f"{score:.2f}")
            if text_bits:
                text = "  ".join(text_bits)
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
        if placeholder:
            cv2.putText(
                img,
                "*",
                (5, int(20 * font_scale) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 2.0,
                (0, 0, 255),
                max(1, thickness),
                cv2.LINE_AA,
            )
        out_path = output_dir / path.name
        if cv2.imwrite(str(out_path), img):
            written += 1
            if written % 10 == 0 or written == 1:
                LOGGER.info("frame %s saved to %s", path.name, out_path)
    LOGGER.info("Saved %d frame(s) to %s", written, output_dir)
    return written


def _run_command(cmd: List[str]) -> None:
    """Run a subprocess command with logging and error handling.

    Args:
        cmd: Command tokens.

    Raises:
        RuntimeError: If the command exits with a non-zero status.
    """
    LOGGER.info("Running: %s", shlex.join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "") + (proc.stdout or "")
        snippet = "\n".join(stderr.splitlines()[:40])
        raise RuntimeError(f"Command failed ({proc.returncode}):\n{snippet}")


def _stage_frames(output_dir: Path) -> Path:
    """Create numbered symlinks for ffmpeg export.

    Args:
        output_dir: Directory containing frame images.

    Returns:
        Path to staging directory with numbered symlinks.
    """
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(f for f in output_dir.iterdir() if f.suffix.lower() in exts)
    stage_dir = output_dir / "_mp4_stage"
    stage_dir.mkdir(exist_ok=True)
    for entry in stage_dir.iterdir():
        try:
            entry.unlink()
        except Exception:
            pass
    for i, src in enumerate(files, start=1):
        link = stage_dir / f"{i:06d}.png"
        if link.exists() or link.is_symlink():
            link.unlink()
        rel = os.path.relpath(src, start=stage_dir)
        try:
            link.symlink_to(rel)
        except (OSError, NotImplementedError):
            try:
                link.hardlink_to(src)
            except Exception:
                shutil.copyfile(src, link)
    return stage_dir


def _export_mp4(output_dir: Path, mp4_path: Path, fps: int, crf: int) -> None:
    """Export frames in ``output_dir`` to an MP4 using ffmpeg.

    Symlinks are staged in ``_mp4_stage`` to guarantee ordering. The function
    prefers ``libx264`` and falls back to ``h264_nvenc`` or ``mpeg4`` when the
    preferred codec is unavailable.
    """
    stage_dir = _stage_frames(output_dir)
    if not any(stage_dir.iterdir()):
        LOGGER.warning("No frames to export at %s", output_dir)
        return
    pattern = str(stage_dir / "*.png")
    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
    ]

    has_x264 = (
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "encoder=libx264"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    codec_cmd: List[str]
    if has_x264:
        codec = "libx264"
        codec_cmd = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
        if crf is not None and crf >= 0:
            codec_cmd += ["-crf", str(crf)]
    else:
        has_nvenc = (
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-h", "encoder=h264_nvenc"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if has_nvenc:
            codec = "h264_nvenc"
            codec_cmd = ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p"]
            if crf is not None and crf >= 0:
                codec_cmd += ["-crf", str(crf)]
        else:
            codec = "mpeg4"
            codec_cmd = ["-c:v", "mpeg4", "-qscale:v", "2", "-pix_fmt", "yuv420p"]
    LOGGER.info("Using codec %s for MP4 export", codec)
    cmd = base + codec_cmd + [str(mp4_path)]
    _run_command(cmd)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir", type=Path, required=True, help="Input frames directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save annotated frames",
    )
    parser.add_argument("--detections-json", type=Path, help="Detections JSON file")
    parser.add_argument("--tracks-json", type=Path, help="Tracks JSON file")
    parser.add_argument(
        "--mode",
        choices=["auto", "class", "track", "detect"],
        default="auto",
        help="Overlay mode (default: auto)",
    )
    parser.add_argument(
        "--label", action="store_true", help="Draw class labels and scores"
    )
    parser.add_argument("--id", action="store_true", help="Draw track IDs")
    parser.add_argument(
        "--draw-court",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw tennis court polygon",
    )
    parser.add_argument(
        "--draw-court-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw internal court lines when available",
    )
    parser.add_argument(
        "--draw-court-axes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw small court axes when homography is known",
    )
    parser.add_argument(
        "--confidence-thr", type=float, default=0.0, help="Minimum score threshold"
    )
    parser.add_argument(
        "--only-class",
        type=str,
        help="Comma separated list of class names or IDs to keep",
    )
    parser.add_argument(
        "--palette-seed", type=int, default=0, help="Seed for color palette"
    )
    parser.add_argument("--class-map", type=Path, help="JSON or YAML class name map")
    parser.add_argument(
        "--thickness", type=int, default=2, help="Bounding box thickness"
    )
    parser.add_argument("--font-scale", type=float, default=0.5, help="Text font scale")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Limit number of frames"
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument(
        "--end", type=int, default=-1, help="End frame index (-1 = last)"
    )
    parser.add_argument(
        "--export-mp4",
        type=str,
        help="Optional MP4 export path or full ffmpeg command",
    )
    parser.add_argument("--fps", type=int, default=25, help="FPS for MP4 export")
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="CRF for libx264; set -1 to disable",
    )
    parser.add_argument("--roi-json", type=Path, help="Court ROI polygon JSON")
    parser.add_argument(
        "--only-court",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw only the court contour",  # tennis tuning
    )
    parser.add_argument(
        "--primary-id-stick",
        type=int,
        default=0,
        help="Sticky ID label for main player (0=disabled)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)

    if args.mode == "auto":
        if args.tracks_json and args.tracks_json.exists():
            mode = "track"
        elif args.detections_json and args.detections_json.exists():
            mode = "class"
        else:
            LOGGER.error("No detections or tracks JSON provided")
            return 1
    else:
        mode = args.mode

    if mode in {"class", "detect"} and not args.detections_json:
        LOGGER.error("--detections-json required for class/detect mode")
        return 1
    if mode == "track" and not args.tracks_json:
        LOGGER.error("--tracks-json required for track mode")
        return 1

    global PALETTE_SEED, CLASS_MAP
    PALETTE_SEED = args.palette_seed
    if args.class_map:
        try:
            CLASS_MAP.update(_load_class_map(args.class_map))
        except Exception as exc:  # pragma: no cover - file errors
            LOGGER.error("Failed to load class map: %s", exc)
            return 1

    if mode in {"class", "detect"}:
        frame_map = _load_detections(args.detections_json)
    else:
        frame_map = _load_tracks(args.tracks_json)
        try:
            _validate_track_ids(frame_map)
        except ValueError as exc:
            LOGGER.error(str(exc))
            return 1

    allowed_names, allowed_ids = _parse_class_filter(args.only_class)

    roi_poly: Optional[Polygon] = None
    roi_lines: Optional[Dict[str, List[List[float]]]] = None
    roi_map: Optional[Dict[str, Dict[str, Any]]] = None
    if args.roi_json:
        roi_map = _load_roi_map(args.roi_json)
        if roi_map:
            first = next(iter(roi_map.values()))
            if first.get("polygon"):
                roi_poly = Polygon(first.get("polygon"))  # type: ignore[arg-type]
            if first.get("lines"):
                roi_lines = first.get("lines")

    primary_map: Dict[int, int] = {}
    if mode == "track" and args.primary_id_stick > 0:
        counts: Dict[int, int] = defaultdict(int)
        for dets in frame_map.values():
            for d in dets:
                if d.get("class") not in {0, "person"}:
                    continue
                tid = d.get("track_id")
                if tid is None:
                    continue
                bbox = d.get("bbox")
                if not bbox:
                    continue
                counts[tid] += 1
        if counts:
            primary = max(counts, key=counts.get)
            primary_map[primary] = args.primary_id_stick

    written = _draw_overlay(
        args.frames_dir,
        frame_map,
        args.output_dir,
        args.label,
        args.id,
        args.confidence_thr,
        allowed_names,
        allowed_ids,
        args.thickness,
        args.font_scale,
        args.start,
        args.end,
        args.max_frames,
        mode,
        args.draw_court,
        args.draw_court_lines,
        roi_poly,
        roi_lines,
        roi_map,
        args.only_court,
        primary_map,
        args.draw_court_axes,
    )

    if written == 0:
        LOGGER.warning("No overlays were drawn. Adjust filters or verify input files.")
    if written and args.export_mp4:
        try:
            val = args.export_mp4
            if val.strip().startswith("ffmpeg"):
                _run_command(shlex.split(val))
            else:
                _export_mp4(args.output_dir, Path(val), args.fps, args.crf)
        except RuntimeError as exc:  # pragma: no cover - ffmpeg failure
            LOGGER.error("ffmpeg failed: %s", exc)
            return 1
    return 0 if written else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
