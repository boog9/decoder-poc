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
import glob
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

try:  # pragma: no cover - optional dependency
    from services.court_detector.court_reference_tcd import CANONICAL_LINES
except Exception:  # pragma: no cover - missing optional dep
    CANONICAL_LINES: List[List[Tuple[float, float]]] = []

LOGGER = logging.getLogger("draw_overlay")
CLASS_MAP: Dict[int, str] = {0: "person", 32: "sports ball", 100: "tennis_court"}
PALETTE_SEED = 0
COURT_CLASS_ID = 100


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class FlexibleBool(argparse.Action):
    """Boolean flag that supports both value and ``--no-`` forms.

    Examples:
        ``--flag`` → ``True``
        ``--flag=false`` or ``--flag false`` → ``False``
        ``--no-flag`` → ``False`` (value ignored)
    """

    TRUE_SET = {"1", "true", "t", "yes", "y", "on"}
    FALSE_SET = {"0", "false", "f", "no", "n", "off"}

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any | None,
        option_string: str | None = None,
    ) -> None:
        """Set destination value based on ``values`` and ``option_string``."""

        if option_string and option_string.startswith("--no-"):
            setattr(namespace, self.dest, False)
            return

        if values is None:
            setattr(namespace, self.dest, True)
            return

        v = str(values).strip().lower()
        if v in self.TRUE_SET:
            setattr(namespace, self.dest, True)
        elif v in self.FALSE_SET:
            setattr(namespace, self.dest, False)
        else:  # pragma: no cover - argparse already guards
            parser.error(
                f"{option_string} expects a boolean (true/false), got {values!r}"
            )


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

    if not hasattr(poly, "bounds") or not hasattr(poly, "area"):
        return False
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


# ---------- ROI helpers ----------
def _is_3x3(H: Any) -> bool:
    """Return True if ``H`` looks like a 3x3 matrix."""

    try:
        arr = np.asarray(H, dtype=float)
    except Exception:
        return False
    return arr.shape == (3, 3) and np.isfinite(arr).all()


def _det3x3(H: Any) -> float:
    """Safely compute determinant of a 3x3 matrix."""

    try:
        return float(np.linalg.det(np.array(H, np.float64)))
    except Exception:
        return 0.0


def normalize_quad(poly_raw: Any) -> Optional[List[Tuple[float, float]]]:
    """Normalize polygon-like object to a 4-point list or return ``None``."""

    # shapely Polygon
    try:
        if hasattr(poly_raw, "geom_type") and hasattr(poly_raw, "exterior"):
            coords = list(poly_raw.exterior.coords)[:4]
            if len(coords) == 4:
                return [(float(x), float(y)) for (x, y, *_) in coords]
    except Exception:
        pass
    # geojson-like dict
    if isinstance(poly_raw, dict) and "coordinates" in poly_raw:
        try:
            coords = poly_raw["coordinates"][0][:4]
            if len(coords) == 4:
                return [(float(x), float(y)) for (x, y, *_) in coords]
        except Exception:
            return None
    # plain list/tuple
    if isinstance(poly_raw, (list, tuple)) and len(poly_raw) == 4:
        try:
            return [(float(p[0]), float(p[1])) for p in poly_raw]
        except Exception:
            return None
    return None


def resolve_roi_record(roi_raw: Any, i: int, fname: str) -> Optional[Dict[str, Any]]:
    """Resolve ROI record for frame ``i`` / ``fname`` from ``roi_raw``."""

    base = os.path.basename(fname)
    stem0 = f"frame_{i:06d}"
    candidates = [
        i,
        str(i),
        base,
        stem0 + ".png",
        stem0 + ".jpg",
        stem0 + ".jpeg",
    ]
    if i == 0:
        stem1 = "frame_000001"
        candidates += [stem1 + ".png", stem1 + ".jpg", stem1 + ".jpeg"]

    def pick_from_dict(D: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        for k in candidates:
            if k in D:
                return D[k], f"key:{k}"
        for v in D.values():
            if isinstance(v, dict):
                f = v.get("frame")
                if f and f in (*candidates, base):
                    return v, f"frame:{f}"
                if v.get("index") == i or v.get("frame_idx") == i + 1:
                    return v, "index/frame_idx"
        return None, "none"

    def pick_from_list(L: List[Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        if 0 <= i < len(L) and isinstance(L[i], dict):
            return L[i], "list[i]"
        for v in L:
            if not isinstance(v, dict):
                continue
            f = v.get("frame")
            if f and f in (*candidates, base):
                return v, f"frame:{f}"
            if v.get("index") == i or v.get("frame_idx") == i + 1:
                return v, "index/frame_idx"
        return None, "none"

    rec, how = (
        pick_from_dict(roi_raw)
        if isinstance(roi_raw, dict)
        else pick_from_list(roi_raw)
        if isinstance(roi_raw, list)
        else (None, "invalid-roi")
    )
    if rec:
        LOGGER.debug(f"[ROI] matched by {how} for frame={base}")
    else:
        LOGGER.warning(f"[ROI] no match for frame={base}")
    return rec

def _compute_court_lines(h: List[List[float]]) -> Dict[str, List[List[float]]]:
    """Project canonical court lines via homography ``h``.

    This helper is intended for tests and debugging.

    Args:
        h: Homography mapping canonical ``[0,1]`` coordinates to image pixels.

    Returns:
        Mapping ``line_<idx>`` to projected polylines in image space.
    """

    if not CANONICAL_LINES:
        return {}
    H = np.array(h, np.float32)
    out: Dict[str, List[List[float]]] = {}
    for i, poly in enumerate(CANONICAL_LINES):
        pts = np.array(poly, np.float32).reshape(-1, 1, 2)
        if hasattr(cv2, "perspectiveTransform"):
            prj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        else:  # pragma: no cover - cv2 minimal build
            homog = np.concatenate([pts.reshape(-1, 2), np.ones((pts.shape[0], 1), np.float32)], axis=1)
            prj = (homog @ H.T)
            prj = prj[:, :2] / prj[:, 2:3]
        out[f"line_{i}"] = prj.tolist()
    return out


def _order_poly(poly: Iterable[Iterable[float]]) -> np.ndarray:
    """Return polygon points ordered TL, TR, BR, BL.

    Args:
        poly: Iterable of four ``(x, y)`` points.

    Returns:
        NumPy array of shape ``(4, 2)`` ordered top-left, top-right,
        bottom-right, bottom-left.
    """

    P = np.asarray(list(poly), np.float32)
    s = P.sum(1)
    d = np.diff(P, axis=1).ravel()
    tl, br, tr, bl = np.argmin(s), np.argmax(s), np.argmin(d), np.argmax(d)
    return np.array([P[tl], P[tr], P[br], P[bl]], np.float32)


def _ensure_H(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return 3x3 homography either from JSON or computed from a quad."""

    H = rec.get("homography")
    if _is_3x3(H) and abs(_det3x3(H)) > 1e-9:
        LOGGER.debug("[ROI] homography: 3x3 from JSON")
        return np.asarray(H, dtype=float)
    quad = normalize_quad(rec.get("polygon"))
    if quad is not None:
        # IMPORTANT: bring polygon to canonical TL, TR, BR, BL order
        dst = _order_poly(quad)
        src = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        Hm = cv2.getPerspectiveTransform(src, dst)
        LOGGER.debug(
            "[ROI] homography: computed from quad (ordered TL,TR,BR,BL)"
        )
        return Hm
    return None


def _draw_canonical_lines(
    img: np.ndarray, H: np.ndarray, color: Tuple[int, int, int], thickness: int
) -> None:
    """Draw canonical tennis court lines projected by ``H``."""

    for poly in CANONICAL_LINES:
        pts = np.array(poly, np.float32).reshape(-1, 1, 2)
        prj = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(int)
        for a, b in zip(prj[:-1], prj[1:]):
            cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)


def _draw_diag_grid(
    img: np.ndarray, H: np.ndarray, color: Tuple[int, int, int], thickness: int
) -> None:
    """Draw diagnostic grid in canonical space for debugging."""

    pts = []
    for t in np.linspace(0, 1, 11):
        pts.append([(t, 0.0), (t, 1.0)])
        pts.append([(0.0, t), (1.0, t)])
    for poly in pts:
        arr = np.array(poly, np.float32).reshape(-1, 1, 2)
        prj = cv2.perspectiveTransform(arr, H).reshape(-1, 2).astype(int)
        cv2.line(img, tuple(prj[0]), tuple(prj[1]), color, thickness, cv2.LINE_AA)


def _project_and_draw_lines_dict(
    img: np.ndarray,
    lines: Dict[str, List[List[float]]],
    H: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Project and draw a dictionary of lines via homography ``H``."""

    for poly in lines.values():
        pts = np.array(poly, np.float32).reshape(-1, 1, 2)
        try:
            prj = cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(int)
        except Exception:
            continue
        for a, b in zip(prj[:-1], prj[1:]):
            cv2.line(img, tuple(a), tuple(b), color, thickness, cv2.LINE_AA)


def normalize_dets(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize detection records to a common schema.

    Args:
        rec: Record potentially containing ``detections`` or ``det`` arrays.

    Returns:
        List of dictionaries with ``cls``, ``conf`` and ``bbox`` fields.
    """

    out: List[Dict[str, Any]] = []
    arr = rec.get("detections")
    if arr is None:
        arr = rec.get("det")
    if not arr:
        return out
    for d in arr:
        if not isinstance(d, dict):
            continue
        cls_val = d.get("class", d.get("cls"))
        conf_val = d.get("score", d.get("conf"))
        box = d.get("bbox") or d.get("xyxy")
        if box and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            det_dict: Dict[str, Any] = {"bbox": [x1, y1, x2, y2]}
            if cls_val is not None:
                det_dict["cls"] = int(cls_val)
                det_dict["class"] = int(cls_val)
            if conf_val is not None:
                det_dict["conf"] = float(conf_val)
                det_dict["score"] = float(conf_val)
            for key in ("polygon", "lines", "homography", "placeholder"):
                if key in d:
                    det_dict[key] = d[key]
            out.append(det_dict)
    return out



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


def _load_detections(json_path: Path | None) -> Dict[str, List[dict]]:
    """Load detections from ``json_path`` into frame mapping."""

    if json_path is None:
        return {}
    with json_path.open() as fh:
        data = json.load(fh)
    by_frame: Dict[str, List[dict]] = {}
    if not isinstance(data, list):
        return by_frame
    for rec in data:
        if not isinstance(rec, dict):
            continue
        frame = rec.get("frame")
        if "detections" in rec or "det" in rec:
            norm = normalize_dets(rec)
        else:
            norm = normalize_dets({"detections": [rec]})
        if frame is not None:
            by_frame[str(frame)] = norm
    return by_frame


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
    delta = 0.05
    axes = [(0.0, 0.0), (1.0, 1.0)]
    for base in axes:
        p0, px, py = _apply_h(
            [base, (base[0] + delta, base[1]), (base[0], base[1] + delta)], h
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
def _draw_court(
    img: np.ndarray,
    court_rec: Dict[str, Any],
    draw_lines: bool = True,
    draw_poly: bool = True,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    alpha: float = 0.25,
    diag_grid: bool = False,
) -> None:
    """Draw court polygon with optional canonical lines.

    Args:
        img: Image in BGR format.
        court_rec: Record containing ``polygon`` and optional ``homography``.
        draw_lines: Whether to draw court lines.
        draw_poly: Whether to draw the outer polygon.
        color: Line color in BGR.
        thickness: Line thickness in pixels.
        alpha: Transparency factor for polygon fill.
        diag_grid: If ``True`` draw diagnostic grid instead of court lines.
    """

    H = _ensure_H(court_rec)
    if court_rec.get("placeholder", False):
        if H is None:
            return
        LOGGER.debug("[ROI] placeholder overridden by valid homography")

    poly = court_rec.get("polygon") or []
    has_poly = False
    pts_draw = None
    if poly:
        pts_src = list(poly.exterior.coords) if hasattr(poly, "exterior") else list(poly)
        has_poly = len(pts_src) >= 4
        if has_poly:
            try:
                pts_draw = np.asarray(pts_src, dtype=np.int32).reshape(-1, 1, 2)
            except Exception:
                pts_draw = None

    if draw_poly and has_poly and pts_draw is not None:
        overlay = img.copy() if hasattr(img, "copy") else img
        try:
            if hasattr(cv2, "fillPoly"):
                cv2.fillPoly(overlay, [pts_draw], color)
            if hasattr(cv2, "addWeighted"):
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
            if hasattr(cv2, "polylines"):
                cv2.polylines(img, [pts_draw], True, color, thickness)
        except Exception:
            if hasattr(cv2, "polylines"):
                cv2.polylines(img, [pts_draw], True, color, thickness)

    if not draw_lines:
        return
    lines = court_rec.get("lines") if isinstance(court_rec, dict) else None
    if H is not None:
        if diag_grid:
            _draw_diag_grid(img, H, color, thickness)
        elif lines:
            _project_and_draw_lines_dict(img, lines, H, color, thickness)
        else:
            _draw_canonical_lines(img, H, color, thickness)
    else:
        if lines:
            for poly in lines.values():
                pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], False, color, max(1, thickness // 2), cv2.LINE_AA)

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
    roi_raw: Any | None,
    only_court: bool,
    primary_map: Dict[int, int],
    draw_court_axes: bool = False,
    save_stride: int = 1,
    output_ext: str = "png",
    diag_grid: bool = False,
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
    frame_counter = 0
    for idx, path, dets in items:
        # guard rails for range selection
        if idx < start:
            continue
        if end != -1 and idx > end:
            continue
        if max_frames and written >= max_frames:
            break
        if frame_counter % save_stride != 0:
            frame_counter += 1
            continue
        frame_counter += 1

        img = _imread_safe(path)
        if img is None:
            LOGGER.warning("Failed to read %s", path)
            continue
        h, w = img.shape[:2]
        frame_roi = (resolve_roi_record(roi_raw, idx, path.name) if roi_raw else None)
        frame_poly = roi_poly
        placeholder = False
        if frame_roi:
            if frame_roi.get("polygon"):
                frame_poly = Polygon(frame_roi["polygon"])  # type: ignore[arg-type]
            placeholder = bool(frame_roi.get("placeholder"))

        out_path = output_dir / f"{path.stem}.{output_ext}"

        if roi_raw and frame_poly is not None:
            poly_pts = np.array(list(frame_poly.exterior.coords), np.float32)
            filtered: List[dict] = []
            for det in dets:
                box = det.get("bbox")
                if not box or len(box) != 4:
                    continue
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                if cv2.pointPolygonTest(poly_pts, (cx, cy), False) > 0:
                    filtered.append(det)
            dets = filtered

        if roi_raw:
            dets = [
                d
                for d in dets
                if not (d.get("class") == COURT_CLASS_ID and d.get("polygon"))
            ]

        if draw_court or draw_court_lines:
            if frame_poly is not None and _roi_is_frame(frame_poly):
                pass
            else:
                crec = {
                    "polygon": frame_poly,
                    "homography": frame_roi.get("homography") if frame_roi else None,
                    "lines": frame_roi.get("lines", {}) if frame_roi else {},
                    "placeholder": placeholder,
                }
                H_curr = _ensure_H(crec)
                if H_curr is not None:
                    crec["homography"] = H_curr.tolist()
                _draw_court(
                    img,
                    crec,
                    draw_lines=draw_court_lines,
                    draw_poly=draw_court,
                    thickness=thickness,
                    diag_grid=diag_grid,
                )
                if draw_court_axes and H_curr is not None:
                    _draw_axes(img, H_curr, thickness)

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
            write_params: List[int] = []
            if output_ext.lower() in {"jpg", "jpeg"}:
                write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
            if write_params:
                ok = cv2.imwrite(str(out_path), img, write_params)
            else:
                ok = cv2.imwrite(str(out_path), img)
            if ok:
                LOGGER.info("frame %s saved to %s", path.name, out_path)
                written += 1
            continue
        for det in dets:
            det_cls = det.get("class", det.get("cls"))
            if det_cls == COURT_CLASS_ID and det.get("polygon"):
                if draw_court or draw_court_lines:
                    _draw_court(
                        img,
                        det,
                        draw_lines=draw_court_lines,
                        draw_poly=draw_court,
                        thickness=thickness,
                        diag_grid=diag_grid,
                    )
                if draw_court_axes and det.get("homography"):
                    _draw_axes(img, det["homography"], thickness)
                continue
            score = float(det.get("score", det.get("conf", 0.0)))
            if score < confidence_thr:
                continue
            cls_val = det.get("class", det.get("cls", ""))
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
                if det.get("score") is not None or det.get("conf") is not None:
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
        write_params: List[int] = []
        if output_ext.lower() in {"jpg", "jpeg"}:
            write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        if write_params:
            ok = cv2.imwrite(str(out_path), img, write_params)
        else:
            ok = cv2.imwrite(str(out_path), img)
        if ok:
            LOGGER.info("frame %s saved to %s", path.name, out_path)
            written += 1
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


def _ffmpeg_try(cmd: List[str]) -> Tuple[int, str, str]:
    """Execute ``cmd`` and return ``(code, stdout, stderr)``."""
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError as exc:  # pragma: no cover - ffmpeg missing
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _encode_mp4(stage_dir: Path, out_path: Path, fps: int, crf_val: int) -> None:
    """Encode ``stage_dir`` PNG frames into ``out_path``.

    Ensures explicit ``-f mp4`` and uses a ``.tmp.mp4`` temporary file so that
    container detection does not depend on the filename extension. The function
    attempts ``libx264`` first and progressively falls back to alternative
    rate-control modes and codecs.
    """

    pngs = glob.glob(os.path.join(stage_dir, "*.png"))
    if not pngs:
        raise RuntimeError(f"No PNG frames in {stage_dir}")

    base = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(stage_dir, "*.png"),
    ]

    def run(codec: str, rc_args: List[str]) -> bool | str:
        tmp_out = out_path.with_suffix(".tmp.mp4")
        cmd = (
            base
            + ["-c:v", codec]
            + rc_args
            + [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(tmp_out),
            ]
        )
        print("[INFO] Running:", " ".join(shlex.quote(x) for x in cmd))
        code, _stdout, stderr = _ffmpeg_try(cmd)
        if code == 0:
            try:
                os.chmod(tmp_out, 0o664)
            except OSError:
                pass
            try:
                os.replace(tmp_out, out_path)
            except FileNotFoundError:  # pragma: no cover - odd ffmpeg success
                raise RuntimeError(
                    "FFmpeg reported success but no output file was produced."
                )
            print(
                f"[INFO] Exported MP4: {out_path} ({len(pngs)} frames @ {fps} fps)"
            )
            return True
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass
        return stderr

    err: bool | str
    if crf_val >= 0:
        err = run("libx264", ["-crf", str(crf_val)])
        if err is True:
            return
        if isinstance(err, str) and (
            "Unrecognized option 'crf'" in err or "Option not found" in err
        ):
            print(
                f"[INFO] Switching RC: libx264 -crf unsupported -> using -x264-params crf={crf_val}"
            )
            err = run("libx264", ["-x264-params", f"crf={crf_val}"])
            if err is True:
                return
            print(
                f"[INFO] Switching RC: libx264 -x264-params failed -> using -qp {crf_val}"
            )
            err = run("libx264", ["-qp", str(crf_val)])
            if err is True:
                return
    else:
        err = run("libx264", [])
        if err is True:
            return

    print(f"[INFO] Falling back to h264_nvenc (cq≈{crf_val + 1})")
    err2 = run(
        "h264_nvenc",
        ["-cq", str(max(1, min(51, crf_val + 1)))] if crf_val >= 0 else [],
    )
    if err2 is True:
        return
    if isinstance(err2, str) and (
        "Unrecognized option 'cq'" in err2 or "Option not found" in err2
    ):
        print(
            f"[INFO] Switching RC: h264_nvenc -cq unsupported -> using -rc constqp -qp {crf_val + 2}"
        )
        err2 = run(
            "h264_nvenc",
            ["-rc", "constqp", "-qp", str(max(0, min(51, crf_val + 2)))]
            if crf_val >= 0
            else ["-rc", "constqp", "-qp", "23"],
        )
        if err2 is True:
            return

    print("[INFO] Falling back to mpeg4 (qscale=2)")
    err3 = run("mpeg4", ["-qscale:v", "2"])
    if err3 is True:
        return

    last_err = err3 if isinstance(err3, str) else (
        err2 if isinstance(err2, str) else (err if isinstance(err, str) else "unknown error")
    )
    raise RuntimeError("FFmpeg export failed:\n" + "\n".join(str(last_err).splitlines()[:40]))


def _export_mp4(output_dir: Path, mp4_path: Path, fps: int, crf: int) -> None:
    """Stage frames and export an MP4 with robust fallbacks."""

    stage_dir = _stage_frames(output_dir)
    if not any(stage_dir.iterdir()):
        LOGGER.warning("No frames to export at %s", output_dir)
        return
    _encode_mp4(stage_dir, mp4_path, fps, crf)


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
        choices=["detect", "track", "class", "roi"],
        default="detect",
        help="Overlay mode",
    )
    parser.add_argument(
        "--diag-grid",
        action="store_true",
        help="Draw diagnostic grid instead of canonical lines (debug)",
    )
    parser.add_argument(
        "--label", action="store_true", help="Draw class labels and scores"
    )
    parser.add_argument("--id", action="store_true", help="Draw track IDs")
    parser.add_argument(
        "--draw-court",
        "--no-draw-court",
        dest="draw_court",
        nargs="?",
        default=True,
        action=FlexibleBool,
        help="Draw tennis court polygon (accepts true/false; also supports --no-draw-court).",
    )
    parser.add_argument(
        "--draw-court-lines",
        "--no-draw-court-lines",
        dest="draw_court_lines",
        nargs="?",
        default=True,
        action=FlexibleBool,
        help="Draw internal court lines (accepts true/false; also supports --no-draw-court-lines).",
    )
    parser.add_argument(
        "--draw-court-axes",
        "--no-draw-court-axes",
        dest="draw_court_axes",
        nargs="?",
        default=False,
        action=FlexibleBool,
        help="Draw small court axes (accepts true/false; also supports --no-draw-court-axes).",
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
    parser.add_argument(
        "--roi-json",
        type=Path,
        help="ROI polygon JSON or court.json (per-frame entries)",
    )
    parser.add_argument(
        "--only-court",
        "--no-only-court",
        dest="only_court",
        nargs="?",
        default=False,
        action=FlexibleBool,
        help="Draw only the court contour (accepts true/false; also supports --no-only-court).",
    )
    parser.add_argument(
        "--primary-id-stick",
        type=int,
        default=0,
        help="Sticky ID label for main player (0=disabled)",
    )
    parser.add_argument(
        "--save-stride",
        type=int,
        default=1,
        help="Save every Nth frame",
    )
    parser.add_argument(
        "--output-ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image extension",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv)

    mode = args.mode

    roi_path = args.roi_json
    if roi_path is None:
        default_court = Path.cwd() / "court.json"
        if default_court.exists():
            roi_path = default_court

    if mode in {"class", "detect"} and not args.detections_json:
        LOGGER.error("--detections-json required for detect/class modes")
        return 1
    if mode == "track" and not args.tracks_json:
        LOGGER.error("--tracks-json required for track mode")
        return 1
    if mode == "roi" and roi_path is None:
        LOGGER.error("--roi-json required for roi mode")
        return 1

    global PALETTE_SEED, CLASS_MAP
    PALETTE_SEED = args.palette_seed
    if args.class_map:
        try:
            CLASS_MAP.update(_load_class_map(args.class_map))
        except Exception as exc:  # pragma: no cover - file errors
            LOGGER.error("Failed to load class map: %s", exc)
            return 1

    roi_raw: Any | None = None
    if roi_path and roi_path.exists():
        with roi_path.open() as fh:
            roi_raw = json.load(fh)

    if mode == "roi":
        frames_list: List[str] = []
        for ext in ("png", "jpg", "jpeg"):
            frames_list.extend(
                sorted(p.name for p in args.frames_dir.glob(f"frame_*.{ext}"))
            )
        frame_map = {name: [] for name in frames_list}
    elif mode in {"class", "detect"}:
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
    if roi_raw:
        if isinstance(roi_raw, dict):
            pts = roi_raw.get("polygon") or roi_raw.get("roi")
            if not pts:
                for v in roi_raw.values():
                    if isinstance(v, dict) and v.get("polygon"):
                        pts = v["polygon"]
                        break
            if pts:
                roi_poly = Polygon(pts)  # type: ignore[arg-type]
        elif isinstance(roi_raw, list):
            for item in roi_raw:
                if isinstance(item, dict):
                    pts = item.get("polygon") or item.get("roi")
                    if pts:
                        roi_poly = Polygon(pts)  # type: ignore[arg-type]
                        break

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
        roi_raw,
        args.only_court,
        primary_map,
        args.draw_court_axes,
        args.save_stride,
        args.output_ext,
        args.diag_grid,
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
