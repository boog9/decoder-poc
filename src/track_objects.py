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
"""ByteTrack based tracking for person and ball detections."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import inspect
from loguru import logger
try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # Pillow is an optional dependency

from .detect_objects import (
    _extract_frame_id,
    _normalize_bbox,
    _update_tracker,
    CLASS_ID_TO_NAME,
    CLASS_NAME_TO_ID,
    CLASS_ALIASES,
    _norm,
    _get_tlwh_from_track,
)
from .ball_kalman import BallKalmanFilter


def make_byte_tracker(
    *,
    track_thresh: float | None,
    min_score: float,
    match_thresh: float | None,
    track_buffer: int | None,
    fps: int,
):
    """Return a ``BYTETracker`` instance compatible with multiple forks.

    The constructor of :class:`BYTETracker` differs across forks. This helper
    inspects the available parameters and initialises the tracker using the
    appropriate signature while logging the chosen path.

    Args:
        track_thresh: Score threshold for variant B of the constructor. Falls
            back to ``min_score`` when ``None``.
        min_score: Minimum detection score; used as ``high_thresh`` for
            variant A.
        match_thresh: IoU matching threshold. Defaults to ``0.8`` when
            ``None``.
        track_buffer: Number of frames to keep lost tracks. Defaults to ``30``
            when ``None``.
        fps: Video frame rate.

    Returns:
        BYTETracker: Initialised tracker instance.
    """

    from bytetrack_vendor.tracker.byte_tracker import BYTETracker

    sig = inspect.signature(BYTETracker.__init__)
    params = sig.parameters

    high_thresh = min_score
    low_thresh = min(high_thresh * 0.5, 0.6)
    match = match_thresh if match_thresh is not None else 0.8
    buffer_ = track_buffer if track_buffer is not None else 30

    if "high_thresh" in params:
        logger.debug(
            "BYTETracker init variant A: high_thresh={:.3f}, low_thresh={:.3f}, "
            "match_thresh={:.3f}, track_buffer={}, fps={}",
            high_thresh,
            low_thresh,
            match,
            buffer_,
            fps,
        )
        return BYTETracker(
            high_thresh=high_thresh,
            low_thresh=low_thresh,
            match_thresh=match,
            track_buffer=buffer_,
            frame_rate=fps,
        )

    thresh = track_thresh if track_thresh is not None else high_thresh
    logger.debug(
        "BYTETracker init variant B: track_thresh={:.3f}, match_thresh={:.3f}, "
        "track_buffer={}, fps={}",
        thresh,
        match,
        buffer_,
        fps,
    )
    return BYTETracker(
        track_thresh=thresh,
        track_buffer=buffer_,
        match_thresh=match,
        frame_rate=fps,
    )


def _load_detections_grouped(
    path: Path, min_score: float
) -> Dict[int, Dict[int, List[dict]]]:
    """Return detections grouped by frame and class.

    Logs a warning if frame indices are not strictly increasing in the source
    file but always returns a numerically sorted dictionary.
    """

    with path.open() as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError("detections-json must contain a list")

    frames: Dict[int, Dict[int, List[dict]]] = {}
    prev = -1
    for item in raw:
        if "detections" in item:
            frame_id = _extract_frame_id(item.get("frame"))
            if frame_id is None:
                continue
            if frame_id < prev:
                logger.warning("Out-of-order frame %s after %s", frame_id, prev)
            prev = frame_id
            for det in item.get("detections", []):
                cls_val = det.get("class")
                if cls_val is None:
                    logger.warning("Skipping detection with null class in %s", det)
                    continue
                if isinstance(cls_val, str):
                    cls_key = CLASS_ALIASES.get(_norm(cls_val), _norm(cls_val))
                    if cls_key not in CLASS_NAME_TO_ID:
                        logger.warning(
                            "Skipping detection with unknown class '%s' in %s",
                            cls_val,
                            det,
                        )
                        continue
                    cls_id = CLASS_NAME_TO_ID[cls_key]
                else:
                    try:
                        cls_id = int(cls_val)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Skipping detection with invalid class %r in %s",
                            cls_val,
                            det,
                        )
                        continue
                    if cls_id not in CLASS_ID_TO_NAME:
                        logger.warning(
                            "Skipping detection with unknown class id %s in %s",
                            cls_id,
                            det,
                        )
                        continue
                score = float(det.get("score", 0.0))
                if score < min_score:
                    continue
                bbox = _normalize_bbox(det.get("bbox"))
                if bbox is None:
                    continue
                frames.setdefault(frame_id, {}).setdefault(cls_id, []).append(
                    {"bbox": bbox, "score": score}
                )
            continue

        frame_id = _extract_frame_id(item.get("frame"))
        if frame_id is None:
            continue
        if frame_id < prev:
            logger.warning("Out-of-order frame %s after %s", frame_id, prev)
        prev = frame_id
        cls_val = item.get("class")
        if cls_val is None:
            logger.warning("Skipping detection with null class in %s", item)
            continue
        if isinstance(cls_val, str):
            cls_key = CLASS_ALIASES.get(_norm(cls_val), _norm(cls_val))
            if cls_key not in CLASS_NAME_TO_ID:
                logger.warning(
                    "Skipping detection with unknown class '%s' in %s",
                    cls_val,
                    item,
                )
                continue
            cls_id = CLASS_NAME_TO_ID[cls_key]
        else:
            try:
                cls_id = int(cls_val)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping detection with invalid class %r in %s",
                    cls_val,
                    item,
                )
                continue
            if cls_id not in CLASS_ID_TO_NAME:
                logger.warning(
                    "Skipping detection with unknown class id %s in %s",
                    cls_id,
                    item,
                )
                continue
        score = float(item.get("score", 0.0))
        if score < min_score:
            continue
        bbox = _normalize_bbox(item.get("bbox"))
        if bbox is None:
            continue
        frames.setdefault(frame_id, {}).setdefault(cls_id, []).append(
            {"bbox": bbox, "score": score}
        )

    return dict(sorted(frames.items()))


def _extract_court_map(raw: list) -> Dict[int, List[List[float]]]:
    """Return mapping of frame index to court polygon.

    Supports both nested and flat detection formats and normalises class
    aliases. Only detections labelled as ``tennis_court`` are kept. Any
    entry without a valid polygon is skipped silently.
    """

    court_cls = CLASS_NAME_TO_ID.get("tennis_court")
    result: Dict[int, List[List[float]]] = {}
    for item in raw:
        if "detections" in item:
            frame_id = _extract_frame_id(item.get("frame"))
            if frame_id is None:
                continue
            for det in item.get("detections", []):
                cls_val = det.get("class")
                if isinstance(cls_val, str):
                    cls_key = CLASS_ALIASES.get(_norm(cls_val), _norm(cls_val))
                    if cls_key != "tennis_court":
                        continue
                else:
                    try:
                        if int(cls_val) != court_cls:
                            continue
                    except (TypeError, ValueError):
                        continue
                poly = det.get("polygon")
                if poly:
                    result[frame_id] = poly
            continue

        frame_id = _extract_frame_id(item.get("frame"))
        if frame_id is None:
            continue
        cls_val = item.get("class")
        if isinstance(cls_val, str):
            cls_key = CLASS_ALIASES.get(_norm(cls_val), _norm(cls_val))
            if cls_key != "tennis_court":
                continue
        else:
            try:
                if int(cls_val) != court_cls:
                    continue
            except (TypeError, ValueError):
                continue
        poly = item.get("polygon")
        if poly:
            result[frame_id] = poly
    return result


def _to_H3x3(h):
    """Return homography ``h`` as a 3x3 matrix.

    Accepts either a nested 3x3 list or a flat list of length 9 and returns a
    nested list representation. Values are returned as-is when ``h`` already
    appears to be 3x3.
    """

    if isinstance(h, list) and len(h) == 9 and not isinstance(h[0], list):
        return [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], h[8]],
        ]
    return h


def _extract_homography_map(raw: list) -> Dict[int, List[List[float]]]:
    """Return mapping of frame index to homography matrix."""

    court_cls = CLASS_NAME_TO_ID.get("tennis_court")
    result: Dict[int, List[List[float]]] = {}
    for item in raw:
        if item.get("placeholder"):
            continue
        if "detections" in item:
            frame_id = _extract_frame_id(item.get("frame"))
            if frame_id is None:
                continue
            for det in item.get("detections", []):
                if det.get("placeholder"):
                    continue
                cls_val = det.get("class")
                if isinstance(cls_val, str):
                    cls_key = CLASS_ALIASES.get(_norm(cls_val), _norm(cls_val))
                    if cls_key != "tennis_court":
                        continue
                else:
                    try:
                        if int(cls_val) != court_cls:
                            continue
                    except (TypeError, ValueError):
                        continue
                h = det.get("homography")
                if h:
                    H = _to_H3x3(h)
                    if (
                        isinstance(H, list)
                        and len(H) == 3
                        and all(isinstance(r, list) and len(r) == 3 for r in H)
                        and all(math.isfinite(v) for r in H for v in r)
                    ):
                        result[frame_id] = H
            continue

        frame_id = _extract_frame_id(item.get("frame"))
        if frame_id is None:
            continue
        cls_val = item.get("class")
        if isinstance(cls_val, str):
            cls_key = CLASS_ALIASES.get(_norm(cls_val), _norm(cls_val))
            if cls_key != "tennis_court":
                continue
        else:
            try:
                if int(cls_val) != court_cls:
                    continue
            except (TypeError, ValueError):
                continue
        h = item.get("homography")
        if h:
            H = _to_H3x3(h)
            if (
                isinstance(H, list)
                and len(H) == 3
                and all(isinstance(r, list) and len(r) == 3 for r in H)
                and all(math.isfinite(v) for r in H for v in r)
            ):
                result[frame_id] = H
    return result


def _is_identity_homography(h: List[List[float]]) -> bool:
    """Return ``True`` if ``h`` is close to identity."""

    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            if abs(h[i][j] - expected) > 1e-3:
                return False
    return True


def _apply_homography(h: List[List[float]], x: float, y: float) -> Tuple[float, float]:
    """Project ``(x, y)`` via homography ``h`` (3x3). Returns floats."""

    denom = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(denom) < 1e-9:
        return float(x), float(y)
    u = (h[0][0] * x + h[0][1] * y + h[0][2]) / denom
    v = (h[1][0] * x + h[1][1] * y + h[1][2]) / denom
    return float(u), float(v)


def _pre_min_area_quantile(detections: List[dict], q: float) -> float:
    """Return the quantile of person box area across all frames.

    Parameters
    ----------
    detections:
        Flat list of person detections with ``bbox`` in ``[x1, y1, x2, y2]``.
    q:
        Quantile in the range ``[0, 1]``.

    Returns
    -------
    float
        Area threshold below which detections should be discarded.
    """

    if q <= 0 or not detections:
        return 0.0
    areas = [
        max(0.0, (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        for d in detections
    ]
    if not areas:
        return 0.0
    areas.sort()
    pos = q * (len(areas) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(areas) - 1)
    weight = pos - lo
    return areas[lo] * (1 - weight) + areas[hi] * weight


def _poly_is_full_frame(pts: List[List[float]]) -> bool:
    """Return True if ``pts`` spans the entire frame."""  # tennis tuning

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    area = (maxx - minx) * (maxy - miny)
    poly_area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        poly_area += x1 * y2 - x2 * y1
    poly_area = abs(poly_area) / 2.0
    return (
        abs(poly_area - area) < 1e-3
        and minx <= 1
        and miny <= 1
        and (maxx >= 1000 or maxy >= 1000)
    )


def _pre_nms_persons(frame_dets: List[dict], iou_thr: float = 0.6) -> List[dict]:
    """Apply greedy NMS to person detections."""

    if iou_thr <= 0.0:
        return frame_dets

    def _iou_tlbr(a: List[float], b: List[float]) -> float:
        xa = max(a[0], b[0])
        ya = max(a[1], b[1])
        xb = min(a[2], b[2])
        yb = min(a[3], b[3])
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union else 0.0

    dets = sorted(frame_dets, key=lambda d: d.get("score", 0.0), reverse=True)
    keep: List[dict] = []
    for det in dets:
        if all(_iou_tlbr(det["bbox"], k["bbox"]) <= iou_thr for k in keep):
            keep.append(det)
    return keep


def _pre_court_gate(
    frame_id: int, frame_dets: List[dict], court_map: Dict[int, List[List[float]]]
) -> List[dict]:
    """Filter detections whose centre lies outside the court polygon."""

    poly = court_map.get(frame_id)
    if not poly:
        return frame_dets

    def _inside(x: float, y: float, pts: List[List[float]]) -> bool:
        inside = False
        n = len(pts)
        px1, py1 = pts[0]
        for i in range(1, n + 1):
            px2, py2 = pts[i % n]
            if ((py1 > y) != (py2 > y)) and (
                x < (px2 - px1) * (y - py1) / (py2 - py1 + 1e-9) + px1
            ):
                inside = not inside
            px1, py1 = px2, py2
        return inside

    out: List[dict] = []
    for det in frame_dets:
        bbox = det["bbox"]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if _inside(cx, cy, poly):
            out.append(det)
    return out


def _pre_topk_persons(frame_dets: List[dict], k: int = 3) -> List[dict]:
    """Keep only the top-``k`` person detections by score."""

    if k <= 0:
        return frame_dets
    return sorted(frame_dets, key=lambda d: d.get("score", 0.0), reverse=True)[:k]


def _stitch_predictive(
    tracks: List[dict],
    iou_thr: float = 0.55,
    max_gap: int = 5,
    max_speed: float = 50.0,
    aspect_tol: float = 0.35,
    scale_tol: float = 0.35,
) -> List[dict]:
    """Merge short track fragments based on predictive IoU matching."""
    alias: Dict[int, int] = {}

    def _canon(tid: int) -> int:
        while tid in alias:
            tid = alias[tid]
        return tid

    person_cls = CLASS_NAME_TO_ID.get("person")
    by_id: Dict[int, List[dict]] = {}
    for det in tracks:
        if det.get("class") != person_cls:
            continue
        by_id.setdefault(int(det["track_id"]), []).append(det)
    frags = []
    for tid, dets in by_id.items():
        dets.sort(key=lambda d: d["frame"])
        frags.append((tid, dets))
    frags.sort(key=lambda x: x[1][0]["frame"])

    for i in range(len(frags)):
        tid2, dets2 = frags[i]
        tid2 = _canon(tid2)
        start2 = dets2[0]["frame"]
        cx2 = dets2[0]["tlwh"][0] + dets2[0]["tlwh"][2] / 2.0
        cy2 = dets2[0]["tlwh"][1] + dets2[0]["tlwh"][3] / 2.0
        w2, h2 = dets2[0]["tlwh"][2], dets2[0]["tlwh"][3]
        best = None
        best_iou = 0.0
        for j in range(i):
            tid1, dets1 = frags[j]
            tid1 = _canon(tid1)
            end1 = dets1[-1]["frame"]
            gap = start2 - end1
            if gap <= 0 or gap > max_gap:
                continue
            if len(dets1) >= 2:
                c1_prev = dets1[-2]
            else:
                c1_prev = dets1[-1]
            cx1a = dets1[-1]["tlwh"][0] + dets1[-1]["tlwh"][2] / 2.0
            cy1a = dets1[-1]["tlwh"][1] + dets1[-1]["tlwh"][3] / 2.0
            cx1b = c1_prev["tlwh"][0] + c1_prev["tlwh"][2] / 2.0
            cy1b = c1_prev["tlwh"][1] + c1_prev["tlwh"][3] / 2.0
            dt = dets1[-1]["frame"] - c1_prev["frame"]
            vx = (cx1a - cx1b) / max(dt, 1)
            vy = (cy1a - cy1b) / max(dt, 1)
            pred_cx = cx1a + vx * gap
            pred_cy = cy1a + vy * gap
            dist = math.hypot(cx2 - pred_cx, cy2 - pred_cy)
            if dist > max_speed * gap:
                continue
            pred = [pred_cx - w2 / 2.0, pred_cy - h2 / 2.0, w2, h2]
            iou = _iou(pred, dets2[0]["tlwh"])
            if iou < iou_thr:
                continue
            aspect1 = dets1[-1]["tlwh"][2] / max(dets1[-1]["tlwh"][3], 1e-6)
            aspect2 = w2 / max(h2, 1e-6)
            if abs(aspect1 - aspect2) / max(aspect1, 1e-6) > aspect_tol:
                continue
            area1 = dets1[-1]["tlwh"][2] * dets1[-1]["tlwh"][3]
            area2 = w2 * h2
            if abs(area1 - area2) / max(area1, 1e-6) > scale_tol:
                continue
            if iou > best_iou:
                best_iou = iou
                best = tid1
        if best is not None:
            best = _canon(best)
        tgt = _canon(tid2)
        if best is not None and best != tgt:
            for det in dets2:
                det["track_id"] = best
            by_id.setdefault(best, []).extend(dets2)
            alias[tgt] = best
            by_id.pop(tgt, None)

    return tracks


def _smooth_tracks(
    tracks: List[dict],
    method: str = "ema",
    alpha: float = 0.3,
    window: int = 7,
) -> List[dict]:
    """Smooth track coordinates in-place for tracked items only.

    Applies to entries that have BOTH 'track_id' and 'tlwh'.
    Non-tracked items (e.g., court polygons) are left untouched.
    """

    if method == "none":
        return tracks

    from collections import defaultdict

    groups: Dict[int, List[dict]] = defaultdict(list)
    # Filter only tracked detections with TLWH
    for det in tracks:
        if "track_id" in det and "tlwh" in det:
            try:
                groups[int(det["track_id"])].append(det)
            except Exception:
                # Skip malformed IDs silently
                continue

    for dets in groups.values():
        if len(dets) < 2:
            # Nothing to smooth for 0/1-length tracks
            continue
        dets.sort(key=lambda d: d["frame"])

        if method == "ema":
            prev = dets[0]["tlwh"][:]
            for det in dets:
                cur = det["tlwh"]
                prev = [alpha * c + (1 - alpha) * p for c, p in zip(cur, prev)]
                det["tlwh"] = prev

        elif method == "sg":
            try:
                from scipy.signal import savgol_filter  # type: ignore
            except Exception:
                # SciPy not available, skip gracefully
                continue
            # Window must be odd and <= len(dets)
            if window % 2 == 1 and len(dets) >= window:
                series = [[d["tlwh"][i] for d in dets] for i in range(4)]
                for i in range(4):
                    series[i] = savgol_filter(series[i], window, 2).tolist()
                for d, x, y, w, h in zip(dets, *series):
                    d["tlwh"] = [float(x), float(y), float(w), float(h)]

        # Keep bbox in sync with tlwh
        for det in dets:
            x, y, w, h = det["tlwh"]
            det["bbox"] = [x, y, x + w, y + h]

    return tracks


def _refine_appearance(
    tracks: List[dict],
    frames_dir: Path,
    lambda_app: float = 0.3,
    iou_thr: float = 0.5,
) -> List[dict]:
    """Refine track IDs using HSV histogram similarity."""

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("appearance-refine skipped: OpenCV/NumPy not available")
        return tracks

    person_cls = CLASS_NAME_TO_ID.get("person")
    by_frame: Dict[int, List[dict]] = {}
    for det in tracks:
        if det.get("class") == person_cls:
            by_frame.setdefault(int(det["frame"]), []).append(det)

    def _hist(img, tlwh: List[float]):
        x, y, w, h = [int(v) for v in tlwh]
        cx = x + int(w * 0.45)
        cy = y + int(h * 0.45)
        cw = max(1, int(w * 0.1))
        ch = max(1, int(h * 0.1))
        patch = img[cy : cy + ch, cx : cx + cw]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 12, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    for frame in sorted(by_frame):
        prev = by_frame.get(frame - 1)
        if not prev:
            continue
        img_cur = cv2.imread(str(frames_dir / f"frame_{frame:06d}.png"))
        img_prev = cv2.imread(str(frames_dir / f"frame_{frame-1:06d}.png"))
        if img_cur is None or img_prev is None:
            continue
        for det in by_frame[frame]:
            hist_cur = _hist(img_cur, det["tlwh"])
            if hist_cur is None:
                continue
            best_id = det["track_id"]
            for prev_det in prev:
                if _iou(prev_det["tlwh"], det["tlwh"]) < iou_thr:
                    continue
                hist_prev = _hist(img_prev, prev_det["tlwh"])
                if hist_prev is None:
                    continue
                num = float(np.dot(hist_prev, hist_cur))
                denom = float(
                    np.linalg.norm(hist_prev) * np.linalg.norm(hist_cur) + 1e-6
                )
                if denom and num / denom > lambda_app:
                    best_id = prev_det["track_id"]
                    break
            det["track_id"] = best_id
    return tracks


def _filter_ball_candidates(
    dets: List[dict],
    frame_w: float,
    frame_h: float,
    kf: BallKalmanFilter,
    max_area_q: float,
    max_aspect: float,
    max_speed: float,
    max_accel: float,
    dt: float,
    H: List[List[float]] | None = None,
) -> List[dict]:
    """Select and validate a single ball detection.

    Args:
        dets: Candidate detections for the current frame.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        kf: Kalman filter instance tracking the ball centre.
        max_area_q: Maximum box area as a fraction of frame area.
        max_speed: Maximum speed in pixels per second.
        max_accel: Maximum acceleration in pixels per second squared.
        dt: Frame time step in seconds.
        H: Optional homography for court-plane distances.

    Returns:
        List containing at most one validated detection.
    """

    max_area_px = max_area_q * frame_w * frame_h
    valid: List[Tuple[dict, float, float]] = []
    for det in dets:
        x0, y0, x1, y1 = det["bbox"]
        w = x1 - x0
        h = y1 - y0
        area = w * h
        aspect = w / max(h, 1e-9)
        if area > max_area_px or not (0.6 <= aspect <= max_aspect):
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if H is not None:
            cx, cy = _apply_homography(H, cx, cy)
        valid.append((det, cx, cy))
    if not valid:
        kf.predict()
        return []

    pred = kf.predict()
    px, py = float(pred[0, 0]), float(pred[1, 0])
    best_det: dict | None = None
    best_c: tuple[float, float] | None = None
    best_dist = float("inf")
    for det, cx, cy in valid:
        if kf.last_pos is not None and kf.last_vel is not None:
            vx = (cx - kf.last_pos[0]) / dt
            vy = (cy - kf.last_pos[1]) / dt
            speed = math.hypot(vx, vy)
            ax = (vx - kf.last_vel[0]) / dt
            ay = (vy - kf.last_vel[1]) / dt
            accel = math.hypot(ax, ay)
            if speed > max_speed or accel > max_accel:
                continue
        dist = (px - cx) ** 2 + (py - cy) ** 2
        if dist < best_dist:
            best_det = det
            best_c = (cx, cy)
            best_dist = dist

    if best_det is None or best_c is None:
        return []
    kf.update(best_c)
    return [best_det]


def track_detections(
    detections_json: Path,
    output_json: Path,
    min_score: float,
    fps: int,
    reid_reuse_window: int,
    p_track_thresh: float,
    p_high_thresh: float,
    p_match_thresh: float,
    p_track_buffer: int,
    b_track_thresh: float,
    b_high_thresh: float,
    b_match_thresh: float,
    b_track_buffer: int,
    b_min_box_area: float,
    b_max_aspect_ratio: float,
    ball_max_area_q: float,
    ball_max_speed: float,
    ball_max_accel: float,
    pre_nms_iou: float = 0.0,
    pre_min_area_q: float = 0.0,
    pre_topk: int = 0,
    pre_court_gate: bool = False,
    half_court_gate: bool = False,
    assoc_plane: str = "pixel",
    assoc_w_iou: float = 1.0,
    assoc_w_plane: float = 0.0,
    assoc_plane_thresh: float = 0.25,
    court_json: Path | None = None,
    stitch: bool = False,
    stitch_iou: float = 0.55,
    stitch_gap: int = 5,
    stitch_speed: float = 50.0,
    stitch_aspect_tol: float = 0.35,
    stitch_scale_tol: float = 0.35,
    smooth: str = "none",
    smooth_alpha: float = 0.3,
    smooth_window: int = 7,
    appearance_refine: bool = False,
    appearance_lambda: float = 0.3,
    frames_dir: Path | None = None,
) -> None:
    """Track detections for persons and balls separately."""

    with detections_json.open() as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError("detections-json must contain a list")

    frame_sizes: Dict[int, Tuple[float, float]] = {}
    for item in raw:
        fid = _extract_frame_id(item.get("frame"))
        if fid is None:
            continue
        w = item.get("width")
        h = item.get("height")
        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
            frame_sizes[fid] = (float(w), float(h))

    default_size: Tuple[int, int] | None = None
    if not frame_sizes and frames_dir is not None and Image is not None:
        try:
            first = next(frames_dir.iterdir())
            with Image.open(first) as img:
                default_size = img.size  # (width, height)
        except Exception:
            default_size = None

    frames = _load_detections_grouped(detections_json, min_score)

    court_cls = CLASS_NAME_TO_ID.get("tennis_court")
    if court_json:
        try:
            with court_json.open() as fh:
                court_raw = json.load(fh)
        except Exception:
            court_raw = []
    else:
        court_raw = raw
    court_map = _extract_court_map(court_raw)
    homography_map = (
        _extract_homography_map(court_raw) if assoc_plane == "court" else {}
    )

    court_mid_y = None
    total_frames = len(frames)
    poly_frames = len(court_map)
    disabled_full_frame = False
    if court_map:
        first_poly = next(iter(court_map.values()))
        if _poly_is_full_frame(first_poly):
            court_map = {}  # tennis tuning: ignore full-frame polygons
            poly_frames = 0
            disabled_full_frame = True  # tennis tuning
            pre_court_gate = False
        else:
            ys = [p[1] for p in first_poly]
            court_mid_y = (min(ys) + max(ys)) / 2.0  # tennis tuning
    if pre_court_gate:
        if poly_frames > 0:
            logger.info(
                "court polygons available on {}/{} frames ({:.1f}%)",
                poly_frames,
                total_frames,
                (poly_frames / max(total_frames, 1)) * 100.0,
            )
        else:
            logger.warning("pre-court-gate enabled but no court polygons available")
    else:
        if disabled_full_frame:
            logger.info(
                "pre-court-gate disabled: detected full-frame court polygon (placeholder); "
                "provide a real court.json to enable gating"
            )

    min_area = 0.0
    if pre_min_area_q > 0:
        all_persons: List[dict] = []
        person_id = CLASS_NAME_TO_ID.get("person")
        for cls_map in frames.values():
            all_persons.extend(cls_map.get(person_id, []))
        min_area = _pre_min_area_quantile(all_persons, pre_min_area_q)
        logger.info(
            "pre-min-area-q {:.2f} -> min_area {:.2f}",
            pre_min_area_q,
            min_area,
        )

    if appearance_refine:
        if frames_dir is None:
            logger.warning("appearance-refine enabled but frames-dir not set; skipping")
            appearance_refine = False
        elif not any(frames_dir.glob("frame_*.png")):
            logger.warning(
                "appearance-refine enabled but no frames found in %s; skipping",
                frames_dir,
            )
            appearance_refine = False

    logger.info(
        "tracker params: p_match_thresh={:.2f} p_track_buffer={} reuse={} fps={}",
        p_match_thresh,
        p_track_buffer,
        reid_reuse_window,
        fps,
    )

    logger.info("tracking order: numeric by frame_index")

    trackers = {
        CLASS_NAME_TO_ID["person"]: make_byte_tracker(
            track_thresh=p_track_thresh,
            min_score=min_score,
            match_thresh=p_match_thresh,
            track_buffer=p_track_buffer,
            fps=fps,
        ),
        CLASS_NAME_TO_ID["sports ball"]: make_byte_tracker(
            track_thresh=b_track_thresh,
            min_score=min_score,
            match_thresh=b_match_thresh,
            track_buffer=b_track_buffer,
            fps=fps,
        ),
    }

    # Active and reuse caches for person and ball trackers
    logger.info(
        "Person tracker: match_thresh={:.2f} buffer={} reuse={}",
        p_match_thresh,
        p_track_buffer,
        reid_reuse_window,
    )

    active: Dict[int, Dict[int, Tuple[List[float], int]]] = {
        cid: {} for cid in trackers
    }
    reuse: Dict[int, List[dict]] = {cid: [] for cid in trackers}

    out: List[dict] = []
    ball_kf = BallKalmanFilter(1.0 / fps)

    for frame_id in sorted(frames):
        frame_w, frame_h = frame_sizes.get(frame_id, default_size or (0.0, 0.0))
        if not frame_w or not frame_h:
            max_x2 = 0.0
            max_y2 = 0.0
            for cls_map in frames[frame_id].values():
                for det in cls_map:
                    max_x2 = max(max_x2, det["bbox"][2])
                    max_y2 = max(max_y2, det["bbox"][3])
            poly = court_map.get(frame_id)
            if poly:
                for x, y in poly:
                    max_x2 = max(max_x2, x)
                    max_y2 = max(max_y2, y)
            frame_w = max_x2 or 1920.0
            frame_h = max_y2 or 1080.0

        for cls_id, tracker in trackers.items():
            dets = frames[frame_id].get(cls_id, [])
            if cls_id == CLASS_NAME_TO_ID["sports ball"]:
                # Keep only by min area here; aspect/area/speed/accel gating is centralized in _filter_ball_candidates
                dets = [
                    d
                    for d in dets
                    if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]) >= b_min_box_area
                ]
                H = None
                if assoc_plane == "court":
                    H = homography_map.get(frame_id)
                dets = _filter_ball_candidates(
                    dets,
                    frame_w,
                    frame_h,
                    ball_kf,
                    ball_max_area_q,
                    b_max_aspect_ratio,
                    ball_max_speed,
                    ball_max_accel,
                    1.0 / fps,
                    H,
                )
            else:
                n_start = len(dets)
                if pre_nms_iou > 0:
                    dets = _pre_nms_persons(dets, pre_nms_iou)
                if pre_court_gate:
                    dets = _pre_court_gate(frame_id, dets, court_map)
                if pre_topk > 0:
                    dets = _pre_topk_persons(dets, pre_topk)
                if min_area > 0:
                    dets = [
                        d
                        for d in dets
                        if (d["bbox"][2] - d["bbox"][0])
                        * (d["bbox"][3] - d["bbox"][1])
                        >= min_area
                    ]
                if half_court_gate and court_mid_y is not None and active[cls_id]:
                    for det in dets:
                        cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                        cy = (det["bbox"][1] + det["bbox"][3]) / 2.0
                        side_det = cy > court_mid_y
                        for bbox, _ in active[cls_id].values():
                            cx2 = bbox[0] + bbox[2] / 2.0
                            cy2 = bbox[1] + bbox[3] / 2.0
                            side_tr = cy2 > court_mid_y
                            if side_det != side_tr:
                                x0 = max(det["bbox"][0], bbox[0])
                                y0 = max(det["bbox"][1], bbox[1])
                                x1 = min(det["bbox"][2], bbox[0] + bbox[2])
                                y1 = min(det["bbox"][3], bbox[1] + bbox[3])
                                if x1 > x0 and y1 > y0:
                                    inter = (x1 - x0) * (y1 - y0)
                                    area_det = (det["bbox"][2] - det["bbox"][0]) * (
                                        det["bbox"][3] - det["bbox"][1]
                                    )
                                    area_tr = bbox[2] * bbox[3]
                                    iou = inter / (area_det + area_tr - inter + 1e-9)
                                    if iou > 0.1:
                                        det["score"] *= 0.5  # tennis tuning
                                        break
                if n_start and frame_id % 30 == 0:
                    removed = n_start - len(dets)
                    logger.debug(
                        "frame {}: pre-filters removed {}/{} person detections ({:.1f}%)",
                        frame_id,
                        removed,
                        n_start,
                        (removed / n_start) * 100.0,
                    )
            tlwhs = [
                (
                    b[0],
                    b[1],
                    b[2] - b[0],
                    b[3] - b[1],
                )
                for b in [d["bbox"] for d in dets]
            ]
            scores = [d["score"] for d in dets]
            classes = [cls_id] * len(dets)
            H = None
            court_dims: Tuple[float, float] | None = None
            if assoc_plane == "court":
                H = homography_map.get(frame_id)
                poly = court_map.get(frame_id)
                if H is not None and poly:
                    if _is_identity_homography(H) or _poly_is_full_frame(poly):
                        H = None
                    else:
                        pts = [_apply_homography(H, p[0], p[1]) for p in poly]
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        cw = max(xs) - min(xs)
                        ch = max(ys) - min(ys)
                        if cw <= 1e-3 or ch <= 1e-3:
                            logger.debug(
                                "homography rejected on frame %s: projected court too small (cw=%.4f, ch=%.4f)",
                                frame_id,
                                cw,
                                ch,
                            )
                            H = None
                        else:
                            court_dims = (cw, ch)
                else:
                    H = None
            tracks = _update_tracker(
                tracker,
                tlwhs,
                scores,
                classes,
                frame_id,
                homography=H if assoc_plane == "court" else None,
                assoc_w_iou=assoc_w_iou,
                assoc_w_plane=assoc_w_plane,
                plane_thresh=assoc_plane_thresh,
                court_dims=court_dims if assoc_plane == "court" else None,
                img_size=(frame_w, frame_h),
            )

            current_ids = set()
            for tr in tracks:
                tlwh = list(_get_tlwh_from_track(tr))
                tid = getattr(tr, "track_id", id(tr))
                reuse_id = _reuse_id(reuse[cls_id], tlwh, frame_id, reid_reuse_window)
                if reuse_id is not None:
                    tid = reuse_id
                out.append(
                    {
                        "frame": frame_id,
                        "class": cls_id,
                        "bbox": [
                            tlwh[0],
                            tlwh[1],
                            tlwh[0] + tlwh[2],
                            tlwh[1] + tlwh[3],
                        ],
                        "tlwh": tlwh,
                        "score": float(getattr(tr, "score", 1.0)),
                        "track_id": tid,
                    }
                )
                active[cls_id][tid] = (tlwh, frame_id)
                current_ids.add(tid)

            vanished = set(active[cls_id]) - current_ids
            for vid in vanished:
                bbox, last_frame = active[cls_id].pop(vid)
                reuse[cls_id].append({"id": vid, "bbox": bbox, "frame": last_frame})
            reuse[cls_id] = [
                r for r in reuse[cls_id] if frame_id - r["frame"] <= reid_reuse_window
            ]

    if court_cls is not None:
        for frame_id, poly in court_map.items():
            out.append(
                {
                    "frame": frame_id,
                    "class": court_cls,
                    "polygon": poly,
                    "score": 1.0,
                }
            )

    if stitch:
        out = _stitch_predictive(
            out,
            iou_thr=stitch_iou,
            max_gap=stitch_gap,
            max_speed=stitch_speed,
            aspect_tol=stitch_aspect_tol,
            scale_tol=stitch_scale_tol,
        )
    if appearance_refine and frames_dir is not None:
        out = _refine_appearance(
            out, frames_dir, lambda_app=appearance_lambda, iou_thr=0.5
        )
    if smooth != "none":
        out = _smooth_tracks(out, method=smooth, alpha=smooth_alpha, window=smooth_window)

    for item in out:
        item.pop("tlwh", None)

    out.sort(key=lambda d: (d["frame"], d.get("class", -1), d.get("track_id", -1)))
    with output_json.open("w") as fh:
        json.dump(out, fh, indent=2)


def _reuse_id(
    cache: List[dict], bbox: List[float], frame_id: int, window: int
) -> int | None:
    """Return cached ID if ``bbox`` matches a recently vanished track."""

    x, y, w, h = bbox
    for item in cache:
        if frame_id - item["frame"] > window:
            continue
        iou = _iou(bbox, item["bbox"])
        if iou >= 0.6:
            cache.remove(item)
            return item["id"]
    return None


def _iou(b1: List[float], b2: List[float]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union else 0.0


__all__ = ["track_detections", "_load_detections_grouped", "make_byte_tracker"]
