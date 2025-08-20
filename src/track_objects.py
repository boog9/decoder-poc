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
from pathlib import Path
from typing import Dict, List, Tuple

import inspect
from loguru import logger

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
) -> None:
    """Track detections for persons and balls separately."""

    with detections_json.open() as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError("detections-json must contain a list")

    frames = _load_detections_grouped(detections_json, min_score)

    court_cls = CLASS_NAME_TO_ID.get("tennis_court")
    court_map = _extract_court_map(raw)

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

    active: Dict[int, Dict[int, Tuple[List[float], int]]] = {
        cid: {} for cid in trackers
    }
    reuse: Dict[int, List[dict]] = {cid: [] for cid in trackers}

    out: List[dict] = []

    for frame_id in sorted(frames):
        for cls_id, tracker in trackers.items():
            dets = frames[frame_id].get(cls_id, [])
            if cls_id == CLASS_NAME_TO_ID["sports ball"]:
                dets = [
                    d
                    for d in dets
                    if (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
                    >= b_min_box_area
                    and (
                        (d["bbox"][2] - d["bbox"][0])
                        / max(d["bbox"][3] - d["bbox"][1], 1e-3)
                    )
                    <= b_max_aspect_ratio
                ]
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
            tracks = _update_tracker(tracker, tlwhs, scores, classes, frame_id)

            current_ids = set()
            for tr in tracks:
                bbox = list(_get_tlwh_from_track(tr))
                tid = getattr(tr, "track_id", id(tr))
                reuse_id = _reuse_id(reuse[cls_id], bbox, frame_id, reid_reuse_window)
                if reuse_id is not None:
                    tid = reuse_id
                out.append(
                    {
                        "frame": frame_id,
                        "class": cls_id,
                        "bbox": [
                            bbox[0],
                            bbox[1],
                            bbox[0] + bbox[2],
                            bbox[1] + bbox[3],
                        ],
                        "score": float(getattr(tr, "score", 1.0)),
                        "track_id": tid,
                    }
                )
                active[cls_id][tid] = (bbox, frame_id)
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
