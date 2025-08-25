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
"""Detection merging utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..utils.classes import CLASS_NAME_TO_ID

BALL_ID = CLASS_NAME_TO_ID["sports ball"]


@dataclass
class Detection:
    """Simple representation of a detection."""

    bbox: List[float]
    score: float
    class_id: int
    source: str


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _wbf(dets: List[Detection], iou_thres: float) -> List[Detection]:
    clusters: List[List[Detection]] = []
    for det in dets:
        for cluster in clusters:
            if _iou(det.bbox, cluster[0].bbox) >= iou_thres:
                cluster.append(det)
                break
        else:
            clusters.append([det])
    fused: List[Detection] = []
    for cluster in clusters:
        total = sum(d.score for d in cluster)
        x1 = sum(d.bbox[0] * d.score for d in cluster) / total
        y1 = sum(d.bbox[1] * d.score for d in cluster) / total
        x2 = sum(d.bbox[2] * d.score for d in cluster) / total
        y2 = sum(d.bbox[3] * d.score for d in cluster) / total
        score = max(d.score for d in cluster)
        fused.append(
            Detection(bbox=[x1, y1, x2, y2], score=score, class_id=cluster[0].class_id, source="wbf")
        )
    return fused


def _nms(dets: List[Detection], iou_thres: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for det in dets:
        if all(_iou(det.bbox, k.bbox) < iou_thres for k in kept):
            kept.append(det)
    return kept


def merge_detections(
    det_lists: Iterable[List[Detection]],
    iou: Dict[int, float] | None = None,
    bonuses: Dict[str, Dict[int, float]] | None = None,
    topk: Dict[int, int] | None = None,
) -> List[Detection]:
    """Merge detections from multiple passes."""

    all_dets: List[Detection] = []
    for dets in det_lists:
        for d in dets:
            bonus = bonuses.get(d.source, {}).get(d.class_id, 0.0) if bonuses else 0.0
            all_dets.append(
                Detection(bbox=d.bbox, score=d.score + bonus, class_id=d.class_id, source=d.source)
            )

    merged: List[Detection] = []
    by_class: Dict[int, List[Detection]] = {}
    for d in all_dets:
        by_class.setdefault(d.class_id, []).append(d)
    for cid, dets in by_class.items():
        thr = iou.get(cid, 0.5) if iou else 0.5
        if cid == BALL_ID:
            merged_cls = _wbf(dets, thr)
        else:
            merged_cls = _nms(dets, thr)
        merged_cls = sorted(merged_cls, key=lambda d: d.score, reverse=True)
        if topk and cid in topk:
            merged_cls = merged_cls[: topk[cid]]
        merged.extend(merged_cls)
    return merged
