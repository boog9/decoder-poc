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
"""Object detection on a directory of frames using YOLOX.

This script requires a CUDA-enabled GPU and YOLOX 0.3+.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Sequence, Dict
import re

try:
    from shapely.geometry import box
except Exception:  # pragma: no cover - optional dependency
    box = None  # type: ignore

import numpy as np

from loguru import logger
import os
import sys

from .utils.classes import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID

# Add ByteTrack root to ``sys.path`` once and import ``BYTETracker`` normally.
# The tracker module lives under ``yolox.tracker.byte_tracker`` in the bundled
# repository at ``externals/ByteTrack``.
BT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../externals/ByteTrack")
)
if BT_ROOT not in sys.path:
    sys.path.insert(0, BT_ROOT)
try:
    from yolox.tracker.byte_tracker import BYTETracker
except Exception as exc:  # pragma: no cover - optional dependency
    logger.error("Could not import BYTETracker: {}", exc)
    BYTETracker = None

from PIL import Image
from tqdm import tqdm

import torch

try:
    from yolox.data.datasets import COCO_CLASSES

    YOLOX_NUM_CLASSES = len(COCO_CLASSES)
except Exception:  # pragma: no cover - optional dependency
    YOLOX_NUM_CLASSES = 80

if not torch.cuda.is_available():
    raise RuntimeError("CUDA device required for YOLOX")

LOGGER = logging.getLogger(__name__)

YOLOX_MODELS = {"yolox-s", "yolox-m", "yolox-l", "yolox-x"}

# Number of classes in the default COCO-trained YOLOX models.

# Alias mapping for name variants coming from different datasets.
CLASS_ALIASES = {"sports ball": "ball"}

# Map CLI model names to torch.hub callable names.
_YOLOX_MODEL_MAP = {
    "yolox-s": "yolox_s",
    "yolox-m": "yolox_m",
    "yolox-l": "yolox_l",
    "yolox-x": "yolox_x",
}

# Index of detections by (frame_id, detection_idx) for tracking.
_det_index: dict[tuple[int, int], dict] = {}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for detection and tracking."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=False)

    # Detection subcommand (default)
    det = sub.add_parser("detect", help="Run YOLOX detection")
    det.add_argument("--frames-dir", type=Path, required=True)
    det.add_argument("--output-json", type=Path, required=True)
    det.add_argument(
        "--model", type=str, default="yolox-s", choices=sorted(YOLOX_MODELS)
    )
    det.add_argument("--img-size", type=int, default=640)
    det.add_argument("--conf-thres", type=float, default=0.3)
    det.add_argument("--nms-thres", type=float, default=0.45)
    det.add_argument("--classes", nargs="+", type=int, default=None)

    # Tracking subcommand
    tr = sub.add_parser("track", help="Run ByteTrack on YOLOX detections")
    tr.add_argument("--detections-json", type=Path, required=True)
    tr.add_argument("--output-json", type=Path, required=True)
    tr.add_argument("--min-score", type=float, default=0.3)

    # Parse provided arguments or ``sys.argv`` when ``argv`` is ``None``.
    args = parser.parse_args(argv)

    # Default to ``detect`` when no subcommand is supplied.
    if args.command is None:
        args.command = "detect"
    return args


def _update_tracker(tracker, tlwhs, scores, classes, frame_id):
    """Call ``tracker.update`` with a compatible signature."""

    sig = inspect.signature(tracker.update)
    params = list(sig.parameters)
    if params and params[0] == "self":
        params = params[1:]

    if params == ["tlwhs", "scores", "classes", "frame_id"]:
        return tracker.update(tlwhs, scores, classes, frame_id)

    if params == ["tlwhs", "scores", "frame_id"]:
        return tracker.update(tlwhs, scores, frame_id)

    def _cls_id(c):
        if isinstance(c, int):
            return c
        return CLASS_NAME_TO_ID.get(CLASS_ALIASES.get(c, c), -1)

    if params == ["dets", "frame_id"]:
        cls_arr = np.array([_cls_id(c) for c in classes], dtype=np.float32)[
            :, None
        ]
        dets = np.concatenate(
            [
                np.asarray(tlwhs, dtype=np.float32),
                np.asarray(scores, dtype=np.float32)[:, None],
                cls_arr,
            ],
            axis=1,
        )
        return tracker.update(dets, frame_id)

    if params == ["output_results", "img_info", "img_size"]:
        cls_arr = np.array([_cls_id(c) for c in classes], dtype=np.float32)[
            :, None
        ]
        dets = np.concatenate(
            [
                np.asarray(tlwhs, dtype=np.float32),
                np.asarray(scores, dtype=np.float32)[:, None],
                cls_arr,
            ],
            axis=1,
        )

        im_w = max(b[0] + b[2] for b in tlwhs) if tlwhs else 1920
        im_h = max(b[1] + b[3] for b in tlwhs) if tlwhs else 1080
        img_info = (im_h, im_w, 1.0)
        img_size = (im_w, im_h)

        dets_tensor = torch.as_tensor(dets, dtype=torch.float32)
        return tracker.update(dets_tensor, img_info, img_size)

    # --- version with MOT style arguments --------------------------------
    if {"img_info", "img_size"} & set(params):
        cls_arr = np.array([_cls_id(c) for c in classes], dtype=np.float32)[
            :, None
        ]
        dets = np.concatenate(
            [
                np.asarray(tlwhs, dtype=np.float32),
                np.asarray(scores, dtype=np.float32)[:, None],
                cls_arr,
            ],
            axis=1,
        )

        im_w = max(b[0] + b[2] for b in tlwhs) if tlwhs else 1920
        im_h = max(b[1] + b[3] for b in tlwhs) if tlwhs else 1080
        img_info = (im_h, im_w, 1.0)
        img_size = (im_w, im_h)

        tensor_dets = None
        if torch:
            try:
                tensor_dets = torch.as_tensor(dets, dtype=torch.float32)
            except Exception:  # pragma: no cover - optional torch
                tensor_dets = None

        candidates = []
        if tensor_dets is not None and len(params) == 3:
            candidates.append((tensor_dets, img_info, img_size))
        candidates.extend(
            [
                (dets, img_info, img_size),
                (img_info, img_size),
                (dets, img_info),
                (dets, img_size),
            ]
        )

        for args in candidates:
            try:
                return tracker.update(*args)
            except (TypeError, AttributeError):
                continue

    raise RuntimeError(f"Unknown BYTETracker.update signature: {params}")


def _first_det_for_track(tid: int, frame_id: int) -> dict:
    """Return first detection dict for ``tid``."""

    for (fid, _), det in _det_index.items():
        if det.get("track_id") == tid:
            return det
    return {"class": None}


def _bbox_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Return IoU between two bounding boxes."""
    if box is None:
        return 0.0
    b1_box = box(*b1)
    b2_box = box(*b2)
    inter = b1_box.intersection(b2_box).area
    union = b1_box.union(b2_box).area
    return inter / union if union > 0 else 0.0


def _load_model(model_name: str):
    """Load a YOLOX model from the local ByteTrack implementation."""

    if model_name not in YOLOX_MODELS:
        raise ValueError(f"Unsupported model {model_name}")

    try:
        from yolox.exp import get_exp
        from yolox.utils import fuse_model
    except Exception as exc:  # pragma: no cover - missing submodule
        raise ImportError(
            "YOLOX modules not found. Did you clone the ByteTrack submodule and "
            "run 'bash build_externals.sh'?"
        ) from exc

    variant = model_name.split("-", 1)[-1]
    LOGGER.info("Loading %s", model_name)

    exp = get_exp(f"yolox_{variant}.py", model_name)
    model = exp.get_model()
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    ckpt_path = weights_dir / f"yolox_{variant}.pth"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Weights not found: {ckpt_path}. Place pretrained YOLOX weights in the 'weights' directory."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval().to(device)


def _letterbox_image(
    img: Image.Image, size: int
) -> tuple[Image.Image, float, int, int]:
    """Resize ``img`` with unchanged aspect ratio using padding."""

    w0, h0 = img.size
    ratio = min(size / w0, size / h0)
    new_w, new_h = int(w0 * ratio), int(h0 * ratio)
    resized = img.resize((new_w, new_h))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, ratio, pad_x, pad_y


def _preprocess_image(
    path: Path, size: int
) -> tuple[torch.Tensor, float, float, float, int, int]:
    """Preprocess image using YOLOX's :class:`ValTransform`.

    This function mirrors the official YOLOX ``demo.py`` preprocessing. It
    returns a tensor ready for inference along with metadata required for
    projecting detections back to the original image coordinates.
    """

    import cv2  # defer heavy import
    from yolox.data.data_augment import ValTransform

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")

    h0, w0 = img.shape[:2]

    preproc = ValTransform(legacy=False)
    img_processed, _ = preproc(img, None, (size, size))

    _, new_h, new_w = img_processed.shape
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2

    tensor = torch.from_numpy(img_processed).unsqueeze(0).float().cuda()

    ratio = new_w / w0

    return tensor, ratio, pad_x, pad_y, w0, h0


def _filter_detections(
    outputs: Sequence[Sequence[float]],
    conf_thr: float,
    keep_ids: Sequence[int],
) -> List[Tuple[List[float], float, int]]:
    """Filter YOLOX detections for selected classes."""
    allowed = set(keep_ids)
    results: List[Tuple[List[float], float, int]] = []
    for det in outputs:
        if len(det) == 6:
            x1, y1, x2, y2, score, cls_id = det
        else:
            x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det
            score = obj_conf * cls_conf
        if int(cls_id) not in allowed or float(score) < conf_thr:
            continue
        results.append(
            (
                [float(x1), float(y1), float(x2), float(y2)],
                float(score),
                int(cls_id),
            )
        )
    return results


def detect_folder(
    frames_dir: Path,
    out_json: Path,
    model_name: str,
    img_size: int,
    conf_thres: float = 0.3,
    nms_thres: float = 0.45,
    class_ids: Sequence[int] | None = None,
) -> None:
    """Run detection over ``frames_dir`` and write results.

    If ``class_ids`` is not provided, detections for all YOLOX classes are
    returned. Otherwise only detections for the specified classes are kept.

    Args:
        frames_dir: Directory containing frame images.
        out_json: File to write detection results.
        model_name: Variant name of the YOLOX model to load.
        img_size: Target input size for the model.
    """
    if class_ids is None:
        class_ids = list(range(YOLOX_NUM_CLASSES))
    model = _load_model(model_name)
    frames = sorted(
        [
            p
            for p in frames_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".png"}
        ]
    )
    if not frames:
        LOGGER.warning("No frames found in %s", frames_dir)
        return

    out: List[dict] = []
    start = time.perf_counter()
    with tqdm(total=len(frames), desc="Detecting") as pbar:
        for frame in frames:
            tensor, ratio, pad_x, pad_y, w0, h0 = _preprocess_image(
                frame, img_size
            )
            with torch.no_grad():
                raw = model(tensor)[0]

            # Align output tensor with expected (B, N, 85) shape.
            if isinstance(raw, list):
                raw = model.head.decode_outputs(raw, dtype=raw[0].dtype)
            elif raw.dim() == 2:
                raw = raw.unsqueeze(0)

            outputs = raw
            from yolox.utils import postprocess

            processed = postprocess(
                outputs,
                num_classes=80,
                conf_thre=conf_thres,
                nms_thre=nms_thres,
                class_agnostic=False,
            )

            det = (
                processed[0]
                if processed and processed[0] is not None
                else None
            )
            preds = det.cpu() if det is not None else torch.empty((0, 6))
            preds_list = preds.tolist()

            detections = []
            for bbox, score, cls_id in _filter_detections(
                preds_list, conf_thres, class_ids
            ):
                assert (
                    0 <= cls_id < YOLOX_NUM_CLASSES
                ), f"Unexpected class id {cls_id}"
                x1_p, y1_p, x2_p, y2_p = bbox
                x0 = max((x1_p - pad_x) / ratio, 0.0)
                y0 = max((y1_p - pad_y) / ratio, 0.0)
                x1 = min((x2_p - pad_x) / ratio, w0)
                y1 = min((y2_p - pad_y) / ratio, h0)

                ix0, iy0, ix1, iy1 = (
                    int(round(x0)),
                    int(round(y0)),
                    int(round(x1)),
                    int(round(y1)),
                )

                if ix1 > ix0 and iy1 > iy0:
                    detections.append(
                        {
                            "bbox": [ix0, iy0, ix1, iy1],
                            "score": float(score),
                            "class": int(cls_id),
                        }
                    )
                else:
                    LOGGER.debug(
                        "Discarded invalid box %s from %s",
                        [x0, y0, x1, y1],
                        frame.name,
                    )

            out.append({"frame": frame.name, "detections": detections})
            pbar.update(1)
    elapsed = time.perf_counter() - start

    free, total = torch.cuda.mem_get_info()
    LOGGER.info(
        "GPU memory: %.2f/%.2f GB used",
        (total - free) / 1024**3,
        total / 1024**3,
    )
    LOGGER.info("Processed %d frames in %.2fs", len(frames), elapsed)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)


def track_detections(
    detections_json: Path, output_json: Path, min_score: float = 0.3
) -> None:
    """Run ByteTrack on detection results and write tracks.

    The ``detections_json`` file is expected to contain a flat list where each
    element has ``frame``, ``class``, ``bbox`` and ``score`` fields. The output
    ``tracks.json`` will store one entry per frame with the assigned ``track_id``.
    """

    if BYTETracker is None:
        raise ImportError(
            "BYTETracker import failed. Make sure you have run:\n"
            "  pip install -r requirements.txt   # adds scipy\n"
            "  bash build_externals.sh           # compiles yolox C-extension\n"
        )

    with detections_json.open() as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError("detections-json must contain a list")

    _det_index.clear()
    frames: Dict[int, list[dict]] = {}
    for det in sorted(raw, key=lambda x: x["frame"]):
        frame_match = re.search(r"\d+", str(det["frame"]))
        if not frame_match:
            LOGGER.debug(
                "Skipping detection with invalid frame: %s", det["frame"]
            )
            continue
        frame_id = int(frame_match.group())
        cls_val = det.get("class")
        if isinstance(cls_val, str):
            cls_key = CLASS_ALIASES.get(cls_val, cls_val)
            if cls_key not in CLASS_NAME_TO_ID:
                continue
            cls_id = CLASS_NAME_TO_ID[cls_key]
        else:
            cls_id = int(cls_val)
            if cls_id not in CLASS_ID_TO_NAME:
                continue
        score = float(det.get("score", 0.0))
        if score < min_score:
            continue
        bbox = [float(v) for v in det["bbox"]]
        idx = len(frames.setdefault(frame_id, []))
        entry = {"bbox": bbox, "score": score, "class": cls_id}
        frames[frame_id].append(entry)
        _det_index[(frame_id, idx)] = entry

    # Create tracker instance with recommended parameters.
    tracker = BYTETracker(
        track_thresh=min_score,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
    )
    output: list[dict] = []
    track_ids = set()
    for frame_id in sorted(frames):
        dets = frames[frame_id]
        LOGGER.debug("Frame %s: %s detections", frame_id, len(dets))
        tlwhs = [
            [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            for b in (d["bbox"] for d in dets)
        ]
        scores = [d["score"] for d in dets]
        classes = [d["class"] for d in dets]
        online = _update_tracker(tracker, tlwhs, scores, classes, frame_id)
        LOGGER.debug("Frame %s: %s tracks", frame_id, len(online))

        frame_det_map = {
            idx: det
            for (fid, idx), det in _det_index.items()
            if fid == frame_id and "bbox" in det
        }
        used_dets: set[int] = set()

        for obj in online:
            # Access ByteTrack's internal bounding box representation directly for efficiency
            x, y, w, h = obj._tlwh  # pylint: disable=protected-access
            tlbr = [x, y, x + w, y + h]
            best_iou = 0.0
            best_idx: int | None = None
            for idx, det in frame_det_map.items():
                if idx in used_dets:
                    continue
                iou = _bbox_iou(tlbr, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is not None and best_iou > 0.3:
                _det_index[(frame_id, best_idx)]["track_id"] = obj.track_id
                used_dets.add(best_idx)
                LOGGER.debug(
                    "Assigned track_id %s to det #%s (IoU=%.2f)",
                    obj.track_id,
                    best_idx,
                    best_iou,
                )
            else:
                LOGGER.warning(
                    "No matching detection found for track %s at frame %s",
                    obj.track_id,
                    frame_id,
                )

        for obj in online:
            x, y, w, h = obj.tlwh

            cls_idx = getattr(obj, "cls", getattr(obj, "class_id", None))
            if cls_idx is None:
                det = _first_det_for_track(obj.track_id, frame_id)
                cls_idx = det.get("class")

            cls_name = CLASS_ID_TO_NAME.get(int(cls_idx), "unknown")

            bbox = [int(x), int(y), int(x + w), int(y + h)]
            output.append(
                {
                    "frame": frame_id,
                    "class": int(cls_idx),
                    "track_id": obj.track_id,
                    "bbox": bbox,
                    "score": float(obj.score),
                }
            )
            track_ids.add(obj.track_id)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fh:
        json.dump(output, fh, indent=2)

    logger.info("Processed {} frames", len(frames))
    logger.info("Active tracks: {}", len(track_ids))
    summary = {
        CLASS_ID_TO_NAME[cid]: sum(1 for t in output if t["class"] == cid)
        for cid in (CLASS_NAME_TO_ID["person"], CLASS_NAME_TO_ID["ball"])
    }
    logger.info("Track summary: {}", summary)
    logger.info("Saved {} tracked detections to {}", len(output), output_json)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    if args.command == "track":
        try:
            track_detections(
                args.detections_json, args.output_json, args.min_score
            )
        except Exception as exc:  # pragma: no cover - top level
            logger.exception("Tracking failed")
            raise SystemExit(1) from exc
    else:
        try:
            detect_folder(
                args.frames_dir,
                args.output_json,
                args.model,
                args.img_size,
                args.conf_thres,
                args.nms_thres,
                args.classes,
            )
        except Exception as exc:  # pragma: no cover - top level
            LOGGER.exception("Detection failed")
            raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
