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
import json
import logging
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Sequence, Dict

from loguru import logger
import os
import sys

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

# Mapping of human-readable class names to COCO class IDs. Adjust if using a
# different dataset.
CLASS_MAP = {
    "person": 0,
    "sports ball": 32,
}

# Map CLI model names to torch.hub callable names.
_YOLOX_MODEL_MAP = {
    "yolox-s": "yolox_s",
    "yolox-m": "yolox_m",
    "yolox-l": "yolox_l",
    "yolox-x": "yolox_x",
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for detection and tracking."""

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=False)

    # Detection subcommand (default)
    det = sub.add_parser("detect", help="Run YOLOX detection")
    det.add_argument("--frames-dir", type=Path, required=True)
    det.add_argument("--output-json", type=Path, required=True)
    det.add_argument("--model", type=str, default="yolox-s", choices=sorted(YOLOX_MODELS))
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


def _load_model(model_name: str):
    """Load a YOLOX model via ``torch.hub`` on CUDA."""
    if model_name not in YOLOX_MODELS:
        raise ValueError(f"Unsupported model {model_name}")
    torch_name = _YOLOX_MODEL_MAP[model_name]
    LOGGER.info("Loading %s on CUDA", model_name)
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", torch_name, pretrained=True)
    return model.eval().cuda()


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
            ([float(x1), float(y1), float(x2), float(y2)], float(score), int(cls_id))
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
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"}]
    )
    if not frames:
        LOGGER.warning("No frames found in %s", frames_dir)
        return

    out: List[dict] = []
    start = time.perf_counter()
    with tqdm(total=len(frames), desc="Detecting") as pbar:
        for frame in frames:
            tensor, ratio, pad_x, pad_y, w0, h0 = _preprocess_image(frame, img_size)
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

            det = processed[0] if processed and processed[0] is not None else None
            preds = det.cpu() if det is not None else torch.empty((0, 6))
            preds_list = preds.tolist()

            detections = []
            for bbox, score, cls_id in _filter_detections(
                preds_list, conf_thres, class_ids
            ):
                assert 0 <= cls_id < YOLOX_NUM_CLASSES, (
                    f"Unexpected class id {cls_id}"
                )
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
                        "Discarded invalid box %s from %s", [x0, y0, x1, y1], frame.name
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
    """Run ByteTrack on detection results and write tracks."""

    if BYTETracker is None:
        raise ImportError(
            "BYTETracker import failed. Make sure you have run:\n"
            "  pip install -r requirements.txt   # adds scipy\n"
            "  bash build_externals.sh           # compiles yolox C-extension\n"
        )

    with detections_json.open() as fh:
        raw = json.load(fh)

    frames: Dict[int, list[dict]] = {}
    for idx, frame_obj in enumerate(sorted(raw, key=lambda x: x["frame"]), start=1):
        for det in frame_obj.get("detections", []):
            cls_id = det.get("class")
            if cls_id == CLASS_MAP["person"]:
                cls = "person"
            elif cls_id == CLASS_MAP["sports ball"]:
                cls = "ball"
            else:
                continue
            score = float(det.get("score", 0.0))
            if score < min_score:
                continue
            frames.setdefault(idx, []).append(
                {"bbox": det["bbox"], "score": score, "class": cls}
            )

    tracker = BYTETracker(track_thresh=min_score, frame_rate=30)
    output: list[dict] = []
    track_ids = set()
    for frame_id in sorted(frames):
        dets = frames[frame_id]
        tlwhs = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in (d["bbox"] for d in dets)]
        scores = [d["score"] for d in dets]
        classes = [d["class"] for d in dets]
        online = tracker.update(tlwhs, scores, classes, frame_id)
        for obj, cls in zip(online, classes):
            x, y, w, h = obj.tlwh
            bbox = [int(x), int(y), int(x + w), int(y + h)]
            output.append(
                {
                    "frame": frame_id,
                    "class": cls,
                    "track_id": obj.track_id,
                    "bbox": bbox,
                    "score": float(obj.score),
                }
            )
            track_ids.add(obj.track_id)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fh:
        json.dump(output, fh, indent=2)

    logger.info("Processed %d frames", len(frames))
    logger.info("Active tracks: %d", len(track_ids))
    summary = {}
    for cls in ("person", "ball"):
        summary[cls] = sum(1 for t in output if t["class"] == cls)
    logger.info("Track summary: %s", summary)
    logger.info("Saved %d tracked detections to %s", len(output), output_json)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    if args.command == "track":
        try:
            track_detections(args.detections_json, args.output_json, args.min_score)
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
