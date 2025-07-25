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
from typing import Iterable, List, Tuple, Sequence

from PIL import Image
from tqdm import tqdm

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA device required for YOLOX")

LOGGER = logging.getLogger(__name__)

YOLOX_MODELS = {"yolox-s", "yolox-m", "yolox-l", "yolox-x"}

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
    """Parse CLI arguments for :mod:`detect_objects`.

    Args:
        argv: Optional list of command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory of input frames",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write detections JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolox-s",
        choices=sorted(YOLOX_MODELS),
        help="YOLOX model variant",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Resize frames to this square size before detection.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.3,
        help="Confidence threshold for detections (default: 0.3)",
    )
    parser.add_argument(
        "--nms-thres",
        type=float,
        default=0.45,
        help="IoU threshold for non-max suppression (default: 0.45)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=list(CLASS_MAP.values()),
        help="Class IDs to detect",
    )
    return parser.parse_args(argv)


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
    return [
        (
            [float(det[0]), float(det[1]), float(det[2]), float(det[3])],
            float(det[4]),
            int(det[5]),
        )
        for det in outputs
        if int(det[5]) in allowed and float(det[4]) >= conf_thr
    ]




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

    Only detections with ``class_ids`` are kept.

    Args:
        frames_dir: Directory containing frame images.
        out_json: File to write detection results.
        model_name: Variant name of the YOLOX model to load.
        img_size: Target input size for the model.
    """
    model = _load_model(model_name)
    if class_ids is None:
        class_ids = [CLASS_MAP["person"]]
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
            outputs_list = det.cpu().tolist() if det is not None else []

            detections = []
            for bbox, score, cls in _filter_detections(outputs_list, conf_thres, class_ids):
                x0 = max((bbox[0] - pad_x) / ratio, 0.0)
                y0 = max((bbox[1] - pad_y) / ratio, 0.0)
                x1 = min((bbox[2] - pad_x) / ratio, w0)
                y1 = min((bbox[3] - pad_y) / ratio, h0)

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
                            "score": score,
                            "class": cls,
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


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    try:
        invalid = [c for c in args.classes if c not in CLASS_MAP.values()]
        if invalid:
            valid = ", ".join(str(v) for v in sorted(CLASS_MAP.values()))
            raise SystemExit(f"Unknown class id {invalid[0]}. Available: {valid}")
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
