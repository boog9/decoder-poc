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
"""Single image detection demo using YOLOX."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Dict, Sequence

import torch
from PIL import Image, ImageDraw

from . import detect_objects as dobj
from .utils.classes import CLASS_NAME_TO_ID

LOGGER = logging.getLogger(__name__)


def _load_model_device(model_name: str, device: torch.device):
    """Load a YOLOX model on the specified device."""
    if model_name not in dobj.YOLOX_MODELS:
        raise ValueError(f"Unsupported model {model_name}")
    torch_name = dobj._YOLOX_MODEL_MAP[model_name]
    LOGGER.info("Loading %s on %s", model_name, device)
    model = torch.hub.load("Megvii-BaseDetection/YOLOX", torch_name, pretrained=True)
    return model.eval().to(device)


def _annotate(image_path: Path, detections: List[Dict], output: Path) -> None:
    """Draw detections on the image and save to ``output``."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    img.save(output)


def detect_image(
    image_path: Path,
    model_name: str,
    img_size: int,
    conf_thres: float,
    nms_thres: float,
    device: torch.device,
    class_ids: Sequence[int] | None = None,
) -> List[Dict]:
    """Run YOLOX detection on ``image_path``.

    Args:
        image_path: Path to the input image.
        model_name: YOLOX model variant name.
        img_size: Square size for model input.
        conf_thres: Confidence threshold.
        nms_thres: NMS IoU threshold.
        device: Device to run inference on.
        class_ids: Classes to keep. Defaults to only the ``person`` class.

    Returns:
        List of detection dictionaries.
    """
    model = _load_model_device(model_name, device)
    tensor, ratio, pad_x, pad_y, w0, h0 = dobj._preprocess_image(image_path, img_size)
    with torch.no_grad():
        raw = model(tensor)[0]
    if isinstance(raw, list):
        raw = model.head.decode_outputs(raw, dtype=raw[0].dtype)
    elif raw.dim() == 2:
        raw = raw.unsqueeze(0)
    from yolox.utils import postprocess

    processed = postprocess(
        raw,
        num_classes=80,
        conf_thre=conf_thres,
        nms_thre=nms_thres,
        class_agnostic=False,
    )
    det = processed[0] if processed and processed[0] is not None else None
    outputs_list = det.cpu().tolist() if det is not None else []

    detections: List[Dict] = []
    if class_ids is None:
        class_ids = [CLASS_NAME_TO_ID["person"]]
    for bbox, score, cls in dobj._filter_detections(outputs_list, conf_thres, class_ids):
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
                {"bbox": [ix0, iy0, ix1, iy1], "score": score, "class": cls}
            )
    return detections


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="yolox-x",
        choices=sorted(dobj.YOLOX_MODELS),
        help="YOLOX model variant",
    )
    parser.add_argument("--img-size", type=int, default=640, help="Resize square size")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--nms-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cpu or gpu)",
    )
    parser.add_argument("--save-result", type=Path, help="Where to save annotated image")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    device = torch.device("cuda" if args.device.lower() in {"gpu", "cuda"} else "cpu")
    try:
        detections = detect_image(
            args.image,
            args.model,
            args.img_size,
            args.conf_thres,
            args.nms_thres,
            device,
        )
        if args.save_result:
            _annotate(args.image, detections, args.save_result)
        print(json.dumps({"image": args.image.name, "detections": detections}, indent=2))
    except Exception as exc:  # pragma: no cover - top level error
        LOGGER.exception("Detection failed")
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
