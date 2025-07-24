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
"""Object detection on a directory of frames using YOLOX."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms.functional import to_tensor

LOGGER = logging.getLogger(__name__)

YOLOX_MODELS = {"yolox-s", "yolox-m", "yolox-l", "yolox-x"}

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
        help=(
            "Resize frames to this square size before detection. "
            "Should be a multiple of 32 (default: 640)."
        ),
    )
    return parser.parse_args(argv)


def _load_model(model_name: str, device: str = "cuda"):
    """Load a YOLOX model via ``torch.hub``."""
    if model_name not in YOLOX_MODELS:
        raise ValueError(f"Unsupported model {model_name}")
    torch_name = _YOLOX_MODEL_MAP[model_name]
    LOGGER.info("Loading %s model on %s", model_name, device)
    model = torch.hub.load(
        "Megvii-BaseDetection/YOLOX", torch_name, pretrained=True
    )
    model = model.eval().to(device)
    return model


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
) -> tuple[torch.Tensor, float, int, int, int, int]:
    """Load image and letterbox to ``size`` square tensor.

    Returns the tensor and resize metadata for back-projection.
    """

    if size % 32 != 0:
        raise ValueError("img_size must be a multiple of 32")
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    img, ratio, pad_x, pad_y = _letterbox_image(img, size)
    tensor = to_tensor(img)
    return tensor, ratio, pad_x, pad_y, w0, h0


def _filter_person_detections(
    outputs: torch.Tensor,
) -> List[Tuple[List[float], float, int]]:
    """Filter YOLOX detections to only ``person`` class."""
    results = []
    # YOLOX outputs: [x1, y1, x2, y2, score, class]
    for det in outputs:
        cls = int(det[5])
        if cls != 0:
            continue
        bbox = det[:4].tolist()
        score = float(det[4])
        results.append((bbox, score, cls))
    return results


def _deletterbox(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    ratio: float,
    pad_x: int,
    pad_y: int,
    w0: int,
    h0: int,
) -> tuple[int, int, int, int]:
    """Map letterboxed coordinates back to the original image."""

    x0 = max((x0 - pad_x) / ratio, 0.0)
    x1 = min((x1 - pad_x) / ratio, float(w0))
    y0 = max((y0 - pad_y) / ratio, 0.0)
    y1 = min((y1 - pad_y) / ratio, float(h0))
    return int(x0), int(y0), int(x1), int(y1)


def detect_folder(
    frames_dir: Path, out_json: Path, model_name: str, img_size: int
) -> None:
    """Run detection over ``frames_dir`` and write results.

    Args:
        frames_dir: Directory containing frame images.
        out_json: File to write detection results.
        model_name: Variant name of the YOLOX model to load.
        img_size: Target input size for the model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(model_name, device)
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
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)[0]

            detections = []
            for bbox, score, cls in _filter_person_detections(outputs):
                x0, y0, x1, y1 = _deletterbox(
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    ratio,
                    pad_x,
                    pad_y,
                    w0,
                    h0,
                )
                detections.append(
                    {"bbox": [x0, y0, x1, y1], "score": score, "class": cls}
                )

            out.append({"frame": frame.name, "detections": detections})
            pbar.update(1)
    elapsed = time.perf_counter() - start

    if device == "cuda":
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
        detect_folder(
            args.frames_dir, args.output_json, args.model, args.img_size
        )
    except Exception as exc:  # pragma: no cover - top level
        LOGGER.error("Detection failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
