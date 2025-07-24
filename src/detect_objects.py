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


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir", type=Path, required=True, help="Directory of input frames"
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
    return parser.parse_args(argv)


def _load_model(model_name: str, device: str = "cuda"):
    """Load a YOLOX model via ``torch.hub``."""
    if model_name not in YOLOX_MODELS:
        raise ValueError(f"Unsupported model {model_name}")
    LOGGER.info("Loading %s model on %s", model_name, device)
    model = torch.hub.load(
        "Megvii-BaseDetection/YOLOX", model_name, pretrained=True
    )
    model = model.eval().to(device)
    return model


def _preprocess_image(path: Path) -> torch.Tensor:
    """Load image from ``path`` and convert to tensor."""
    img = Image.open(path).convert("RGB")
    tensor = to_tensor(img)
    return tensor


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


def detect_folder(frames_dir: Path, out_json: Path, model_name: str) -> None:
    """Run detection over ``frames_dir`` and write results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(model_name, device)
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
            tensor = _preprocess_image(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)[0]
            detections = [
                {"bbox": b, "score": s, "class": c}
                for b, s, c in _filter_person_detections(outputs)
            ]
            out.append({"frame": frame.name, "detections": detections})
            pbar.update(1)
    elapsed = time.perf_counter() - start

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        LOGGER.info(
            "GPU memory: %.2f/%.2f GB used",
            (total - free) / 1024 ** 3,
            total / 1024 ** 3,
        )
    LOGGER.info("Processed %d frames in %.2fs", len(frames), elapsed)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        detect_folder(args.frames_dir, args.output_json, args.model)
    except Exception as exc:  # pragma: no cover - top level
        LOGGER.error("Detection failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
