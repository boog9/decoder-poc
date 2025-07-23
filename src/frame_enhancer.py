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
"""Frame enhancement CLI using Swin2SR."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency missing
    raise ImportError(
        "Pillow is required. Install with 'pip install pillow'"
    ) from exc
    
from tqdm import tqdm

try:
    from transformers import AutoImageProcessor
    try:  # Newer Transformers versions
        from transformers import AutoModelForImageSuperResolution
    except ImportError:  # pragma: no cover - older Transformers
        from transformers import (
            Swin2SRForImageSuperResolution as AutoModelForImageSuperResolution,
        )
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "transformers with Swin2SR support is required.\n"
        "Install or upgrade with 'pip install -U transformers'"
    ) from exc

DEFAULT_MODEL_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of input frames (png/jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where enhanced frames will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of frames to process per batch (default: 4)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "Hugging Face repo ID or local path to the Swin2SR model "
            f"(default: {DEFAULT_MODEL_ID})"
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision for inference (requires CUDA).",
    )
    return parser.parse_args(argv)


def _load_model(device: str, model_id: str, fp16: bool = False):
    """Load Swin2SR model and processor on given device.

    Args:
        device: ``cuda`` or ``cpu``.
        model_id: Hugging Face repo ID or path to a local model directory.
        fp16: Whether to convert the model to ``float16``.

    Returns:
        Tuple of model and processor.
    """
    import torch

    model = AutoModelForImageSuperResolution.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = model.eval().to(device)
    if fp16:
        model = model.half()
    return model, processor


def _load_image(path: Path):
    """Load an RGB image."""
    img = Image.open(path).convert("RGB")
    return img


def _save_image(tensor, path: Path) -> None:
    """Save a tensor image to ``path``."""
    import numpy as np

    array = (
        tensor.clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    )
    img = Image.fromarray(array)
    img.save(path)


def enhance_frames(
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 4,
    model_id: str = DEFAULT_MODEL_ID,
    fp16: bool = False,
) -> None:
    """Enhance frames in ``input_dir`` and save to ``output_dir``.

    Args:
        input_dir: Directory with input ``.png`` or ``.jpg`` frames.
        output_dir: Directory where enhanced frames will be written.
        batch_size: Number of frames to process per batch.
        model_id: Hugging Face model identifier.
        fp16: Whether to run the model using ``float16`` precision.
    """
    import torch

    images = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg"}]
    )
    if not images:
        LOGGER.warning("No images found in %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _load_model(device, model_id, fp16)
    total_start = time.perf_counter()
    processed = 0
    with tqdm(total=len(images), unit="img", desc="Enhancing") as pbar:
        for i in range(0, len(images), batch_size):
            batch_paths = images[i : i + batch_size]
            batch = [_load_image(p) for p in batch_paths]
            inputs = processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if fp16:
                inputs = {k: v.half() if hasattr(v, "half") else v for k, v in inputs.items()}
            start = time.perf_counter()
            try:
                with torch.no_grad():
                    out = model(**inputs).reconstruction
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    LOGGER.error(
                        "CUDA out of memory. Try reducing --batch-size or set"
                        " PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
                    )
                raise
            elapsed = time.perf_counter() - start
            per_image = elapsed / len(batch_paths)
            for tensor, src in zip(out, batch_paths):
                _save_image(tensor, output_dir / src.name)
            processed += len(batch_paths)
            pbar.update(len(batch_paths))
            pbar.set_postfix({"ms/img": f"{per_image*1000:.2f}"})
            if device == "cuda":
                torch.cuda.empty_cache()
    total_elapsed = time.perf_counter() - total_start
    fps = processed / total_elapsed if total_elapsed > 0 else 0
    print(
        f"Frames: {processed} | Time: {total_elapsed:.2f}s | FPS: {fps:.2f}",
        flush=True,
    )


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        enhance_frames(
            args.input_dir,
            args.output_dir,
            args.batch_size,
            args.model_id,
            args.fp16,
        )

    except Exception as exc:  # pragma: no cover - top level
        LOGGER.error("Failed to enhance frames: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
