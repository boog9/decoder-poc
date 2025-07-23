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
    return parser.parse_args(argv)


def _load_model(device: str):
    """Load Swin2SR model on given device."""
    import timm
    import torch

    model = timm.create_model("swin2sr-lightweight-x4-64", pretrained=True)

    model = model.eval().to(device)
    return model


def _load_image(path: Path):
    """Load an image tensor."""
    import torch
    import numpy as np

    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def _save_image(tensor, path: Path) -> None:
    """Save a tensor image to ``path``."""
    import numpy as np

    array = (
        tensor.clamp(0, 1).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    )
    img = Image.fromarray(array)
    img.save(path)


def enhance_frames(input_dir: Path, output_dir: Path, batch_size: int = 4) -> None:
    """Enhance frames in ``input_dir`` and save to ``output_dir``."""
    import torch

    images = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg"}]
    )
    if not images:
        LOGGER.warning("No images found in %s", input_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(device)

    total_start = time.perf_counter()
    processed = 0
    with tqdm(total=len(images), unit="img", desc="Enhancing") as pbar:
        for i in range(0, len(images), batch_size):
            batch_paths = images[i : i + batch_size]
            batch = [_load_image(p) for p in batch_paths]
            batch_tensor = torch.stack(batch).to(device)
            start = time.perf_counter()
            with torch.no_grad():
                out = model(batch_tensor)
            elapsed = time.perf_counter() - start
            per_image = elapsed / len(batch_paths)
            for tensor, src in zip(out, batch_paths):
                _save_image(tensor, output_dir / src.name)
            processed += len(batch_paths)
            pbar.update(len(batch_paths))
            pbar.set_postfix({"ms/img": f"{per_image*1000:.2f}"})
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
        enhance_frames(args.input_dir, args.output_dir, args.batch_size)
    except Exception as exc:  # pragma: no cover - top level
        LOGGER.error("Failed to enhance frames: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
