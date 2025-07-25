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
"""Draw ROI detections on frame images."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
from typing import Any

LOGGER = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing input frame images.",
    )
    parser.add_argument(
        "--detections-json",
        type=Path,
        required=True,
        help="JSON file with detection data from detect_objects",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where annotated PNG frames will be written",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size used during detection",
    )
    parser.add_argument(
        "--label",
        action="store_true",
        help="Draw boxes with color per class label",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="red",
        help="Rectangle outline color (default: red)",
    )
    return parser.parse_args(argv)


def _load_detections(path: Path) -> List[dict]:
    """Load detection list from ``path``."""
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Invalid detections format")
    return data


def _sanitize_bbox(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Ensure ``bbox`` coordinates are top-left to bottom-right.

    Args:
        bbox: Bounding box as ``[x1, y1, x2, y2]``.

    Returns:
        Sanitized bounding box tuple ``(x1, y1, x2, y2)`` with coordinates
        sorted so ``x2 >= x1`` and ``y2 >= y1``.
    """

    x1, y1, x2, y2 = bbox
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _preprocess_params(img: Any, size: int) -> Tuple[float, float, float, int, int]:
    """Compute resize ratio and padding for ``img``.

    This replicates the behaviour of YOLOX :class:`ValTransform` used in
    :mod:`src.detect_objects`. The function does not require the YOLOX package
    and therefore allows drawing ROIs in environments where YOLOX is not
    installed.

    Args:
        img: Image array in ``BGR`` format.
        size: Target square size for inference.

    Returns:
        A tuple ``(ratio, pad_x, pad_y, width, height)`` with the resize ratio,
        horizontal padding, vertical padding and the original dimensions.
    """

    h0, w0 = img.shape[:2]

    scale = min(size / w0, size / h0)
    new_w = int(w0 * scale)
    new_h = int(h0 * scale)
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2

    ratio = new_w / w0

    return ratio, pad_x, pad_y, w0, h0


def _backproject_bbox(
    bbox: Tuple[float, float, float, float],
    ratio: float,
    pad_x: float,
    pad_y: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Project ``bbox`` from model coordinates to the original image.

    Args:
        bbox: Bounding box ``(x1, y1, x2, y2)`` from the model output.
        ratio: Resize ratio used during preprocessing.
        pad_x: Horizontal padding used during preprocessing.
        pad_y: Vertical padding used during preprocessing.
        width: Original image width.
        height: Original image height.

    Returns:
        Integer bounding box coordinates in the original image space.
    """

    x1, y1, x2, y2 = bbox
    x1 = max(int(round((x1 - pad_x) / ratio)), 0)
    y1 = max(int(round((y1 - pad_y) / ratio)), 0)
    x2 = min(int(round((x2 - pad_x) / ratio)), width)
    y2 = min(int(round((y2 - pad_y) / ratio)), height)

    return x1, y1, x2, y2


def _color_bgr(name: str) -> Tuple[int, int, int]:
    """Return a BGR tuple for a given color name."""

    colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }
    return colors.get(name.lower(), (0, 0, 255))


COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


CLASS_COLORS = {
    0: (255, 56, 56),
    32: (0, 255, 0),
}


def _label_color(class_id: int) -> Tuple[int, int, int]:
    """Return color for ``class_id``."""

    return CLASS_COLORS.get(class_id, (0, 0, 255))


def draw_rois(
    frames_dir: Path,
    detections_json: Path,
    output_dir: Path,
    img_size: int,
    color: str = "red",
    label: bool = False,
) -> None:
    """Overlay detection ROIs on frames and save to ``output_dir`` as PNG.

    Args:
        frames_dir: Directory of frame images.
        detections_json: JSON file with detection results.
        output_dir: Destination for annotated PNG images.
        img_size: Unused. Present for backwards compatibility.
        color: Outline color for rectangles.
        label: If ``True``, color boxes by class and draw labels with score.
    """
    detections = _load_detections(detections_json)
    output_dir.mkdir(parents=True, exist_ok=True)

    bgr = _color_bgr(color)

    for entry in detections:
        frame_name = entry.get("frame")
        detections_raw = entry.get("detections", [])
        if not frame_name:
            LOGGER.debug("Skipping detection entry without frame")
            continue
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            LOGGER.warning("Frame not found: %s", frame_path)
            continue

        img = cv2.imread(str(frame_path))
        if img is None:
            LOGGER.warning("Failed to read %s", frame_path)
            continue

        for det in detections_raw:
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                LOGGER.debug("Invalid bbox %s in %s", bbox, frame_name)
                continue
            x1, y1, x2, y2 = map(int, _sanitize_bbox(bbox))

            if x2 > x1 and y2 > y1:
                class_id = det.get("class", -1)
                score = det.get("score")
                if label:
                    clr = _label_color(class_id)
                    if 0 <= class_id < len(COCO_CLASS_NAMES):
                        class_name = COCO_CLASS_NAMES[class_id]
                    else:
                        class_name = f"id{class_id}"
                    if score is None:
                        text = class_name
                    else:
                        text = f"{class_name}:{score * 100:.1f}%"
                    (tw, th), bl = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    top = max(y1 - th - bl - 2, 0)
                    cv2.rectangle(
                        img, (x1, top), (x1 + tw, top + th + bl), clr, -1
                    )
                    cv2.putText(
                        img,
                        text,
                        (x1, top + th),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                else:
                    clr = bgr
                cv2.rectangle(img, (x1, y1), (x2, y2), clr, 2)
            else:
                LOGGER.debug(
                    "Discarded invalid box %s from %s", [x1, y1, x2, y2], frame_name
                )

        out_name = Path(frame_name).with_suffix(".png").name
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), img)
        LOGGER.debug("Wrote %s", out_path)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    try:
        draw_rois(
            args.frames_dir,
            args.detections_json,
            args.output_dir,
            args.img_size,
            args.color,
            args.label,
        )
    except Exception as exc:  # pragma: no cover - top level
        LOGGER.error("Failed to draw ROIs: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
