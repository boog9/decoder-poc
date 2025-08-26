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
"""CLI for running the Tennis Court Detector."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .tcd_model import BallTrackerNet
from .utils_weights import load_tcd_state_dict
from .postproc import heat_to_peak_xy, refine_kps_if_needed


def preprocess_bgr(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Resize BGR image to 640×360 and convert to ``[0, 1]`` RGB tensor."""

    rgb = cv2.cvtColor(cv2.resize(img, (640, 360)), cv2.COLOR_BGR2RGB).astype(
        np.float32
    )
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return x / 255.0


def estimate_homography_and_polygon(
    points_xy: List[Tuple[float | None, float | None]], refer_kps: np.ndarray
) -> Tuple[None, None]:
    """Placeholder for homography estimation.

    Returns ``(None, None)`` until proper correspondence is implemented.
    """

    pts = np.array([p for p in points_xy if p[0] is not None], dtype=np.float32)
    if pts.shape[0] < 6:
        return None, None
    return None, None


def polygon_from_projected_kps(projected_pts: np.ndarray) -> None:
    """Placeholder for polygon construction from projected keypoints."""

    return None


def save_heat_overlay(img_bgr: np.ndarray, pred15: np.ndarray, out_path: str) -> None:
    """Save heatmap overlay for debugging."""

    hm = np.maximum(0, pred15).sum(0)
    mx = float(hm.max())
    if mx <= 0:
        hm = np.zeros_like(hm, dtype=np.uint8)
    else:
        hm = (255 * (hm / mx)).astype(np.uint8)
    hm = cv2.applyColorMap(
        cv2.resize(hm, (img_bgr.shape[1], img_bgr.shape[0])), cv2.COLORMAP_JET
    )
    over = (0.6 * img_bgr + 0.4 * hm).astype(np.uint8)
    cv2.imwrite(out_path, over)


def process_frames(
    frames_dir: str,
    weights_path: str,
    out_json: str,
    sample_rate: int = 1,
    low_thresh: int = 170,
    dump_heatmaps: bool = False,
    device: str = "cpu",
    kp_json_path: str | None = None,
) -> None:
    """Run court detection on a directory of frames."""

    device_str = device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    dev = torch.device(device_str)

    model = BallTrackerNet(out_channels=15)
    sd = load_tcd_state_dict(weights_path)
    model.load_state_dict(sd, strict=True)
    model.eval().to(dev)

    patterns = [
        "frame_*.png",
        "frame_*.jpg",
        "frame_*.jpeg",
        "*.png",
        "*.jpg",
        "*.jpeg",
    ]
    frames: List[str] = []
    for pat in patterns:
        frames.extend(glob.glob(os.path.join(frames_dir, pat)))

    def _natkey(p: str) -> List[int | str]:
        """Key for natural sorting based on numeric suffix."""

        b = os.path.basename(p)
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", b)]

    frames = sorted(frames, key=_natkey)
    if not frames:
        print(f"[court][ERROR] No frames found in {frames_dir} (png/jpg)")
        raise SystemExit(2)
    out = []
    all_kps: List[dict] = []

    for i, fp in enumerate(frames):
        if (i % sample_rate) != 0:
            out.append(
                {
                    "frame": os.path.basename(fp),
                    "polygon": [],
                    "lines": {},
                    "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "score": 0.0,
                    "placeholder": True,
                }
            )
            continue

        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            out.append(
                {
                    "frame": os.path.basename(fp),
                    "polygon": [],
                    "lines": {},
                    "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "score": 0.0,
                    "placeholder": True,
                }
            )
            continue

        x = preprocess_bgr(img, dev)
        with torch.inference_mode():
            hm = model(x)[0]
            pred = torch.sigmoid(hm).cpu().numpy()

        if dump_heatmaps:
            save_heat_overlay(img, pred, fp + ".heat.png")

        points: List[Tuple[int | None, int | None]] = []
        for k in range(14):
            heat = (pred[k] * 255.0).astype(np.uint8)
            xk, yk = heat_to_peak_xy(heat, low_thresh=low_thresh, max_radius=25)
            xk, yk = refine_kps_if_needed(img, xk, yk, k_idx=k)
            points.append((xk, yk))

        H, poly = None, None
        if H is None or poly is None:
            out.append(
                {
                    "frame": os.path.basename(fp),
                    "polygon": [],
                    "lines": {},
                    "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "score": 0.0,
                    "placeholder": True,
                }
            )
        else:
            out.append(
                {
                    "frame": os.path.basename(fp),
                    "polygon": poly,
                    "lines": {},
                    "homography": H.tolist(),
                    "score": 1.0,
                    "placeholder": False,
                }
            )
        all_kps.append({"frame": os.path.basename(fp), "kps": points})

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if kp_json_path:
        with open(kp_json_path, "w", encoding="utf-8") as f:
            json.dump(all_kps, f, indent=2)
    total = len(out)
    placeholders = sum(1 for it in out if it.get("placeholder", True))
    print(f"[court] frames={total} placeholders={placeholders} valid={total - placeholders}")


def build_argparser() -> argparse.ArgumentParser:
    """Construct the command line parser."""

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", required=True, help="Directory with input frames")
    p.add_argument("--output-json", required=True, help="Path for output JSON")
    p.add_argument("--out-json", dest="output_json", help=argparse.SUPPRESS)
    p.add_argument(
        "--weights", default="/app/weights/tcd.pth", help="Path to TCD weights"
    )
    p.add_argument("--device", default="cpu", help="Execution device: cpu or cuda")
    p.add_argument("--sample-rate", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--stride", dest="sample_rate", type=int, help=argparse.SUPPRESS)
    p.add_argument("--min-score", type=float, default=0.55, help="Reserved for future use")
    p.add_argument(
        "--mask-thr",
        type=float,
        default=0.67,
        help="Normalized threshold on sigmoid heatmaps [0..1]; 0.67 ~= 170/255",
    )
    p.add_argument("--score-metric", default="max", help="Reserved for future use")
    p.add_argument(
        "--dump-heatmaps",
        action="store_true",
        help="Write heatmap overlays next to frames",
    )
    p.add_argument(
        "--dump-kps-json",
        dest="kp_json_path",
        help="Optional path to write raw keypoints per frame",
    )
    return p


def main(args: List[str] | None = None) -> None:
    """Entry point for the CLI."""

    parser = build_argparser()
    ns = parser.parse_args(args)  # aliases already mapped via dest
    process_frames(
        frames_dir=ns.frames_dir,
        weights_path=ns.weights,
        out_json=ns.output_json,
        sample_rate=ns.sample_rate,
        low_thresh=int(ns.mask_thr * 255),
        dump_heatmaps=ns.dump_heatmaps,
        device=ns.device,
        kp_json_path=ns.kp_json_path,
    )


if __name__ == "__main__":
    main()
