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
"""Minimal court calibration pipeline.

This script loads a checkpoint-based tennis court detector, runs inference on
sampled frames, and writes polygons and placeholder metadata to a JSON file.
The post-processing is intentionally lightweight and serves as a smoke test for
pipeline integration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from services.court_detector.tcd import (
    TennisCourtDetectorFromSD,
    preprocess_to_640x360,
)


def ema_poly(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    """Exponentially smoothed polygon."""

    if prev is None:
        return cur
    return (1.0 - alpha) * prev + alpha * cur


def box_from_heatmaps(
    hm: torch.Tensor,
    mask_thr: float = 0.30,
    score_metric: str = "max",  # 'max' | 'mean' | 'area' | 'auto'
) -> Tuple[np.ndarray, float]:
    """Return polygon and score from raw heatmaps.

    Args:
        hm: Heatmaps of shape ``[1, 15, 360, 640]``.
        mask_thr: Threshold on normalized heatmaps for contour masking.
        score_metric: Aggregation metric for score; one of ``"max"``, ``"mean"``,
            ``"area"`` or ``"auto"``.

    Returns:
        ``(polygon, score)`` where polygon is ``[4, 2]`` in 640×360 coords and
        score is a rough confidence estimate in ``[0, 1]``.
    """

    with torch.no_grad():
        act = hm.clamp_min(0).sum(dim=1, keepdim=True)
        act_np = act.squeeze().cpu().numpy()
        q = float(np.quantile(act_np, 0.999))
        denom = q if q > 1e-6 else float(act_np.max() if act_np.max() > 0 else 1.0)
        act_norm = (act_np / denom).clip(0.0, 1.0)

        s_mean = float(act_norm.mean())
        s_max = float(act_norm.max())
        area_frac = float((act_norm > mask_thr).mean())

        if score_metric == "mean":
            score = s_mean
        elif score_metric == "area":
            score = area_frac
        elif score_metric == "auto":
            score = 0.4 * s_max + 0.4 * s_mean + 0.2 * area_frac
        else:  # "max" (default)
            score = s_max

        mask = (act_norm > mask_thr).astype(np.uint8)
        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.array([[0, 0], [640, 0], [640, 360], [0, 360]], np.float32), score
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32), score


def build_model(weights_path: str, device: str) -> torch.nn.Module:
    """Build a dynamic court detector from a checkpoint."""

    try:
        sd_raw = torch.load(weights_path, map_location="cpu", weights_only=True)  # PyTorch 2.4+
    except TypeError:
        sd_raw = torch.load(weights_path, map_location="cpu")
    sd = sd_raw.get("state_dict", sd_raw) if isinstance(sd_raw, dict) else sd_raw
    model = TennisCourtDetectorFromSD(sd)
    info = model.load_state_dict(sd, strict=False)
    missing = getattr(info, "missing_keys", [])
    unexpected = getattr(info, "unexpected_keys", [])
    if missing or unexpected:
        print(
            f"[WARN] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}",
            file=sys.stderr,
        )
    model.to(device).eval()
    return model


def iter_frames(frames_dir: Path) -> List[Path]:
    """Return sorted list of frame paths."""

    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return [p for p in sorted(frames_dir.iterdir()) if p.suffix.lower() in exts]


def scale_poly(poly640: np.ndarray, width: int, height: int) -> List[List[float]]:
    """Scale polygon from 640×360 space to original image size."""

    sx, sy = width / 640.0, height / 360.0
    poly = poly640.copy()
    poly[:, 0] *= sx
    poly[:, 1] *= sy
    return [[float(x), float(y)] for x, y in poly]


def identity_h() -> List[List[float]]:
    """Return a 3×3 identity homography."""

    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def lerp_poly(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate between two polygons.

    Args:
        p0: Starting polygon of shape ``[4, 2]``.
        p1: Ending polygon of shape ``[4, 2]``.
        t: Interpolation factor in ``[0, 1]``.

    Returns:
        Interpolated polygon ``[4, 2]``.
    """

    return (1.0 - t) * p0 + t * p1


def main() -> None:
    """Entry point for the CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-json", "--out-json", dest="output_json", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--sample-rate", "--stride", dest="sample_rate", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument(
        "--mask-thr",
        type=float,
        default=0.30,
        help="threshold for mask on normalized heatmaps [0..1]",
    )
    parser.add_argument(
        "--score-metric",
        choices=["max", "mean", "area", "auto"],
        default="max",
    )
    parser.add_argument(
        "--fallback",
        choices=["last", "full", "detect"],
        default="last",
        help=(
            "what polygon to use when score < min-score (and mark placeholder): "
            "'last' = last valid polygon if any else full-frame; "
            "'full' = full-frame; "
            "'detect' = use detected polygon even if low-confidence"
        ),
    )
    parser.add_argument("--smooth", choices=["none", "ema"], default="none")
    parser.add_argument("--smooth-alpha", type=float, default=0.3)
    parser.add_argument(
        "--interp",
        choices=["hold", "linear"],
        default="hold",
        help=(
            "interpolate polygons for non-key frames: "
            "'hold' = repeat last, 'linear' = linear interpolation between key frames"
        ),
    )
    args = parser.parse_args()

    # Auto-fallback: switch to CPU if CUDA is unavailable
    if args.device == "cuda" and not torch.cuda.is_available():
        print("| WARN | CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    frames_dir = Path(args.frames_dir)
    out_path = Path(args.output_json)

    print(
        f"| INFO | loading court detector (state_dict) from {args.weights} on {args.device}",
        file=sys.stderr,
    )
    model = build_model(args.weights, args.device)

    frames = iter_frames(frames_dir)
    if not frames:
        raise SystemExit(f"No frames in {frames_dir}")

    results: List[Dict[str, Any]] = []
    prev_poly: Optional[np.ndarray] = None  # EMA smoothing in 640x360
    last_valid_poly: Optional[List[List[float]]] = None  # original frame coords
    last_base_poly: Optional[List[List[float]]] = None  # polygon of last key frame
    gap_buffer: List[Tuple[int, str, int, int]] = []  # [(idx, frame_name, w, h)]
    skipped_count = 0
    lowconf_count = 0

    for idx, fp in enumerate(frames):
        # accumulate skipped frames for later interpolation
        if idx % max(1, args.sample_rate) != 0:
            img_tmp = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img_tmp is not None:
                h_, w_ = img_tmp.shape[:2]
                gap_buffer.append((idx, fp.name, w_, h_))
                skipped_count += 1
            continue

        img0 = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        ten = preprocess_to_640x360(img0).to(args.device)
        with torch.no_grad():
            hm = model(ten)
        poly640, score = box_from_heatmaps(hm, args.mask_thr, args.score_metric)
        if args.smooth == "ema":
            poly640 = ema_poly(prev_poly, poly640, args.smooth_alpha)
        prev_poly = poly640.copy()

        # determine polygon for current key frame
        if score < args.min_score:
            if args.fallback == "detect":
                poly_out = scale_poly(poly640, w0, h0)
            elif args.fallback == "last" and last_valid_poly is not None:
                poly_out = last_valid_poly
            else:  # "full" or no last valid polygon
                poly_out = [
                    [0.0, 0.0],
                    [float(w0), 0.0],
                    [float(w0), float(h0)],
                    [0.0, float(h0)],
                ]
            base_placeholder = True
            lowconf_count += 1
        else:
            poly_out = scale_poly(poly640, w0, h0)
            last_valid_poly = poly_out
            base_placeholder = False

        # interpolate skipped frames if any
        if last_base_poly is not None and gap_buffer:
            p0 = np.asarray(last_base_poly, dtype=np.float32)
            p1 = np.asarray(poly_out, dtype=np.float32)
            n = len(gap_buffer)
            for k, (j_idx, j_name, j_w, j_h) in enumerate(gap_buffer, start=1):
                if args.interp == "linear":
                    t = k / (n + 1)
                    pj = lerp_poly(p0, p1, t)
                else:
                    pj = p0
                pj_list = [[float(x), float(y)] for x, y in pj.tolist()]
                results.append(
                    {
                        "frame": j_name,
                        "polygon": pj_list,
                        "lines": {},
                        "homography": identity_h(),
                        "score": float(score),
                        "placeholder": True,
                    }
                )
            gap_buffer.clear()

        # append current key frame
        results.append(
            {
                "frame": fp.name,
                "polygon": poly_out,
                "lines": {},
                "homography": identity_h(),
                "score": float(score),
                "placeholder": base_placeholder,
            }
        )
        last_base_poly = poly_out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    valid = sum(1 for r in results if not r.get("placeholder", True))
    placeholders = len(results) - valid
    print(
        f"| INFO | valid court frames: {valid}/{len(frames)} "
        f"(placeholders={placeholders}, skipped={skipped_count}, lowconf={lowconf_count})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
