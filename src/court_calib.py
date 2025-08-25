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
"""Court geometry calibration with homography interpolation.

The script samples frames from a directory, runs the tennis court detector on
key frames and linearly interpolates homographies for intermediate frames.

Example:
    python -m src.court_calib --frames-dir frames --out-json court.json \
        --device cuda --weights weights/tcd.pth --min-score 0.6 --stride 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from loguru import logger
from PIL import Image

from . import court_detector as cd
from .utils.checkpoint import verify_torch_ckpt


def _parametrize_h(h: List[List[float]]) -> List[float]:
    """Flatten a homography matrix into eight parameters."""

    norm = h[2][2] if h[2][2] else 1.0
    return [
        h[0][0] / norm,
        h[0][1] / norm,
        h[0][2] / norm,
        h[1][0] / norm,
        h[1][1] / norm,
        h[1][2] / norm,
        h[2][0] / norm,
        h[2][1] / norm,
    ]


def _deparametrize_h(p: List[float]) -> List[List[float]]:
    """Reconstruct a 3x3 homography matrix from eight parameters."""

    return [
        [p[0], p[1], p[2]],
        [p[3], p[4], p[5]],
        [p[6], p[7], 1.0],
    ]


def _interp_h(h0: List[List[float]], h1: List[List[float]], t: float) -> List[List[float]]:
    """Linearly interpolate two homographies."""

    p0 = _parametrize_h(h0)
    p1 = _parametrize_h(h1)
    p = [(1 - t) * a + t * b for a, b in zip(p0, p1)]
    return _deparametrize_h(p)


def _interp_pts(p0: List[List[float]], p1: List[List[float]], t: float) -> List[List[float]]:
    """Interpolate two point lists of equal length."""

    return [[(1 - t) * x0 + t * x1, (1 - t) * y0 + t * y1] for (x0, y0), (x1, y1) in zip(p0, p1)]


def _interp_lines(
    l0: Dict[str, List[List[float]]], l1: Dict[str, List[List[float]]], t: float
) -> Dict[str, List[List[float]]]:
    """Interpolate court line endpoints."""

    names = set(l0) | set(l1)
    out: Dict[str, List[List[float]]] = {}
    for n in names:
        if n in l0 and n in l1:
            out[n] = _interp_pts(l0[n], l1[n], t)
        elif n in l0:
            out[n] = l0[n]
        else:
            out[n] = l1[n]
    return out


def calibrate_court(
    frames_dir: Path,
    *,
    device: str,
    weights: Path,
    min_score: float,
    stride: int,
    allow_placeholder: bool,
) -> List[Dict[str, Any]]:
    """Detect court on key frames and interpolate homographies."""
    try:
        wtype = verify_torch_ckpt(str(weights))
    except Exception as exc:  # pragma: no cover - validation errors
        raise SystemExit(f"ERROR: missing or invalid weights: {weights}") from exc

    frame_paths = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda p: p.name,
    )
    if not frame_paths:
        logger.warning(
            "No frames found in %s (expected *.png/*.jpg/*.jpeg). Returning empty result.",
            frames_dir,
        )
        return []
    detections: Dict[int, Dict[str, Any]] = {}
    for idx, frame in enumerate(frame_paths):
        if idx % stride != 0:
            continue
        try:
            with Image.open(frame) as img:
                det = cd.detect_single_frame(
                    img, device=device, weights=weights, min_score=min_score
                )
        except Exception as e:  # pragma: no cover - IOError paths
            logger.warning("court_calib: failed on %s: %s", frame.name, e)
            continue
        rec = {
            "frame": frame.name,
            "polygon": det.get("polygon", []),
            "lines": det.get("lines", {}),
            "homography": det.get(
                "homography", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            "score": det.get("score", 0.0),
            "placeholder": not det,
        }
        detections[idx] = rec
    valid = {i: r for i, r in detections.items() if not r["placeholder"]}
    if not valid:
        logger.warning("No valid court detections found; aborting.")
        return []
    results: List[Dict[str, Any]] = []
    for i, frame in enumerate(frame_paths):
        name = frame.name
        if i in detections:
            rec = detections[i]
            if rec["placeholder"]:
                prev_idx = max([j for j in valid if j < i], default=None)
                next_idx = min([j for j in valid if j > i], default=None)
                src_idx = prev_idx
                if next_idx is not None and (
                    prev_idx is None or i - prev_idx > next_idx - i
                ):
                    src_idx = next_idx
                if src_idx is not None:
                    src = valid[src_idx]
                    rec = {
                        "frame": name,
                        "polygon": src["polygon"],
                        "lines": src["lines"],
                        "homography": src["homography"],
                        "score": src["score"],
                        "placeholder": allow_placeholder,
                    }
                else:
                    continue
            results.append(rec)
            continue
        prev_idx = max([j for j in valid if j < i], default=None)
        next_idx = min([j for j in valid if j > i], default=None)
        if prev_idx is None and next_idx is None:
            continue
        if prev_idx is None:
            src = valid[next_idx]
            results.append({**src, "frame": name, "placeholder": False})
            continue
        if next_idx is None:
            src = valid[prev_idx]
            results.append({**src, "frame": name, "placeholder": False})
            continue
        r0, r1 = valid[prev_idx], valid[next_idx]
        t = (i - prev_idx) / (next_idx - prev_idx)
        h = _interp_h(r0["homography"], r1["homography"], t)
        poly = _interp_pts(r0["polygon"], r1["polygon"], t)
        lines = _interp_lines(r0["lines"], r1["lines"], t)
        score = (1 - t) * r0["score"] + t * r1["score"]
        results.append(
            {
                "frame": name,
                "polygon": poly,
                "lines": lines,
                "homography": h,
                "score": score,
                "placeholder": False,
            }
        )
    results.sort(key=lambda r: r["frame"])
    valid_count = sum(1 for r in results if not r.get("placeholder"))
    if results:
        logger.info(
            "valid court frames: {}/{} ({:.1f}%)",
            valid_count,
            len(results),
            (valid_count / len(results)) * 100.0,
        )
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True, help="Input frames directory")
    # primary options
    parser.add_argument("--out-json", dest="out_json", type=Path, required=False, help="Output JSON path")
    parser.add_argument("--stride", type=int, default=5, help="Process every Nth frame")
    # backward-compatible aliases
    parser.add_argument("--output-json", dest="out_json", type=Path, required=False, help="Alias of --out-json")
    parser.add_argument("--sample-rate", dest="stride", type=int, required=False, help="Alias of --stride")
    parser.add_argument(
        "--stabilize",
        choices=["ema", "median"],
        required=False,
        help="(no-op) reserved for future smoothing",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--min-score", type=float, default=0.4, help="Minimum confidence score")
    parser.add_argument("--weights", type=Path, required=True, help="Path to detector weights")
    parser.add_argument(
        "--allow-placeholder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep frames with failed detections as placeholders",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    if getattr(args, "stabilize", None):
        logger.warning(
            "Flag --stabilize={} is currently a no-op in court_calib; smoothing may be added later.",
            args.stabilize,
        )
    data = calibrate_court(
        args.frames_dir,
        device=args.device,
        weights=args.weights,
        min_score=args.min_score,
        stride=args.stride,
        allow_placeholder=args.allow_placeholder,
    )
    if args.out_json is None:
        raise SystemExit("ERROR: --out-json/--output-json is required")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump(data, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
