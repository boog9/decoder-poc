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
"""Tennis court detector with optional geometry stabilisation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from statistics import median
from loguru import logger
from PIL import Image


# Normalised template coordinates (unused but kept for reference)
TEMPLATE_POLYGON = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
TEMPLATE_LINES = {
    "baseline_top": [[0.0, 0.0], [1.0, 0.0]],
    "baseline_bottom": [[0.0, 1.0], [1.0, 1.0]],
    "service_center": [[0.5, 0.0], [0.5, 1.0]],
    "service_top": [[0.0, 0.25], [1.0, 0.25]],
    "service_bottom": [[0.0, 0.75], [1.0, 0.75]],
}


def _stub_detect(w: int, h: int) -> Dict[str, Any]:
    """Return a deterministic court geometry for placeholder inference."""
    mx, my = w * 0.1, h * 0.1
    poly = [[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]]
    lines: Dict[str, List[List[float]]] = {
        "baseline_top": [[mx, my], [w - mx, my]],
        "baseline_bottom": [[mx, h - my], [w - mx, h - my]],
        "service_center": [[w / 2, my], [w / 2, h - my]],
        "service_top": [[mx, my + (h - 2 * my) / 4], [w - mx, my + (h - 2 * my) / 4]],
        "service_bottom": [[mx, h - my - (h - 2 * my) / 4], [w - mx, h - my - (h - 2 * my) / 4]],
    }
    return {
        "polygon": poly,
        "lines": lines,
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "score": 0.9,
    }


def _ema(records: List[dict], alpha: float) -> None:
    """Apply exponential moving average smoothing in-place."""

    prev_poly: List[List[float]] | None = None
    prev_lines: Dict[str, List[List[float]]] = {}
    for rec in records:
        poly = rec["polygon"]
        if prev_poly is None:
            prev_poly = [p[:] for p in poly]
        else:
            smoothed = [
                [alpha * x + (1 - alpha) * px, alpha * y + (1 - alpha) * py]
                for (x, y), (px, py) in zip(poly, prev_poly)
            ]
            rec["polygon"] = smoothed
            prev_poly = [p[:] for p in smoothed]
        lines = rec.get("lines", {})
        new_lines: Dict[str, List[List[float]]] = {}
        for name, pts in lines.items():
            prev = prev_lines.get(name)
            if prev is None:
                prev_lines[name] = [p[:] for p in pts]
                new_lines[name] = pts
            else:
                sm = [
                    [alpha * x + (1 - alpha) * px, alpha * y + (1 - alpha) * py]
                    for (x, y), (px, py) in zip(pts, prev)
                ]
                prev_lines[name] = [p[:] for p in sm]
                new_lines[name] = sm
        rec["lines"] = new_lines


def _median(records: List[dict], window: int) -> None:
    """Apply sliding-window median smoothing in-place."""

    poly_buf: List[List[List[float]]] = []
    line_bufs: Dict[str, List[List[List[float]]]] = {}
    for rec in records:
        poly_buf.append(rec["polygon"])
        if len(poly_buf) > window:
            poly_buf.pop(0)
        med_poly = [
            [median([p[i][0] for p in poly_buf]), median([p[i][1] for p in poly_buf])]
            for i in range(len(rec["polygon"]))
        ]
        rec["polygon"] = med_poly
        lines = rec.get("lines", {})
        new_lines: Dict[str, List[List[float]]] = {}
        for name, pts in lines.items():
            buf = line_bufs.setdefault(name, [])
            buf.append(pts)
            if len(buf) > window:
                buf.pop(0)
            med_line = [
                [median([b[i][0] for b in buf]), median([b[i][1] for b in buf])]
                for i in range(len(pts))
            ]
            new_lines[name] = med_line
        rec["lines"] = new_lines


def _interpolate(records: Dict[int, dict], frame_paths: List[Path]) -> List[dict]:
    """Fill missing frames by linear interpolation of polygon vertices."""

    indices = sorted(records)
    results: List[dict] = []
    for i, frame in enumerate(frame_paths):
        name = frame.name
        if i in records:
            results.append(records[i])
            continue
        prev_idx = max([j for j in indices if j < i], default=None)
        next_idx = min([j for j in indices if j > i], default=None)
        if prev_idx is None or next_idx is None:
            continue
        r0, r1 = records[prev_idx], records[next_idx]
        t = (i - prev_idx) / (next_idx - prev_idx)
        interp = [
            [
                (1 - t) * p0[0] + t * p1[0],
                (1 - t) * p0[1] + t * p1[1],
            ]
            for p0, p1 in zip(r0["polygon"], r1["polygon"])
        ]
        score = (1 - t) * r0.get("score", 0.0) + t * r1.get("score", 0.0)
        results.append(
            {
                "frame": name,
                "polygon": interp,
                "score": score,
                "placeholder": False,
            }
        )
    results.sort(key=lambda r: r["frame"])
    return results


def detect_court(
    frames_dir: Path,
    *,
    device: str = "auto",
    weights: Path | None = None,
    use_homography: bool = True,
    refine_kps: bool = False,
    sample_rate: int = 5,
    min_conf: float = 0.4,
    allow_placeholder: bool = False,
    stabilize: str = "ema",
    stabilize_alpha: float = 0.2,
    stabilize_window: int = 7,
) -> List[Dict[str, Any]]:
    """Detect tennis court geometry for frames in ``frames_dir``.

    Returns a list with one entry per frame containing ``polygon`` and optional
    ``lines`` and ``homography`` keys. Frames without a valid detection are
    omitted unless ``allow_placeholder`` is set.
    """

    frame_paths = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    raw: Dict[int, dict] = {}
    for idx, frame in enumerate(frame_paths):
        if idx % sample_rate != 0:
            continue
        try:
            with Image.open(frame) as img:
                w, h = img.size
        except Exception:
            continue
        if weights is None:
            if allow_placeholder:
                poly = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                raw[idx] = {
                    "frame": frame.name,
                    "polygon": poly,
                    "score": 0.0,
                    "placeholder": True,
                    "reason": "no_weights",
                }
            continue
        det = _stub_detect(w, h)
        if det["score"] < min_conf:
            if allow_placeholder:
                poly = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                raw[idx] = {
                    "frame": frame.name,
                    "polygon": poly,
                    "score": det["score"],
                    "placeholder": True,
                    "reason": "low_confidence",
                }
            continue
        rec: Dict[str, Any] = {
            "frame": frame.name,
            "polygon": det["polygon"],
            "score": det["score"],
            "placeholder": False,
        }
        if use_homography:
            rec["homography"] = det["homography"]
        if det.get("lines"):
            rec["lines"] = det["lines"]
        raw[idx] = rec

    records = [raw[i] for i in sorted(raw)]
    if stabilize == "ema":
        _ema(records, stabilize_alpha)
    elif stabilize == "median":
        _median(records, stabilize_window)
    # propagate smoothing back into raw before interpolation
    for i, rec in zip(sorted(raw), records):
        raw[i] = rec

    results = _interpolate(raw, frame_paths)
    valid = [r for r in results if not r.get("placeholder")]
    scores = [r.get("score", 0.0) for r in valid]
    median_score = float(median(scores)) if scores else 0.0
    if hasattr(logger, "info"):
        logger.info(
            "valid polygons: {}/{} ({:.1f}%), median score {:.2f}",
            len(valid),
            len(frame_paths),
            (len(valid) / max(len(frame_paths), 1)) * 100.0,
            median_score,
        )
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True, help="Input frames directory")
    parser.add_argument("--output-json", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--weights", type=Path, default=None, help="TennisCourtDetector weights")
    parser.add_argument("--use-homography", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-kps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-rate", type=int, default=5, help="Process every Nth frame")
    parser.add_argument("--min-conf", type=float, default=0.4, help="Minimum confidence")
    parser.add_argument(
        "--allow-placeholder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit full-frame placeholder when detection fails",
    )
    parser.add_argument(
        "--stabilize",
        choices=["ema", "median", "none"],
        default="ema",
        help="Polygon stabilisation method",
    )
    parser.add_argument("--stabilize-alpha", type=float, default=0.2, help="EMA smoothing factor")
    parser.add_argument("--stabilize-window", type=int, default=7, help="Median window size")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    data = detect_court(
        args.frames_dir,
        device=args.device,
        weights=args.weights,
        use_homography=args.use_homography,
        refine_kps=args.refine_kps,
        sample_rate=args.sample_rate,
        min_conf=args.min_conf,
        allow_placeholder=args.allow_placeholder,
        stabilize=args.stabilize,
        stabilize_alpha=args.stabilize_alpha,
        stabilize_window=args.stabilize_window,
    )
    with args.output_json.open("w") as fh:
        json.dump(data, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
