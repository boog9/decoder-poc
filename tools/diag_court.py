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
"""Diagnose homography stability for court.json outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

REF = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], np.float32)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Return root-mean-square error between ``a`` and ``b``."""

    return float(np.sqrt(np.mean((a - b) ** 2)))


def analyze(records: Iterable[dict]) -> Tuple[List[Tuple[str, float, float]], List[float], List[float]]:
    """Compute per-frame diagnostics."""

    entries: List[Tuple[str, float, float]] = []
    rmses: List[float] = []
    dets: List[float] = []
    for rec in records:
        frame = rec.get("frame", "?")
        poly = rec.get("polygon")
        H = rec.get("homography")
        if poly and H:
            P = np.array(poly, np.float32)
            Hm = np.array(H, np.float32)
            proj = cv2.perspectiveTransform(REF.reshape(-1, 1, 2), Hm).reshape(-1, 2)
            rmse_fwd = rmse(proj, P)
            try:
                H_inv = np.linalg.inv(Hm)
                back = cv2.perspectiveTransform(P.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
                rmse_bwd = rmse(back, REF)
            except np.linalg.LinAlgError:
                rmse_bwd = float("inf")
            err = min(rmse_fwd, rmse_bwd)
            det = float(np.linalg.det(Hm))
        else:
            err = float("inf")
            rmse_fwd = float("inf")
            det = float("nan")
        entries.append((frame, err, det))
        if np.isfinite(rmse_fwd):
            rmses.append(rmse_fwd)
        if np.isfinite(det):
            dets.append(det)
    return entries, rmses, dets


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--court", type=Path, required=True, help="Path to court.json")
    args = parser.parse_args(argv)

    data = json.loads(args.court.read_text())
    entries, rmses, dets = analyze(data)
    if rmses:
        pct = np.percentile(rmses, [0, 50, 90, 100])
        print(f"[RMSE_FWD] min={pct[0]:.2f} med={pct[1]:.2f} p90={pct[2]:.2f} max={pct[3]:.2f}")
    if dets:
        pct = np.percentile(dets, [0, 50, 90, 100])
        print(f"[DET(H)] min={pct[0]:.3f} med={pct[1]:.3f} p90={pct[2]:.3f} max={pct[3]:.3f}")

    top = sorted(entries, key=lambda x: x[1], reverse=True)[:25]
    lines = [f"{f}\t{e:.2f}\t{d:.3f}\n" for f, e, d in top]
    # Пишемо файл поруч із court.json, атомарно та без проблем з правами
    out_dir = Path(args.court).resolve().parent
    out_path = out_dir / "H_CHECK_TOP.txt"
    try:
        if out_path.exists():
            out_path.unlink()
    except Exception:
        pass
    tmp = out_dir / ".tmp_H_CHECK_TOP.txt"
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
