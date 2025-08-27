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
from __future__ import annotations
import argparse, glob, json, os, re
from typing import List, Tuple
import cv2, numpy as np, torch
from .tcd_model import BallTrackerNet
from .utils_weights import load_tcd_state_dict
from .postproc import heat_to_peak_xy, refine_kps_if_needed

def _order_quad_tl_tr_br_bl(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    s = box.sum(axis=1); diff = np.diff(box, axis=1).reshape(-1)
    tl = box[np.argmin(s)]; br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]; bl = box[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

W_IN, H_IN = 640, 360


def preprocess_bgr(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Resize BGR image to 640×360 and convert to [0,1] tensor."""

    rgb = cv2.cvtColor(cv2.resize(img, (W_IN, H_IN)), cv2.COLOR_BGR2RGB).astype(
        np.float32
    )
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return x / 255.0

def save_heat_overlay(img_bgr: np.ndarray, pred15: np.ndarray, out_path: str) -> None:
    hm = np.maximum(0, pred15).sum(0)
    mx = float(hm.max())
    hm = np.zeros_like(hm, dtype=np.uint8) if mx <= 0 else (255 * (hm / mx)).astype(np.uint8)
    hm = cv2.applyColorMap(cv2.resize(hm, (img_bgr.shape[1], img_bgr.shape[0])), cv2.COLORMAP_JET)
    over = (0.6 * img_bgr + 0.4 * hm).astype(np.uint8)
    cv2.imwrite(out_path, over)

def process_frames(frames_dir: str, weights_path: str, out_json: str,
                   sample_rate: int = 1, low_thresh: int = 170,
                   dump_heatmaps: bool = False, device: str = "cpu") -> None:
    device_str = device if not (device == "cuda" and not torch.cuda.is_available()) else "cpu"
    dev = torch.device(device_str)
    model = BallTrackerNet(out_channels=15)
    sd = load_tcd_state_dict(weights_path)
    model.load_state_dict(sd, strict=True)
    model.eval().to(dev)

    patterns = ["frame_*.png", "frame_*.jpg", "frame_*.jpeg"]
    frames: List[str] = []
    for pat in patterns:
        frames.extend(glob.glob(os.path.join(frames_dir, pat)))
    frames = sorted({f for f in frames if not f.endswith(".heat.png")})
    if not frames:
        print(f"[court][ERROR] No frames found in {frames_dir} (png/jpg)")
        raise SystemExit(2)

    out = []; all_kps: List[dict] = []
    for i, fp in enumerate(frames):
        if (i % sample_rate) != 0:
            out.append({"frame": os.path.basename(fp), "polygon": [], "lines": {},
                        "homography": [[1,0,0],[0,1,0],[0,0,1]], "score": 0.0, "placeholder": True})
            continue
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            out.append({"frame": os.path.basename(fp), "polygon": [], "lines": {},
                        "homography": [[1,0,0],[0,1,0],[0,0,1]], "score": 0.0, "placeholder": True})
            continue
        H0, W0 = img.shape[:2]
        x = preprocess_bgr(img, dev)
        with torch.inference_mode():
            hm = model(x)[0]
            pred = torch.sigmoid(hm).cpu().numpy()
        if dump_heatmaps:
            save_heat_overlay(img, pred, fp + ".heat.png")

        points_360: List[Tuple[int | None, int | None]] = []
        for k in range(14):
            heat = (pred[k] * 255.0).astype(np.uint8)
            xk, yk = heat_to_peak_xy(heat, low_thresh=low_thresh, max_radius=25)
            xk, yk = refine_kps_if_needed(img, xk, yk, k_idx=k)
            points_360.append((xk, yk))

        # scale to native resolution
        sx, sy = (W0 / float(W_IN)), (H0 / float(H_IN))
        points_native: List[Tuple[int | None, int | None]] = []
        for (xk, yk) in points_360:
            if xk is None or yk is None:
                points_native.append((None, None))
            else:
                points_native.append((int(round(xk * sx)), int(round(yk * sy))))

        valid = np.array(
            [(x, y) for (x, y) in points_native if x is not None and y is not None],
            dtype=np.float32,
        )
        if valid.shape[0] >= 4:
            rect = cv2.minAreaRect(valid)
            box = cv2.boxPoints(rect)
            box = _order_quad_tl_tr_br_bl(box)
            poly = [[int(p[0]), int(p[1])] for p in box]
            score = float(valid.shape[0]) / 14.0
            out.append({"frame": os.path.basename(fp), "polygon": poly, "lines": {},
                        "homography": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                        "score": score, "placeholder": False})
        else:
            out.append({"frame": os.path.basename(fp), "polygon": [], "lines": {},
                        "homography": [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                        "score": 0.0, "placeholder": True})
        all_kps.append({
            "frame": os.path.basename(fp),
            "kps_native": points_native,
            "kps_360": points_360,
        })

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    total = len(out); placeholders = sum(1 for it in out if it.get("placeholder", True))
    print(f"[court] frames={total} placeholders={placeholders} valid={total - placeholders}")

    kp_path = os.environ.get("KP_JSON_PATH")
    if kp_path:
        with open(kp_path, "w", encoding="utf-8") as f:
            json.dump(all_kps, f, indent=2)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", required=True, help="Directory with input frames")
    p.add_argument("--output-json", help="Path for output JSON")
    p.add_argument("--out-json", dest="output_json", help=argparse.SUPPRESS)
    p.add_argument("--weights", default="/app/weights/tcd.pth", help="Path to TCD weights")
    p.add_argument("--device", default="cpu", help="Execution device: cpu or cuda")
    p.add_argument("--sample-rate", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--stride", dest="sample_rate", type=int, help=argparse.SUPPRESS)
    p.add_argument("--min-score", type=float, default=0.55, help="Reserved for future use")
    p.add_argument("--mask-thr", type=float, default=0.67,
                   help="Normalized threshold on sigmoid heatmaps [0..1]; 0.67 ~= 170/255")
    p.add_argument("--score-metric", default="max", help="Reserved for future use")
    p.add_argument("--dump-heatmaps", action="store_true", help="Write heatmap overlays next to frames")
    p.add_argument("--dump-kps-json", dest="kp_json_path", default=None, help="Write raw keypoints JSON (debug)")
    return p

def main(args: List[str] | None = None) -> None:
    parser = build_argparser()
    ns = parser.parse_args(args)
    if not ns.output_json:
        parser.error("--output-json (or --out-json) is required")
    if ns.kp_json_path:
        os.environ["KP_JSON_PATH"] = ns.kp_json_path
    process_frames(frames_dir=ns.frames_dir, weights_path=ns.weights, out_json=ns.output_json,
                   sample_rate=ns.sample_rate, low_thresh=int(ns.mask_thr * 255),
                   dump_heatmaps=ns.dump_heatmaps, device=ns.device)

if __name__ == "__main__":
    main()
