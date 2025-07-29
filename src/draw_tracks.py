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
"""Visualise tracking results on frame images."""

from __future__ import annotations

from loguru import logger

import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import click
import cv2
import numpy as np
from PIL import Image

from .draw_roi import COCO_CLASS_NAMES, _label_color
from .utils.draw_helpers import IMAGE_EXT, get_font, load_frames

__all__ = ["IMAGE_EXT", "load_frames", "get_font", "visualize_tracks", "cli"]


def _track_color(track_id: int) -> Tuple[int, int, int]:
    digest = hashlib.md5(str(track_id).encode()).digest()
    return digest[0], digest[1], digest[2]


def _class_color(class_name: str | int) -> Tuple[int, int, int]:
    if isinstance(class_name, int):
        cid = class_name
    else:
        try:
            cid = COCO_CLASS_NAMES.index(class_name)
        except ValueError:
            cid = -1
    return _label_color(cid)


def visualize_tracks(
    frames_dir: Path,
    tracks_json: Path,
    output_dir: Path | None = None,
    output_video: Path | None = None,
    label: bool = False,
    palette: str = "track",
    thickness: int = 2,
    max_frames: int | None = None,
    fps: float = 30.0,
) -> None:
    """Overlay tracking results on frames and save images or a video."""

    logger.info("Starting visualize_tracks()")
    logger.info("\u2192 frames_dir = %s", frames_dir)
    logger.info("\u2192 tracks_json = %s", tracks_json)
    logger.info("\u2192 output_dir = %s", output_dir)
    logger.info("\u2192 output_video = %s", output_video)

    if output_video and output_dir:
        raise ValueError("--output-dir and --output-video are mutually exclusive")

    with tracks_json.open() as fh:
        tracks = json.load(fh)
    if not isinstance(tracks, list):
        raise ValueError("tracks-json must contain a list")
    if not tracks:
        logger.warning("tracks.json is empty or has no detections")
        return

    frame_map: Dict[int, List[dict]] = defaultdict(list)
    for det in tracks:
        frame_idx = int(det.get("frame", 0))
        frame_map[frame_idx].append(det)

    if frame_map and min(frame_map.keys()) == 0:
        logger.warning(
            "Detected frame indices start at 0. Normalizing +1 for alignment."
        )
        frame_map = {fid + 1: dets for fid, dets in frame_map.items()}

    frames = load_frames(frames_dir, max_frames)
    if not frames:
        logger.warning("No valid frames found in %s", frames_dir)
        return
    logger.info("Loaded %d frame(s) from %s", len(frames), frames_dir)

    first_img = cv2.imread(str(frames[0]))
    if first_img is None:
        logger.error("Failed to read first frame %s", frames[0])
        raise RuntimeError(f"Failed to read {frames[0]}")

    font, font_scale, line_type = get_font()
    colors_cache: Dict[int, Tuple[int, int, int]] = {}

    writer: subprocess.Popen | None = None
    if output_video:
        h, w = first_img.shape[:2]
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{w}x{h}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-y",
            str(output_video),
        ]
        writer = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert writer.stdin is not None
        # first frame will be written in the main loop
        logger.info("Output mode: writing video to %s at %gfps", output_video, fps)
    else:
        output_dir = output_dir or Path("frames_tracks")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output mode: writing annotated frames to %s", output_dir)

    num_written = 0
    for idx, frame_path in enumerate(frames, start=1):
        if hasattr(logger, "debug"):
            logger.debug("Processing frame %d: %s", idx, frame_path)
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.warning("cv2.imread failed for %s, trying PIL fallback", frame_path)
            try:
                pil_img = Image.open(frame_path).convert("RGB")
                img = np.array(pil_img)
                try:
                    img = img[:, :, ::-1]
                except Exception:
                    pass
            except Exception as exc:
                logger.error("Failed to read frame %s with PIL: %s", frame_path, exc)
                continue
        detections = frame_map.get(idx, [])
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bbox)
            track_id = int(det.get("track_id", -1))
            class_name = det.get("class")

            if palette == "coco":
                color = _class_color(class_name)
            elif palette == "random":
                rng = random.Random(track_id)
                color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            else:
                color = colors_cache.setdefault(track_id, _track_color(track_id))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(img, (cx, cy), max(1, thickness), color, -1)
            if label:
                txt = f"id:{track_id}" if not class_name else f"{class_name} id:{track_id}"
                (tw, th), bl = cv2.getTextSize(txt, font, font_scale, 1)
                top = max(y1 - th - bl - 2, 0)
                cv2.rectangle(img, (x1, top), (x1 + tw, top + th + bl), color, -1)
                cv2.putText(
                    img,
                    txt,
                    (x1, top + th),
                    font,
                    font_scale,
                    (255, 255, 255),
                    1,
                    line_type,
                )

        if writer:
            writer.stdin.write(img.tobytes())
            num_written += 1
        else:
            out_path = output_dir / frame_path.name
            cv2.imwrite(str(out_path), img)
            num_written += 1

        if idx % 100 == 0:
            logger.info("Processed %d frames", idx)

    if writer:
        writer.stdin.close()
        writer.wait()
        if output_video.exists():
            logger.info(
                "Finished writing %d frame(s) to video: %s", num_written, output_video
            )
        else:
            logger.error("Expected output video not found at %s", output_video)
    else:
        logger.info(
            "Finished writing %d frame(s) to %s", num_written, output_dir
        )


def _validate_outputs(ctx: click.Context, param: click.Parameter, value: Path | None) -> Path | None:
    """Validate mutually exclusive output options for the CLI.

    This callback ensures exactly one of ``--output-dir`` or ``--output-video``
    is provided when calling the command.
    """

    other = ctx.params.get("output_video" if param.name == "output_dir" else "output_dir")
    if ctx.resilient_parsing:
        return value
    if (value is None and other is None) or (value is not None and other is not None):
        raise click.BadParameter("Specify exactly one of --output-dir or --output-video")
    return value


@click.command()
@click.option("--frames-dir", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--tracks-json", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None, callback=_validate_outputs)
@click.option("--output-video", type=click.Path(path_type=Path), default=None, callback=_validate_outputs)
@click.option("--label/--no-label", default=False, show_default=True)
@click.option("--palette", type=click.Choice(["coco", "random", "track"]), default="track", show_default=True)
@click.option("--thickness", type=int, default=2, show_default=True)
@click.option("--max-frames", type=int, default=None)
@click.option("--fps", type=float, default=30.0, show_default=True)
def cli(
    frames_dir: Path,
    tracks_json: Path,
    output_dir: Path | None,
    output_video: Path | None,
    label: bool,
    palette: str,
    thickness: int,
    max_frames: int | None,
    fps: float,
) -> None:
    """Command line interface for :func:`visualize_tracks`."""

    logger.info("CLI started")

    visualize_tracks(
        frames_dir,
        tracks_json,
        output_dir if output_video is None else None,
        output_video,
        label,
        palette,
        thickness,
        max_frames,
        fps,
    )

