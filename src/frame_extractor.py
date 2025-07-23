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
"""CLI script for extracting video frames using FFmpeg.

Example:
    python -m src.frame_extractor --input example.mp4 --output frames --fps 30
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def build_ffmpeg_command(input_video: str, output_dir: str, fps: int) -> List[str]:
    """Build the FFmpeg command for frame extraction.

    Args:
        input_video: Path to the input video file.
        output_dir: Directory where frames will be written.
        fps: Frames per second to extract.

    Returns:
        List of command tokens.
    """
    output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_video,
        "-vf",
        f"fps={fps}",
        output_pattern,
        "-progress",
        "-",
    ]


def extract_frames(input_video: str, output_dir: str, fps: int = 30) -> None:
    """Extract frames from ``input_video`` into ``output_dir`` using FFmpeg.

    Args:
        input_video: Video file to read.
        output_dir: Directory to store extracted frames.
        fps: Frame extraction rate.

    Raises:
        FileNotFoundError: If ``input_video`` does not exist.
        RuntimeError: If FFmpeg fails or is not installed.
    """
    if not os.path.isfile(input_video):
        raise FileNotFoundError(f"Input video '{input_video}' not found.")

    os.makedirs(output_dir, exist_ok=True)
    cmd = build_ffmpeg_command(input_video, output_dir, fps)
    logger.info("Running: %s", " ".join(cmd))

    try:
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("frame="):
                    logger.info(line)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg exited with status {proc.returncode}")
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg not installed or not found in PATH.") from exc


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract video frames with FFmpeg")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output folder for frames")
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second to extract"
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = parse_args(argv or sys.argv[1:])
    extract_frames(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()
