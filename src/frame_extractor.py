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
import time
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
    
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def extract_frames(input_video: Path, output_dir: Path, fps: int) -> None:
    """Extract frames from a video using FFmpeg.

    Args:
        input_video: Path to the video file.
        output_dir: Directory to store extracted frames.
        fps: Frames per second for extraction.
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        str(input_video),
        "-vf",
        f"fps={fps}",
        str(output_dir / "frame_%06d.jpg"),
        "-loglevel",
        "error",
        "-progress",
        "pipe:1",
    ]

    LOGGER.debug("Running command: %s", " ".join(cmd))

    frames_processed = 0
    start = time.monotonic()
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("frame="):
                frames_processed = int(line.split("=", 1)[1])
                LOGGER.debug(line)
            elif line.startswith("progress=") and line != "progress=continue":
                LOGGER.debug(line)

    elapsed = time.monotonic() - start
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    LOGGER.info(
        "Completed extraction of %d frames in %.2f seconds", frames_processed, elapsed
    )



def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Directory to save extracted frames.",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=30,
        help="Frames per second to extract (default: 30).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        extract_frames(args.input, args.output, args.fps)
    except Exception as exc:  # pragma: no cover - top-level error handling
        LOGGER.error("Failed to extract frames: %s", exc)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()
