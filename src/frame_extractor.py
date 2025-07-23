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
"""CLI script for extracting video frames using FFmpeg."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
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
