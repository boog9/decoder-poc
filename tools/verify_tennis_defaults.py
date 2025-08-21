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
"""Sanity checks for tennis tracking outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Set


def _metrics(tracks: list[dict]) -> tuple[int, float, float]:
    """Return (unique players, fraction of frames with ball, avg ball track length).

    Args:
        tracks: Flat list of track dictionaries.
    """  # tennis tuning

    person_ids: Set[int] = set()
    ball_frames: Set[int] = set()
    ball_tracks: Dict[int, Set[int]] = {}
    all_frames: Set[int] = set()
    for det in tracks:
        frame = int(det.get("frame", 0))
        cls = det.get("class")
        tid = det.get("track_id")
        all_frames.add(frame)
        if cls in (0, "person") and tid is not None:
            person_ids.add(int(tid))
        if cls in (32, "sports ball") and tid is not None:
            ball_frames.add(frame)
            ball_tracks.setdefault(int(tid), set()).add(frame)
    frac_ball = (len(ball_frames) / len(all_frames)) if all_frames else 0.0
    avg_ball = (
        sum(len(v) for v in ball_tracks.values()) / len(ball_tracks)
        if ball_tracks
        else 0.0
    )
    return len(person_ids), frac_ball, avg_ball


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-json", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.tracks_json.open() as fh:
        data = json.load(fh)
    players, frac_ball, avg_ball = _metrics(data)
    print(
        f"players={players} ball_frame_frac={frac_ball:.2f} avg_ball_track={avg_ball:.1f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
