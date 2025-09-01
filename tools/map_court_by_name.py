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
"""Map court records to frame names with atomic writes and friendly permissions."""

from __future__ import annotations

import glob
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Union

FRAMES_DIR = os.environ.get("FRAMES_DIR", "/app/frames")
IN_PATH = os.environ.get("COURT_JSON", "/app/court.json")
OUT_PATH = os.environ.get("OUT_JSON", "/app/court_by_name.json")


def build_frame_index() -> Dict[str, str]:
    """Return a map candidate_name -> actual file name that exists on disk."""

    idx: Dict[str, str] = {}
    for path in glob.glob(f"{FRAMES_DIR}/*.*"):
        p = pathlib.Path(path)
        stem = p.stem
        ext = p.suffix.lower()
        # 1) record the exact file name as present on disk
        idx[p.name] = p.name
        # 2) normalised stem+ext key maps to the actual on-disk name
        idx[stem + ext] = p.name
    return idx


def candidates(name: Union[int, str]) -> List[str]:
    """Return candidate file names for ``name``.

    Args:
        name: Frame index or explicit file name.

    Returns:
        Possible file name matches.
    """

    if isinstance(name, int):
        nums = [f"{name:06d}", str(name)]
        return [
            f"frame_{n}{ext}" for n in nums for ext in (".png", ".jpg", ".jpeg")
        ] + [f"{n}{ext}" for n in nums for ext in (".png", ".jpg", ".jpeg")]
    if isinstance(name, str):
        base = pathlib.Path(name).stem
        cand: List[str] = [name]
        cand += [base + ext for ext in (".png", ".jpg", ".jpeg")]
        if not name.startswith("frame_"):
            cand += [f"frame_{base}{ext}" for ext in (".png", ".jpg", ".jpeg")]
        seen, out = set(), []
        for item in cand:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out
    return []


def atomic_write_json(path: str, obj: Any, mode: int = 0o664) -> None:
    """Atomically write ``obj`` as JSON to ``path``.

    Args:
        path: Destination path.
        obj: Object to serialise.
        mode: File permissions to apply.
    """

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
    try:
        try:
            os.fchmod(fd, mode)
        except Exception:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, mode)
        except Exception:
            pass
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def main() -> None:
    """Entry point for the court mapping utility."""

    try:
        try:
            os.umask(0o022)
        except Exception:
            pass

        frames_idx = build_frame_index()
        data = json.load(open(IN_PATH, "r", encoding="utf-8"))

        iterable: Iterable = data.items() if isinstance(data, dict) else enumerate(data)
        out: Dict[str, Dict[str, Any]] = {}
        hit = miss = 0

        for _, rec in iterable:
            if not isinstance(rec, dict):
                miss += 1
                continue
            name = rec.get("file", rec.get("frame"))
            actual = None
            for cand in candidates(name):
                actual = frames_idx.get(cand)
                if actual:
                    break
            if actual:
                out[actual] = rec
                hit += 1
            else:
                miss += 1

        atomic_write_json(OUT_PATH, out)
        size = os.path.getsize(OUT_PATH)
        print(
            f"[court-map] frames_idx={len(frames_idx)} records={len(data) if hasattr(data,'__len__') else 'n/a'} hits={hit} miss={miss}"
        )
        print(f"[court-map] wrote: {OUT_PATH} size={size} bytes")
        print("[court-map] first 10 keys:", list(out.keys())[:10])
    except Exception as exc:
        print(f"[court-map][error] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
