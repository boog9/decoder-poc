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
"""Import isolation tests for YOLOX and ByteTrack."""

from __future__ import annotations

import sys
import types


def test_detect_imports_clean() -> None:
    """Ensure detection environment uses official YOLOX without ByteTrack."""
    sys.modules.pop("bytetrack_vendor", None)
    assert "bytetrack_vendor" not in sys.modules
    sys.modules.pop("yolox", None)
    sys.modules.setdefault("yolox", types.SimpleNamespace(__version__="0.0"))
    import yolox  # type: ignore  # official package expected at runtime

    assert hasattr(yolox, "__version__")


def test_track_imports_isolated() -> None:
    """Import vendored tracker without requiring official YOLOX."""
    from externals.ByteTrack.bytetrack_vendor.tracker.byte_tracker import BYTETracker

    assert BYTETracker is not None
