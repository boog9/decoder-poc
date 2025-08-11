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
"""Tests for the `_get_tlwh_from_track` helper."""

from __future__ import annotations

import sys
import types
from pathlib import Path

class _DummyTqdm:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


sys.modules.setdefault("tqdm", types.ModuleType("tqdm")).tqdm = _DummyTqdm
loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
    remove=lambda *a, **k: None,
    add=lambda *a, **k: None,
)
sys.modules.setdefault("loguru", loguru_mod)
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
sys.modules.setdefault("scipy.optimize", types.ModuleType("scipy.optimize")).linear_sum_assignment = lambda *a, **k: ([], [])
np_mod = types.ModuleType("numpy")
np_mod.array = lambda a, dtype=None: a
np_mod.asarray = lambda a, dtype=None: a
np_mod.concatenate = lambda arrs, axis=0: sum(arrs, [])
np_mod.float32 = "float32"
sys.modules.setdefault("numpy", np_mod)
torch_mod = types.ModuleType("torch")
torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
sys.modules.setdefault("torch", torch_mod)
sys.modules.setdefault("torch.cuda", torch_mod.cuda)
sys.modules.setdefault("yolox", types.ModuleType("yolox"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.detect_objects as dobj
dobj.torch.cuda.is_available = lambda: True


def test_get_tlwh_from_track_variants() -> None:
    class TrackTLWHAttr:
        tlwh = [1, 2, 3, 4]

    class TrackTLWHMethod:
        def tlwh(self):  # type: ignore[override]
            return [1, 2, 3, 4]

    class TrackTLBRAttr:
        tlbr = [1, 2, 4, 6]

    class TrackTLBRMethod:
        def tlbr(self):  # type: ignore[override]
            return [1, 2, 4, 6]

    class TrackLegacy:
        _tlwh = [1, 2, 3, 4]

    for obj in [
        TrackTLWHAttr(),
        TrackTLWHMethod(),
        TrackTLBRAttr(),
        TrackTLBRMethod(),
        TrackLegacy(),
    ]:
        assert dobj._get_tlwh_from_track(obj) == (1.0, 2.0, 3.0, 4.0)
