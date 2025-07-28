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
"""Additional tests for :mod:`src.detect_objects`."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
)
sys.modules.setdefault("loguru", loguru_mod)

sys.modules.setdefault("tqdm", types.ModuleType("tqdm")).tqdm = lambda *a, **k: None

torch_mod = types.ModuleType("torch")
torch_mod.as_tensor = lambda arr, dtype=None: arr
torch_mod.float32 = "float32"
torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
sys.modules.setdefault("torch", torch_mod)

pil_mod = types.ModuleType("PIL")
pil_mod.__path__ = []
pil_image_mod = types.ModuleType("PIL.Image")
pil_mod.Image = object
sys.modules.setdefault("PIL", pil_mod)
sys.modules.setdefault("PIL.Image", pil_image_mod)

np_mod = types.ModuleType("numpy")
class _DummyArray(list):
    def __getitem__(self, idx):
        if isinstance(idx, tuple) and idx == (slice(None), None):
            return _DummyArray([[x] for x in self])
        res = super().__getitem__(idx)
        return _DummyArray(res) if isinstance(res, list) else res


def _array(a, dtype=None):
    return _DummyArray(a)


def _asarray(a, dtype=None):
    return _DummyArray(a)


def _concatenate(arrs, axis=0):
    if axis == 0:
        res = []
        for arr in arrs:
            res.extend(arr)
        return _DummyArray(res)
    if axis == 1:
        return _DummyArray([sum(map(list, t), []) for t in zip(*arrs)])
    raise ValueError("axis must be 0 or 1")


np_mod.array = _array
np_mod.asarray = _asarray
np_mod.concatenate = _concatenate
np_mod.float32 = "float32"
sys.modules.setdefault("numpy", np_mod)

import src.detect_objects as dobj


def test_update_tracker_tensor(monkeypatch) -> None:
    class DummyTensor(list):
        pass

    class DummyTorch:
        float32 = "float32"

        def as_tensor(self, arr, dtype=None):
            return DummyTensor(arr)

    monkeypatch.setattr(dobj, "torch", DummyTorch())

    class DummyTracker:
        def __init__(self) -> None:
            self.args = None

        def update(self, outputs, img_info, img_size):
            assert isinstance(outputs, DummyTensor)
            self.args = (outputs, img_info, img_size)
            return ["ok"]

    tracker = DummyTracker()

    import inspect
    orig_sig = inspect.signature

    def _sig(func):
        s = orig_sig(func)
        print('params', list(s.parameters)[1:])
        return s

    monkeypatch.setattr(dobj.inspect, "signature", _sig)

    res = dobj._update_tracker(
        tracker,
        [[0, 0, 10, 20]],
        [0.9],
        ["person"],
        1,
    )

    assert res == ["ok"]
    assert isinstance(tracker.args[0], DummyTensor)
