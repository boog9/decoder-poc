import sys
import types
from pathlib import Path

class _DummyArray(list):
    def __getitem__(self, idx):
        if isinstance(idx, tuple) and idx == (slice(None), None):
            return _DummyArray([[x] for x in self])
        res = super().__getitem__(idx)
        return _DummyArray(res) if isinstance(res, list) else res


np_mod = types.ModuleType("numpy")

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
