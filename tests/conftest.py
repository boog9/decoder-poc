import sys
import types
from pathlib import Path

np_mod = types.ModuleType("numpy")
np_mod.array = lambda a, dtype=None: a
np_mod.asarray = lambda a, dtype=None: a
np_mod.concatenate = lambda arrs, axis=0: sum(arrs, [])
np_mod.float32 = "float32"
sys.modules.setdefault("numpy", np_mod)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
