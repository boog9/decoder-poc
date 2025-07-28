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
"""Tests for :mod:`src.detect_image`."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
import contextlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Provide minimal torch and PIL stubs before importing the module
torch_mod = types.ModuleType("torch")
torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True)
torch_mod.hub = types.SimpleNamespace(load=lambda *a, **k: None)
torch_mod.device = lambda x: x
torch_mod.no_grad = contextlib.nullcontext
sys.modules.setdefault("torch", torch_mod)
tv_mod = types.ModuleType("torchvision")
transforms_mod = types.ModuleType("torchvision.transforms")
functional_mod = types.ModuleType("torchvision.transforms.functional")
functional_mod.to_tensor = lambda img: img
transforms_mod.functional = functional_mod
tv_mod.transforms = transforms_mod
sys.modules.setdefault("torchvision", tv_mod)
sys.modules.setdefault("torchvision.transforms", transforms_mod)
sys.modules.setdefault("torchvision.transforms.functional", functional_mod)
sys.modules.setdefault("tqdm", types.ModuleType("tqdm")).tqdm = lambda *a, **k: a[0] if a else None
loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
)
sys.modules.setdefault("loguru", loguru_mod)
pil_mod = types.ModuleType("PIL")
pil_mod.__path__ = []
pil_image_mod = types.ModuleType("PIL.Image")
pil_draw_mod = types.ModuleType("PIL.ImageDraw")
pil_image_mod.open = lambda p: None
pil_draw_mod.Draw = lambda img: None
pil_mod.Image = object
pil_mod.ImageDraw = pil_draw_mod
sys.modules.setdefault("PIL", pil_mod)
sys.modules.setdefault("PIL.Image", pil_image_mod)
sys.modules.setdefault("PIL.ImageDraw", pil_draw_mod)

import src.detect_image as di


def test_parse_args_defaults() -> None:
    args = di.parse_args(["--image", "img.jpg"])
    assert isinstance(args, argparse.Namespace)
    assert args.model == "yolox-s"
    assert args.img_size == 640


def test_detect_image(monkeypatch) -> None:
    class DummyTensor:
        def unsqueeze(self, dim):
            return self

        def to(self, device):
            return self

    class FakeDet(list):
        dtype = "float32"

        def tolist(self):
            return [list(self)]

        def cpu(self):
            return self

    class FakeModel:
        head = types.SimpleNamespace(decode_outputs=lambda out, dtype: out)

        def __call__(self, tensor):
            return [[FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]]

    monkeypatch.setattr(di, "_load_model_device", lambda *a, **k: FakeModel())
    monkeypatch.setattr(di.dobj, "_preprocess_image", lambda *a, **k: (DummyTensor(), 1.0, 0, 0, 10, 10))
    utils_mod = types.ModuleType("utils")
    utils_mod.postprocess = lambda outputs, num_classes, conf_thre, nms_thre, class_agnostic=False: [FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]
    sys.modules["yolox"] = types.ModuleType("yolox")
    sys.modules["yolox.utils"] = utils_mod

    detections = di.detect_image(Path("img.jpg"), "yolox-s", 640, 0.1, 0.45, device="cpu")

    assert detections
    assert detections[0]["class"] == 0

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)

