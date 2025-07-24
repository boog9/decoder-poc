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
"""Tests for :mod:`src.detect_objects`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch_stub = types.ModuleType("torch")
torch_stub.cuda = types.SimpleNamespace(
    is_available=lambda: False, mem_get_info=lambda: (0, 0)
)
torch_stub.hub = types.SimpleNamespace(load=lambda *a, **k: None)

class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

torch_stub.no_grad = lambda: _NoGrad()
sys.modules.setdefault("torch", torch_stub)

tv_stub = types.ModuleType("torchvision")
tv_transforms = types.ModuleType("torchvision.transforms")
tv_transforms.functional = types.SimpleNamespace(to_tensor=lambda x: x)
tv_stub.transforms = tv_transforms
sys.modules.setdefault("torchvision", tv_stub)
sys.modules.setdefault("torchvision.transforms", tv_transforms)
sys.modules.setdefault("torchvision.transforms.functional", tv_transforms.functional)

sys.modules.setdefault("PIL", types.ModuleType("PIL")).Image = object()
class _DummyTqdm:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n=1):
        pass

    def set_postfix(self, *a, **k):
        pass

sys.modules.setdefault("tqdm", types.ModuleType("tqdm")).tqdm = _DummyTqdm

import src.detect_objects as dobj


def test_parse_args_defaults() -> None:
    args = dobj.parse_args([
        "--frames-dir",
        "frames",
        "--output-json",
        "out.json",
    ])
    assert isinstance(args, argparse.Namespace)
    assert args.model == "yolox-s"


def test_load_model_translates_hyphen(monkeypatch) -> None:
    recorded = {}

    def fake_load(repo, name, pretrained=True):
        recorded["repo"] = repo
        recorded["name"] = name

        class Dummy:
            def eval(self):
                return self

            def to(self, device):
                recorded["device"] = device
                return self

        return Dummy()

    monkeypatch.setattr(dobj.torch.hub, "load", fake_load)
    dobj._load_model("yolox-s", device="cpu")

    assert recorded["name"] == "yolox_s"
    assert recorded["repo"] == "Megvii-BaseDetection/YOLOX"
    assert recorded["device"] == "cpu"


def test_detect_folder_writes_json(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img1.jpg").write_bytes(b"\x00")
    (frames / "img2.jpg").write_bytes(b"\x00")

    class FakeDet(list):
        def tolist(self):
            return list(self)

        def __getitem__(self, item):
            res = super().__getitem__(item)
            if isinstance(item, slice):
                return FakeDet(res)
            return res

    class FakeModel:
        def __call__(self, tensor):
            return [[FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]]

    monkeypatch.setattr(dobj, "_load_model", lambda *a, **k: FakeModel())

    class DummyTensor:
        def unsqueeze(self, dim):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(dobj, "_preprocess_image", lambda p: DummyTensor())

    out_json = tmp_path / "det.json"
    dobj.detect_folder(frames, out_json, "yolox-s")

    with out_json.open() as fh:
        data = json.load(fh)
    assert len(data) == 2
    assert data[0]["detections"][0]["class"] == 0
