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
import pytest

pytest.importorskip("PIL")
pytest.importorskip("torch.cuda")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    args = dobj.parse_args(
        [
            "--frames-dir",
            "frames",
            "--output-json",
            "out.json",
        ]
    )
    assert isinstance(args, argparse.Namespace)
    assert args.model == "yolox-s"
    assert args.img_size == 640


def test_load_model_translates_hyphen(monkeypatch) -> None:
    recorded = {}

    def fake_load(repo, name, pretrained=True):
        recorded["repo"] = repo
        recorded["name"] = name

        class Dummy:
            head = object()

            def eval(self):
                return self

            def cuda(self):
                recorded["cuda"] = True
                return self

        return Dummy()

    monkeypatch.setattr(dobj.torch.hub, "load", fake_load)
    dobj._load_model("yolox-s")

    assert recorded["name"] == "yolox_s"
    assert recorded["repo"] == "Megvii-BaseDetection/YOLOX"
    assert recorded.get("cuda")




def test_detect_folder_writes_json(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img1.jpg").write_bytes(b"\x00")
    (frames / "img2.jpg").write_bytes(b"\x00")

    class FakeDet(list):
        dtype = "float32"

        def tolist(self):
            return [list(self)]

        def cpu(self):
            return self

        def __getitem__(self, item):
            res = super().__getitem__(item)
            if isinstance(item, slice):
                return FakeDet(res)
            return res

    class FakeModel:
        def __call__(self, tensor):
            return [[FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]]

        head = types.SimpleNamespace(
            decode_outputs=lambda out, dtype: out
        )

    monkeypatch.setattr(dobj, "_load_model", lambda *a, **k: FakeModel())

    module = types.ModuleType("yolox")
    utils_mod = types.ModuleType("utils")

    def fake_postprocess(outputs, num_classes, conf_thre, nms_thre, class_agnostic=False):
        return [FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]

    utils_mod.postprocess = fake_postprocess
    module.utils = utils_mod
    sys.modules["yolox"] = module
    sys.modules["yolox.utils"] = utils_mod

    class DummyTensor:
        def unsqueeze(self, dim):
            return self

        def cuda(self):
            return self

    monkeypatch.setattr(
        dobj,
        "_preprocess_image",
        lambda p, s: (DummyTensor(), 1.0, 0, 0, 10, 10),
    )

    out_json = tmp_path / "det.json"
    dobj.detect_folder(frames, out_json, "yolox-s", 640)

    with out_json.open() as fh:
        data = json.load(fh)
    assert len(data) == 2
    assert data[0]["detections"][0]["class"] == 0
    bbox = data[0]["detections"][0]["bbox"]
    assert all(isinstance(v, int) for v in bbox)

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)




def test_detect_folder_uses_decode(monkeypatch, tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img.jpg").write_bytes(b"\x00")

    class FakeDet(list):
        dtype = "float32"
        def tolist(self):
            return [list(self)]

        def cpu(self):
            return self

    class FakeHead:
        def __init__(self) -> None:
            self.called = False

        def decode_outputs(self, out, dtype):
            self.called = True
            return out

    head = FakeHead()

    class FakeModel:
        def __init__(self) -> None:
            self.head = head

        def __call__(self, tensor):
            return [[FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]]

    monkeypatch.setattr(dobj, "_load_model", lambda *a, **k: FakeModel())

    module = types.ModuleType("yolox")
    utils_mod = types.ModuleType("utils")

    def fake_postprocess(outputs, num_classes, conf_thre, nms_thre, class_agnostic=False):
        return [FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]

    utils_mod.postprocess = fake_postprocess
    module.utils = utils_mod
    sys.modules["yolox"] = module
    sys.modules["yolox.utils"] = utils_mod

    class DummyTensor:
        def unsqueeze(self, dim):
            return self

        def cuda(self):
            return self

    monkeypatch.setattr(
        dobj,
        "_preprocess_image",
        lambda p, s: (DummyTensor(), 1.0, 0, 0, 10, 10),
    )

    out_json = tmp_path / "det.json"
    dobj.detect_folder(frames, out_json, "yolox-s", 640)

    assert head.called
    with out_json.open() as fh:
        data = json.load(fh)
    assert data and data[0]["detections"]

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)


def test_detect_folder_single_frame(monkeypatch, tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img.jpg").write_bytes(b"\x00")

    class FakeDet(list):
        dtype = "float32"

        def tolist(self):
            return [list(self)]

        def cpu(self):
            return self

    class FakeModel:
        head = types.SimpleNamespace(decode_outputs=lambda o, dtype: o)

        def __call__(self, tensor):
            return [[FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]]

    monkeypatch.setattr(dobj, "_load_model", lambda *a, **k: FakeModel())

    module = types.ModuleType("yolox")
    utils_mod = types.ModuleType("utils")
    utils_mod.postprocess = (
        lambda outputs, num_classes, conf_thre, nms_thre, class_agnostic=False: [FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 0])]
    )
    module.utils = utils_mod
    sys.modules["yolox"] = module
    sys.modules["yolox.utils"] = utils_mod

    class DummyTensor:
        def unsqueeze(self, dim):
            return self

        def cuda(self):
            return self

    monkeypatch.setattr(
        dobj,
        "_preprocess_image",
        lambda p, s: (DummyTensor(), 1.0, 0, 0, 10, 10),
    )

    out_json = tmp_path / "det.json"
    dobj.detect_folder(frames, out_json, "yolox-s", 640)

    with out_json.open() as fh:
        data = json.load(fh)

    assert len(data) == 1
    assert data[0]["detections"]

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)


@pytest.mark.parametrize("rows", [[[0, 0, 1, 1, 0.9, 0]]])
def test_filter_cpu(rows) -> None:
    assert dobj._filter_detections(rows, 0.5)


