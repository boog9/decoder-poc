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
loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = types.SimpleNamespace(
    info=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    error=lambda *a, **k: None,
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
# dummy ByteTrack module for dynamic import
bt_mod = types.ModuleType("yolox.tracker.byte_tracker")

class _DummyBT:
    def __init__(self, *a, **k):
        pass

setattr(bt_mod, "BYTETracker", _DummyBT)
sys.modules.setdefault("yolox", types.ModuleType("yolox"))
sys.modules.setdefault("yolox.tracker", types.ModuleType("yolox.tracker"))
sys.modules["yolox.tracker.byte_tracker"] = bt_mod

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
    assert args.classes is None


def test_parse_args_custom_classes() -> None:
    args = dobj.parse_args(
        [
            "--frames-dir",
            "frames",
            "--output-json",
            "out.json",
            "--classes",
            "1",
            "2",
        ]
    )
    assert args.classes == [1, 2]


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


def test_track_detections_assigns_ids(tmp_path: Path, monkeypatch) -> None:
    class DummyObj:
        def __init__(self, tid: int, tlwh: list[float], score: float) -> None:
            self.track_id = tid
            self.tlwh = tlwh
            self.score = score

    class DummyTracker:
        def __init__(self, *a, **k) -> None:
            self.last: dict[tuple[float, float, float, float], int] = {}
            self.next_id = 1

        def update(self, tlwhs, scores, classes, frame_id):
            out = []
            for tlwh, score in zip(tlwhs, scores):
                key = tuple(tlwh)
                tid = self.last.get(key)
                if tid is None:
                    tid = self.next_id
                    self.next_id += 1
                    self.last[key] = tid
                out.append(DummyObj(tid, tlwh, score))
            return out

    monkeypatch.setattr(dobj, "BYTETracker", DummyTracker)

    det_json = tmp_path / "det.json"
    det_json.write_text(
        json.dumps(
            [
                {"frame": "f1.png", "detections": [{"bbox": [0, 0, 2, 2], "score": 0.9, "class": 0}]},
                {"frame": "f2.png", "detections": [{"bbox": [0, 0, 2, 2], "score": 0.9, "class": 0}]},
            ]
        )
    )
    out_json = tmp_path / "out.json"
    dobj.track_detections(det_json, out_json, 0.3)

    with out_json.open() as fh:
        out = json.load(fh)

    assert len(out) == 2
    assert out[0]["track_id"] == out[1]["track_id"]


def test_update_tracker_mot_two_params() -> None:
    class DummyTracker:
        def __init__(self) -> None:
            self.args = None

        def update(self, img_info, img_size):
            self.args = (img_info, img_size)
            return ["ok"]

    tracker = DummyTracker()
    res = dobj._update_tracker(
        tracker,
        [[0, 0, 10, 20]],
        [0.9],
        ["person"],
        1,
    )

    assert res == ["ok"]
    assert tracker.args == ((20, 10, 1.0), (10, 20))


def test_update_tracker_mot_three_params() -> None:
    class DummyTracker:
        def __init__(self) -> None:
            self.args = None

        def update(self, outputs, img_info, img_size):
            self.args = (outputs, img_info, img_size)
            return ["ok"]

    tracker = DummyTracker()
    res = dobj._update_tracker(
        tracker,
        [[0, 0, 10, 20]],
        [0.9],
        ["person"],
        1,
    )

    assert res == ["ok"]
    assert tracker.args[1:] == ((20, 10, 1.0), (10, 20))
    assert tracker.args[0][0][:4] == [0, 0, 10, 20]




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


@pytest.mark.parametrize(
    "rows",
    [
        [[0, 0, 1, 1, 0.9, 0]],
        [[0, 0, 1, 1, 0.9, 0.8, 0]],
    ],
)
def test_filter_cpu(rows) -> None:
    assert dobj._filter_detections(rows, 0.5, [0])


def test_detect_folder_respects_classes(monkeypatch, tmp_path: Path) -> None:
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
            return [[FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 42])]]

    monkeypatch.setattr(dobj, "_load_model", lambda *a, **k: FakeModel())

    module = types.ModuleType("yolox")
    utils_mod = types.ModuleType("utils")
    utils_mod.postprocess = (
        lambda outputs, num_classes, conf_thre, nms_thre, class_agnostic=False: [
            FakeDet([0.0, 0.0, 1.0, 1.0, 0.9, 42])
        ]
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
    dobj.detect_folder(frames, out_json, "yolox-s", 640, class_ids=[42])

    with out_json.open() as fh:
        data = json.load(fh)

    assert data[0]["detections"][0]["class"] == 42

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)


def test_detect_folder_seven_element(monkeypatch, tmp_path: Path) -> None:
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
            return [[FakeDet([0.0, 0.0, 1.0, 1.0, 0.8, 0.9, 1])]]

    monkeypatch.setattr(dobj, "_load_model", lambda *a, **k: FakeModel())

    module = types.ModuleType("yolox")
    utils_mod = types.ModuleType("utils")
    utils_mod.postprocess = (
        lambda outputs, num_classes, conf_thre, nms_thre, class_agnostic=False: [
            FakeDet([0.0, 0.0, 1.0, 1.0, 0.8, 0.9, 1])
        ]
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

    assert data[0]["detections"][0]["class"] == 1

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)


