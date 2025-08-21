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
yolox_mod = sys.modules.setdefault("yolox", types.ModuleType("yolox"))
setattr(yolox_mod, "__version__", "0.0")

import src.detect_objects as dobj

dobj.torch.cuda.is_available = lambda: True


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
    assert args.model == "yolox-x"
    assert args.img_size == 640
    assert args.classes is None
    assert args.two_pass is True
    assert args.person_conf == 0.55
    assert args.person_img_size == 1280
    assert args.ball_conf == 0.10
    assert args.ball_img_size == 1280
    assert args.ball_interp_gap_max == 5
    assert args.save_splits is False


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


def test_merge_detections_merges() -> None:
    a = [{"frame": "f1", "detections": [1]}, {"frame": "f2", "detections": [2]}]
    b = [{"frame": "f1", "detections": [3]}, {"frame": "f2", "detections": [4]}]
    merged = dobj.merge_detections(a, b)
    assert merged[0]["detections"] == [1, 3]
    assert merged[1]["detections"] == [2, 4]


def test_allowed_class_alias_ball(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_filter(rows, conf, class_ids):
        captured["ids"] = class_ids
        return []

    monkeypatch.setattr(dobj, "_filter_detections", fake_filter)
    monkeypatch.setattr(
        dobj,
        "_preprocess_image",
        lambda f, s: (dobj.torch.zeros(1, 3, 10, 10), 1.0, 0.0, 0.0, 10, 10),
    )

    class DummyModel:
        def __call__(self, tensor):
            return [dobj.torch.zeros((1, 6))]

        head = types.SimpleNamespace(decode_outputs=lambda out, dtype: out)

    utils_mod = types.ModuleType("yolox.utils")
    utils_mod.postprocess = lambda *a, **k: [None]
    sys.modules["yolox.utils"] = utils_mod

    frame = tmp_path / "f.jpg"
    frame.touch()
    dobj.run_infer(DummyModel(), [frame], 640, 0.5, 0.5, {"ball"})
    assert captured["ids"] == [32]


def test_class_id_alias_ball() -> None:
    """Ensure ``_class_id_from_name`` resolves ball alias."""

    assert dobj._class_id_from_name("ball") == 32


def test_load_model_uses_local_yolox(monkeypatch) -> None:
    recorded = {}

    def fake_get_exp(exp_file: str | None = None, exp_name: str | None = None):
        recorded["exp_file"] = exp_file
        recorded["exp_name"] = exp_name

        class DummyExp:
            def get_model(self):
                class DummyModel:
                    def load_state_dict(self, state):
                        recorded["state"] = state

                    def eval(self):
                        recorded["eval"] = True
                        return self

                    def to(self, device):
                        recorded["device"] = str(device)
                        return self

                return DummyModel()

        return DummyExp()

    def fake_torch_load(path, map_location=None):
        recorded["ckpt"] = str(path)
        return {"model": {}}

    def fake_fuse(model):
        recorded["fused"] = True
        return model

    monkeypatch.setattr(dobj.torch.cuda, "is_available", lambda: True)
    exp_mod = types.ModuleType("yolox.exp")
    exp_mod.get_exp = fake_get_exp
    utils_mod = types.ModuleType("yolox.utils")
    utils_mod.fuse_model = fake_fuse
    sys.modules["yolox.exp"] = exp_mod
    sys.modules["yolox.utils"] = utils_mod
    monkeypatch.setattr(dobj.torch, "load", fake_torch_load)

    dobj._load_model("yolox-s")

    assert recorded["exp_name"] == "yolox_s"
    assert recorded["ckpt"].endswith("weights/yolox_s.pth")
    assert recorded.get("fused")
    assert recorded.get("device") == "cuda"




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
    dobj.detect_folder(frames, out_json, "yolox-s", 640, detect_court=False)

    with out_json.open() as fh:
        data = json.load(fh)
    assert len(data) == 2
    det = data[0]["detections"][0]
    assert det["class_id"] == 0
    assert det["category"] == "person"
    bbox = det["bbox"]
    assert all(isinstance(v, int) for v in bbox)

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)


def test_detect_two_pass_merges(tmp_path: Path, monkeypatch) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "img1.jpg").write_bytes(b"0")
    (frames / "img2.jpg").write_bytes(b"0")

    calls: list = []

    def fake_run_infer(model, frames_seq, img_size, conf, nms, allowed_classes):
        calls.append((img_size, conf, nms, allowed_classes))
        cid = 0 if "person" in allowed_classes else 32
        out = []
        for f in frames_seq:
            out.append(
                {
                    "frame": f.name,
                    "detections": [
                        {
                            "bbox": [0, 0, 1, 1],
                            "score": 1.0,
                            "class": cid,
                            "class_id": cid,
                            "category": dobj.CLASS_ID_TO_NAME[cid],
                        }
                    ],
                }
            )
        return out

    monkeypatch.setattr(dobj, "run_infer", fake_run_infer)
    monkeypatch.setattr(dobj, "_load_model", lambda *a, **k: object())

    out_json = tmp_path / "out.json"
    dobj.detect_two_pass(
        frames,
        out_json,
        "yolox-s",
        0.5,
        0.45,
        960,
        ["person"],
        0.2,
        0.3,
        1280,
        ["sports ball"],
        save_splits=True,
        detect_court=False,
    )

    with out_json.open() as fh:
        merged = json.load(fh)
    byf = {d["frame"]: d["detections"] for d in merged}
    assert "img1.jpg" in byf and len(byf["img1.jpg"]) == 2
    assert calls[0][0] == 960 and calls[1][0] == 1280
    assert (tmp_path / "out_person.json").exists()
    assert (tmp_path / "out_ball.json").exists()


def test_track_detections_assigns_ids(tmp_path: Path, monkeypatch) -> None:
    class DummyObj:
        def __init__(self, tid: int, tlwh: list[float], score: float) -> None:
            self.track_id = tid
            self.tlwh = tlwh
            self._tlwh = tlwh
            self.score = score
            self.cls = 0

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
    monkeypatch.setattr(dobj, "_bbox_iou", lambda a, b: 1.0)

    det_json = tmp_path / "det.json"
    det_json.write_text(
        json.dumps(
            [
                {"frame": 1, "class": 0, "bbox": [0, 0, 2, 2], "score": 0.9},
                {"frame": 2, "class": 0, "bbox": [0, 0, 2, 2], "score": 0.9},
            ]
        )
    )
    out_json = tmp_path / "out.json"
    dobj.track_detections(det_json, out_json, 0.3)

    with out_json.open() as fh:
        out = json.load(fh)

    assert len(out) == 2
    assert out[0]["track_id"] == out[1]["track_id"]


def test_track_detections_iou_matching(tmp_path: Path, monkeypatch) -> None:
    class DummyObj:
        def __init__(self, tid: int, tlwh: list[float], score: float) -> None:
            self.track_id = tid
            self.tlwh = tlwh
            self._tlwh = tlwh
            self.score = score

    class DummyTracker:
        def __init__(self, *a, **k) -> None:
            pass

        def update(self, tlwhs, scores, classes, frame_id):
            return [DummyObj(5, tlwhs[1], scores[1])]

    monkeypatch.setattr(dobj, "BYTETracker", DummyTracker)
    def _simple_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union else 0.0
    monkeypatch.setattr(dobj, "_bbox_iou", _simple_iou)

    det_json = tmp_path / "det.json"
    det_json.write_text(
        json.dumps(
            [
                {"frame": 1, "class": 0, "bbox": [0, 0, 2, 2], "score": 0.9},
                {"frame": 1, "class": 0, "bbox": [10, 10, 12, 12], "score": 0.8},
            ]
        )
    )
    out_json = tmp_path / "out.json"
    dobj.track_detections(det_json, out_json, 0.3)

    assert dobj._det_index[(1, 1)]["track_id"] == 5
    assert "track_id" not in dobj._det_index[(1, 0)]


def test_track_detections_string_frame(tmp_path: Path, monkeypatch) -> None:
    class DummyObj:
        def __init__(self, tid: int, tlwh: list[float], score: float) -> None:
            self.track_id = tid
            self.tlwh = tlwh
            self._tlwh = tlwh
            self.score = score
            self.cls = 0

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
    monkeypatch.setattr(dobj, "_bbox_iou", lambda a, b: 1.0)

    det_json = tmp_path / "det.json"
    det_json.write_text(
        json.dumps(
            [
                {"frame": "frame_000001.png", "class": 0, "bbox": [0, 0, 2, 2], "score": 0.9},
                {"frame": "frame_000002.png", "class": 0, "bbox": [0, 0, 2, 2], "score": 0.9},
            ]
        )
    )
    out_json = tmp_path / "out.json"
    dobj.track_detections(det_json, out_json, 0.3)

    with out_json.open() as fh:
        out = json.load(fh)

    assert len(out) == 2
    assert out[0]["frame"] == 1 and out[1]["frame"] == 2


def test_track_detections_nested_format(tmp_path: Path, monkeypatch) -> None:
    class DummyObj:
        def __init__(self, tid: int, tlwh: list[float], score: float) -> None:
            self.track_id = tid
            self.tlwh = tlwh
            self._tlwh = tlwh
            self.score = score
            self.cls = 32

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
    monkeypatch.setattr(dobj, "_bbox_iou", lambda a, b: 1.0)

    det_json = tmp_path / "det.json"
    det_json.write_text(
        json.dumps(
            [
                {
                    "frame": "frame_000001.png",
                    "detections": [
                        {"class": 32, "bbox": [0, 0, 2, 2], "score": 0.9}
                    ],
                },
                {
                    "frame": "frame_000002.png",
                    "detections": [
                        {
                            "class": "sports ball",
                            "bbox": [0, 0, 2, 2],
                            "score": 0.9,
                        }
                    ],
                },
            ]
        )
    )
    out_json = tmp_path / "out.json"
    dobj.track_detections(det_json, out_json, 0.3)

    with out_json.open() as fh:
        out = json.load(fh)

    assert len(out) == 2
    assert out[0]["frame"] == 1 and out[1]["frame"] == 2
    assert out[0]["class"] == out[1]["class"] == 32
    assert out[0]["track_id"] == out[1]["track_id"]


def test_track_detections_skips_invalid_items(tmp_path: Path, monkeypatch) -> None:
    class DummyObj:
        def __init__(self, tid: int, tlwh: list[float], score: float) -> None:
            self.track_id = tid
            self.tlwh = tlwh
            self._tlwh = tlwh
            self.score = score
            self.cls = 0

    class DummyTracker:
        def __init__(self, *a, **k) -> None:
            self.next_id = 1

        def update(self, tlwhs, scores, classes, frame_id):
            out = []
            for tlwh, score in zip(tlwhs, scores):
                out.append(DummyObj(self.next_id, tlwh, score))
                self.next_id += 1
            return out

    monkeypatch.setattr(dobj, "BYTETracker", DummyTracker)
    monkeypatch.setattr(dobj, "_bbox_iou", lambda a, b: 1.0)

    det_json = tmp_path / "det_nested.json"
    det_json.write_text(
        json.dumps(
            [
                {
                    "frame": "frame_000001.png",
                    "detections": [
                        {"bbox": [0, 0, 10], "score": 0.9, "class": 0},
                        {"bbox": [0, 0, 2, 2], "score": 0.9},
                        {"bbox": [0, 0, 2, 2], "score": 0.9, "class": 0},
                    ],
                }
            ]
        )
    )
    out_json = tmp_path / "out.json"
    dobj.track_detections(det_json, out_json, 0.3)

    with out_json.open() as fh:
        out = json.load(fh)

    assert len(out) == 1
    assert out[0]["frame"] == 1
    assert out[0]["class"] == 0
    assert out[0]["track_id"] == 1


def test_track_detections_flat_skips_invalid_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyObj:
        def __init__(self, tid: int, tlwh: list[float], score: float) -> None:
            self.track_id = tid
            self.tlwh = tlwh
            self._tlwh = tlwh
            self.score = score
            self.cls = 0

    class DummyTracker:
        def __init__(self, *a, **k) -> None:
            self.next_id = 1

        def update(self, tlwhs, scores, classes, frame_id):
            out = []
            for tlwh, score in zip(tlwhs, scores):
                out.append(DummyObj(self.next_id, tlwh, score))
                self.next_id += 1
            return out

    monkeypatch.setattr(dobj, "BYTETracker", DummyTracker)
    monkeypatch.setattr(dobj, "_bbox_iou", lambda a, b: 1.0)

    det_json = tmp_path / "det_flat.json"
    det_json.write_text(
        json.dumps(
            [
                {
                    "frame": "frame_000001.png",
                    "class": 0,
                    "bbox": [0, 0, 10],
                    "score": 0.9,
                },
                {
                    "frame": "frame_000001.png",
                    "class": "banana",
                    "bbox": [0, 0, 2, 2],
                    "score": 0.9,
                },
                {
                    "frame": "frame_000001.png",
                    "class": 0,
                    "bbox": [0, 0, 2, 2],
                    "score": 0.9,
                },
            ]
        )
    )
    out_json = tmp_path / "out.json"
    dobj.track_detections(det_json, out_json, 0.3)

    with out_json.open() as fh:
        out = json.load(fh)

    assert len(out) == 1
    assert out[0]["frame"] == 1
    assert out[0]["class"] == 0
    assert out[0]["track_id"] == 1


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
        [0],
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
        [0],
        1,
    )

    assert res == ["ok"]
    assert tracker.args[1:] == ((20, 10, 1.0), (10, 20))
    assert tracker.args[0][0][:4] == [0, 0, 10, 20]


def test_update_tracker_output_results_signature() -> None:
    class DummyTracker:
        def __init__(self) -> None:
            self.args = None

        def update(self, output_results, img_info, img_size):
            self.args = (output_results, img_info, img_size)
            return ["ok"]

    tracker = DummyTracker()
    res = dobj._update_tracker(
        tracker,
        [[0, 0, 10, 20]],
        [0.9],
        [0],
        1,
    )

    assert res == ["ok"]
    assert tracker.args[1:] == ((20, 10, 1.0), (10, 20))
    assert tracker.args[0][0][:4] == [0, 0, 10, 20]


def test_update_tracker_mot_two_params_dets_img_info() -> None:
    class DummyTracker:
        def __init__(self) -> None:
            self.args = None

        def update(self, dets, img_info):
            self.args = (dets, img_info)
            return ["ok"]

    tracker = DummyTracker()
    res = dobj._update_tracker(
        tracker,
        [[0, 0, 10, 20]],
        [0.9],
        [0],
        1,
    )

    assert res == ["ok"]
    assert tracker.args[1] == (20, 10, 1.0)
    assert tracker.args[0][0][:4] == [0, 0, 10, 20]


def test_update_tracker_mot_two_params_dets_img_size() -> None:
    class DummyTracker:
        def __init__(self) -> None:
            self.args = None

        def update(self, dets, img_size):
            self.args = (dets, img_size)
            return ["ok"]

    tracker = DummyTracker()
    res = dobj._update_tracker(
        tracker,
        [[0, 0, 10, 20]],
        [0.9],
        [0],
        1,
    )

    assert res == ["ok"]
    assert tracker.args[1] == (10, 20)
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
    dobj.detect_folder(frames, out_json, "yolox-s", 640, detect_court=False)

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
    dobj.detect_folder(frames, out_json, "yolox-s", 640, detect_court=False)

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
    dobj.detect_folder(frames, out_json, "yolox-s", 640, class_ids=[42], detect_court=False)

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
    dobj.detect_folder(frames, out_json, "yolox-s", 640, detect_court=False)

    with out_json.open() as fh:
        data = json.load(fh)

    assert data[0]["detections"][0]["class"] == 1

    sys.modules.pop("yolox.utils", None)
    sys.modules.pop("yolox", None)


def test_pre_court_gate_info_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    det_json = tmp_path / "d.json"
    det_json.write_text("[]")
    out_json = tmp_path / "o.json"

    track_mod = types.ModuleType("src.track_objects")
    track_mod.track_detections = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "src.track_objects", track_mod)

    dobj.main(
        [
            "track",
            "--detections-json",
            str(det_json),
            "--output-json",
            str(out_json),
            "--pre-court-gate",
        ]
    )
    assert "pre-court-gate activates" in caplog.text


