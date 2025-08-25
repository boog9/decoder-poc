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
"""Tests for :mod:`src.court_detector` wrapper."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

import src.court_detector as cd


def test_detect_single_frame_state_dict(monkeypatch) -> None:
    """Wrapper loads state_dict weights and returns detection keys."""

    called = {}

    class DummyNet:
        def eval(self):
            return self

        def to(self, device: str):
            return self

        def load_state_dict(self, state, strict: bool = False):  # pragma: no cover - simple mock
            called["state"] = state
            return self

        def __call__(self, x):  # pragma: no cover - simple forward
            called["x"] = x
            return {
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "lines": {},
                "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "score": 0.9,
            }

    monkeypatch.setattr(cd, "_model", None, raising=False)
    monkeypatch.setattr(cd, "_model_device", None, raising=False)
    monkeypatch.setattr(cd, "_model_weights", None, raising=False)
    monkeypatch.setattr(cd, "_maybe_import_external_builder", lambda: lambda: DummyNet())
    import src.utils.checkpoint as ck
    monkeypatch.setattr(ck, "verify_torch_ckpt", lambda p: "state_dict")
    import torch

    monkeypatch.setattr(torch, "load", lambda *a, **k: {"w": 1}, raising=False)

    img = Image.new("RGB", (8, 8))
    out = cd.detect_single_frame(img, device="cpu", weights=Path("w.pth"), min_score=0.5)
    assert "polygon" in out and out["score"] >= 0.5
