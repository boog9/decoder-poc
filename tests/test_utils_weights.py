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
"""Tests for :mod:`services.court_detector.utils_weights`."""

import numpy as np
import pytest

if not hasattr(np, "ndarray"):
    pytest.skip("numpy not available", allow_module_level=True)

import torch
from services.court_detector.utils_weights import load_tcd_state_dict


if not hasattr(torch, "save"):
    pytest.skip("torch not available", allow_module_level=True)


def test_load_tcd_state_dict(tmp_path) -> None:
    """Loader should unwrap nested keys and strip prefixes."""

    ckpt = {"state_dict": {"module.conv.weight": 0}}
    path = tmp_path / "ckpt.pth"
    torch.save(ckpt, path)
    sd = load_tcd_state_dict(str(path))
    assert "conv.weight" in sd
