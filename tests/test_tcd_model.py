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
"""Tests for :mod:`services.court_detector.tcd_model`."""

import numpy as np
import pytest

if not hasattr(np, "ndarray"):
    pytest.skip("numpy not available", allow_module_level=True)

import torch
from services.court_detector.tcd_model import BallTrackerNet


if not hasattr(torch, "zeros"):
    pytest.skip("torch not available", allow_module_level=True)
try:  # ensure tensor creation works
    torch.zeros(1)
except Exception:  # pragma: no cover - environment limitation
    pytest.skip("incomplete torch implementation", allow_module_level=True)


def test_forward_shape() -> None:
    """Model should return 15 heatmaps of size 360x640."""

    model = BallTrackerNet()
    x = torch.zeros(1, 3, 360, 640)
    y = model(x)
    assert y.shape == (1, 15, 360, 640)
