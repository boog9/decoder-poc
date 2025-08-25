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
"""Tests for preprocess utility in ``services.court_detector.tcd``."""

from __future__ import annotations

import numpy as np

from services.court_detector.tcd import preprocess_to_640x360


def test_preprocess_shape_and_range() -> None:
    """Input image is resized and normalized correctly."""

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    tensor = preprocess_to_640x360(img)
    assert tensor.shape == (1, 3, 360, 640)
    assert tensor.dtype.name == "float32"
    assert tensor.min().item() == 0.0 and tensor.max().item() == 0.0
