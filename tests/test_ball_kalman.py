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
"""Tests for :mod:`src.ball_kalman`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ensure real NumPy is used even if other tests stub it.
sys.modules.pop("numpy", None)

from src.ball_kalman import BallKalmanFilter


def test_kalman_predict_update() -> None:
    dt = 1 / 30.0
    kf = BallKalmanFilter(dt)
    kf.update((0.0, 0.0))
    for i in range(1, 5):
        kf.predict()
        kf.update((float(i), 0.0))
    state = kf.predict()
    assert state.shape == (4, 1)
    assert state[0, 0] > 0.0
