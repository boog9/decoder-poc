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
"""Kalman filter for constant-velocity ball tracking."""
from __future__ import annotations

import numpy as np


class BallKalmanFilter:
    """Simple constant-velocity Kalman filter for a 2D point."""

    def __init__(self, dt: float, process_var: float = 1.0, meas_var: float = 1.0) -> None:
        """Initialise filter parameters.

        Args:
            dt: Time step in seconds.
            process_var: Process noise variance.
            meas_var: Measurement noise variance.
        """
        self.dt = dt
        self.F = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        self.Q = process_var * np.eye(4)
        self.R = meas_var * np.eye(2)
        self.P = np.eye(4)
        self.x_state = np.zeros((4, 1), dtype=float)
        self.initialised = False
        self.last_pos: tuple[float, float] | None = None
        self.last_vel: tuple[float, float] | None = None

    def init(self, x: float, y: float) -> None:
        """Initialise state explicitly."""
        self.x_state[:2, 0] = [x, y]
        self.x_state[2:, 0] = 0.0
        self.initialised = True

    def predict(self) -> np.ndarray:
        """Predict next state."""
        if not self.initialised:
            return self.x_state
        self.x_state = self.F @ self.x_state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x_state

    def update(self, meas: tuple[float, float]) -> np.ndarray:
        """Update state with a measurement."""
        z = np.array([[meas[0]], [meas[1]]], dtype=float)
        if not self.initialised:
            self.init(meas[0], meas[1])
        y = z - (self.H @ self.x_state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_state = self.x_state + K @ y
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        if self.last_pos is not None:
            vx = (meas[0] - self.last_pos[0]) / self.dt
            vy = (meas[1] - self.last_pos[1]) / self.dt
            self.last_vel = (vx, vy)
        self.last_pos = meas
        return self.x_state
