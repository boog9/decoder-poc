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
"""Lightweight TennisCourtDetector model builder.

This module provides a minimal neural network architecture that is compatible
with state-dict checkpoints using ``convN.block.*`` naming. The implementation
is intentionally simple and serves as a placeholder for the production model.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Simple convolutional block used in :class:`TCDNet`."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.block(x)


class TCDNet(nn.Module):
    """Tiny convolutional network producing court geometry placeholders."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBlock(3, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 128)
        self.out_poly = nn.Linear(128, 8)
        self.out_h = nn.Linear(128, 9)
        self.out_score = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        poly = torch.sigmoid(self.out_poly(x)).view(-1, 4, 2)[0]
        H_raw = self.out_h(x).view(-1, 3, 3)[0]
        H = torch.eye(3, device=H_raw.device) + 0.05 * H_raw
        score = torch.sigmoid(self.out_score(x))[0, 0]
        poly = poly.tolist()
        H = H.tolist()
        score = float(score.item())
        return {"polygon": poly, "homography": H, "lines": {}, "score": score}

    # The default ``load_state_dict`` is retained; any mismatched keys will be
    # ignored when ``strict=False`` is used by the loader.


def build_tcd_model() -> nn.Module:
    """Create and return the TennisCourtDetector network.

    Returns
    -------
    nn.Module
        Uninitialized detector network.
    """

    return TCDNet()

