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
"""Minimal TennisCourtDetector model builder.

This module exposes a tiny convolutional network compatible with
``convN.block.*``-style state-dict checkpoints. The implementation serves as a
placeholder; replace with the production architecture as needed.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Simple convolutional block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.block(x)


class TCDNet(nn.Module):
    """Very small network producing court placeholders."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels, base_channels * 2
        self.conv1 = ConvBlock(3, c1)
        self.conv2 = ConvBlock(c1, c2)
        self.conv3 = ConvBlock(c2, c3)
        # TODO: add remaining layers and heads matching the real checkpoint

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # TODO: head/decoder according to actual checkpoint
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        return {
            "polygon": polygon,
            "lines": {},
            "homography": None,
            "score": 1.0,
        }


def build_tcd_model(base_channels: int = 64) -> TCDNet:
    """Construct a ``TCDNet`` instance.

    Args:
        base_channels: Width multiplier for the first conv layer.

    Returns:
        Uninitialized ``TCDNet`` model.
    """

    return TCDNet(base_channels=base_channels)
