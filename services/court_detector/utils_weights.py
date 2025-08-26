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
"""Helpers for loading Tennis Court Detector checkpoints."""

from __future__ import annotations

from typing import Any, Dict

import torch


def load_tcd_state_dict(path: str) -> Dict[str, Any]:
    """Load a TCD checkpoint into a clean ``state_dict``.

    The function normalizes common checkpoint layouts by unwrapping nested
    dictionaries and stripping ``module.`` prefixes produced by ``DataParallel``.

    Args:
        path: Path to ``.pth`` file.

    Returns:
        Normalized state dictionary suitable for :func:`torch.nn.Module.load_state_dict`.
    """

    sd: Any = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd


__all__ = ["load_tcd_state_dict"]
