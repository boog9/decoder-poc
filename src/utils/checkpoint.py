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
"""Utilities for verifying model checkpoints."""

from __future__ import annotations

import os

import torch


def verify_torch_ckpt(path: str, min_bytes: int = 1024) -> str:
    """Validate checkpoint and return its type.

    The function first tries to interpret ``path`` as a standard ``state_dict``
    via :func:`torch.load`. If that fails, it attempts to load the file as a
    TorchScript module via :func:`torch.jit.load`.

    Args:
        path: Filesystem path to the checkpoint file.
        min_bytes: Minimum expected size of the file in bytes.

    Returns:
        ``"state_dict"`` if ``torch.load`` succeeds, otherwise ``"jit"`` when
        :func:`torch.jit.load` succeeds.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is smaller than ``min_bytes`` bytes.
        RuntimeError: If neither loading method succeeds.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    size = os.path.getsize(path)
    if size < min_bytes:
        raise ValueError(f"weights too small: {path} ({size} bytes)")

    try:
        torch.load(path, map_location="cpu")
        return "state_dict"
    except Exception as e_load:
        pass

    try:
        torch.jit.load(path, map_location="cpu")
        return "jit"
    except Exception as e_jit:  # pragma: no cover - executed only when load fails
        raise RuntimeError(
            f"Failed to load {path}: torch.load={e_load!r}; jit={e_jit!r}"
        )

