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
"""Tests for CLI argument handling in detect_objects."""

from __future__ import annotations

from pathlib import Path
import sys
import types
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Provide lightweight stubs for optional dependencies
try:  # pragma: no cover - optional import
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    sys.modules["torch"] = torch
    sys.modules.setdefault("torch.cuda", ModuleType("torch.cuda"))

try:  # pragma: no cover - optional import
    import numpy  # type: ignore
except Exception:  # pragma: no cover
    sys.modules["numpy"] = ModuleType("numpy")

from src.detect_objects import parse_args  # noqa: E402


def _base_args() -> list[str]:
    """Return required base CLI arguments."""

    return ["detect", "--frames-dir", "f", "--output-json", "o"]


def test_per_class_conf_args() -> None:
    """Verify that per-class confidence flags parse correctly."""

    ns = parse_args(_base_args() + ["--p-conf", "0.7", "--b-conf", "0.2"])
    assert ns.p_conf == 0.7
    assert ns.b_conf == 0.2


def test_conf_fallback() -> None:
    """Ensure per-class confidences fall back to global threshold."""

    ns = parse_args(_base_args() + ["--conf-thres", "0.4"])
    assert ns.p_conf == 0.4
    assert ns.b_conf == 0.4


def test_aliases_still_work() -> None:
    """Backward-compatible aliases must map to new args."""
    ns = parse_args(
        _base_args()
        + [
            "--person-conf",
            "0.77",
            "--conf-ball",
            "0.11",
            "--person-nms",
            "0.55",
            "--ball-nms",
            "0.66",
        ]
    )
    assert ns.p_conf == 0.77
    assert ns.b_conf == 0.11
    assert ns.p_nms == 0.55
    assert ns.b_nms == 0.66


def test_nms_fallback() -> None:
    """Per-class NMS thresholds should inherit from --nms-thres if unset."""
    ns = parse_args(_base_args() + ["--nms-thres", "0.42"])
    assert ns.p_nms == 0.42
    assert ns.b_nms == 0.42
