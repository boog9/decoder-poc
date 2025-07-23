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
"""Tests for :mod:`src.frame_enhancer`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("PIL", types.ModuleType("PIL")).Image = object()
sys.modules.setdefault("tqdm", types.ModuleType("tqdm")).tqdm = lambda *a, **k: a[0] if a else None

import src.frame_enhancer as fe


def test_parse_args_defaults() -> None:
    args = fe.parse_args(["--input-dir", "in", "--output-dir", "out"])
    assert isinstance(args, argparse.Namespace)
    assert args.batch_size == 4

