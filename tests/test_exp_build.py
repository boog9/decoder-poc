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
"""Tests for ``bytetrack_vendor.exp.build`` helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "externals" / "ByteTrack"))

from bytetrack_vendor.exp import get_exp_by_file
import pytest


def test_get_exp_by_file_loads_custom_exp(tmp_path: Path) -> None:
    exp_file = tmp_path / "custom_exp.py"
    exp_file.write_text(
        "\n".join(
            [
                "class Exp:",
                "    def __init__(self):",
                "        self.name = 'custom'",
                "    def get_model(self):",
                "        return 'model'",
            ]
        )
    )

    exp = get_exp_by_file(str(exp_file))
    assert exp.name == "custom"
    assert exp.get_model() == "model"


def test_get_exp_by_file_missing_file(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.py"
    with pytest.raises(FileNotFoundError):
        get_exp_by_file(str(missing))
    assert "not found" in capsys.readouterr().out.lower()


def test_get_exp_by_file_missing_class(tmp_path: Path, capsys) -> None:
    exp_file = tmp_path / "noexp.py"
    exp_file.write_text("print('hi')\n")
    with pytest.raises(ImportError):
        get_exp_by_file(str(exp_file))
    out = capsys.readouterr().out.lower()
    assert "does not define" in out or "doesn't contain" in out
