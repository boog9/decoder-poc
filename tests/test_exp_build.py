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
"""Tests for ``yolox.exp.build`` helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yolox.exp import get_exp_by_file


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
