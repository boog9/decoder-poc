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
"""Load YOLOX experiment definitions in a robust manner."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from .default import *  # noqa: F401,F403


def get_exp_by_name(exp_name: str):
    """Load an :class:`Exp` given its name within ``yolox.exps.default``."""
    import yolox

    yolox_path = os.path.dirname(yolox.__file__)
    exp_path = os.path.join(yolox_path, "exps", "default", exp_name + ".py")
    return get_exp_by_file(exp_path)


def get_exp_by_file(exp_file: str):
    """Load an :class:`Exp` from a Python file path."""
    exp_path = Path(exp_file).resolve()
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {exp_file}")

    module_name = exp_path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(exp_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {exp_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "Exp"):
        raise ImportError(f"{exp_file} doesn't contain class named 'Exp'")

    return module.Exp()


def get_exp(exp_file: str | None = None, exp_name: str | None = None):
    """Compatibility wrapper mimicking YOLOX's ``get_exp``."""
    if exp_file:
        return get_exp_by_file(exp_file)
    if exp_name:
        return get_exp_by_name(exp_name)
    raise ValueError("Either exp_file or exp_name must be provided")
