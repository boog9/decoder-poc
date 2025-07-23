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
fake_tf = types.ModuleType("transformers")
fake_tf.AutoModelForImageSuperResolution = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
fake_tf.AutoImageProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **k: None)
sys.modules.setdefault("transformers", fake_tf)

import importlib
import src.frame_enhancer as fe


def test_parse_args_defaults() -> None:
    args = fe.parse_args(["--input-dir", "in", "--output-dir", "out"])
    assert isinstance(args, argparse.Namespace)
    assert args.batch_size == 4
    assert args.model_id == fe.DEFAULT_MODEL_ID
    assert args.fp16 is False


def test_parse_args_fp16_flag() -> None:
    args = fe.parse_args(["--input-dir", "in", "--output-dir", "out", "--fp16"])
    assert args.fp16 is True

def test_load_model_uses_transformers(monkeypatch):
    recorded = {}

    def fake_model_pretrained(model_id):
        recorded['model_id'] = model_id

        class Dummy:
            def eval(self):
                return self

            def to(self, device):
                recorded['device'] = device
                return self

            def half(self):
                recorded['half'] = True
                return self

        return Dummy()

    def fake_processor_pretrained(model_id):
        recorded['processor_id'] = model_id
        return 'processor'

    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace())

    importlib.reload(fe)
    monkeypatch.setattr(fe, 'AutoModelForImageSuperResolution',
                         types.SimpleNamespace(from_pretrained=fake_model_pretrained))
    monkeypatch.setattr(fe, 'AutoImageProcessor',
                         types.SimpleNamespace(from_pretrained=fake_processor_pretrained))

    model, processor = fe._load_model('cpu', 'repo/model', fp16=True)

    assert recorded['model_id'] == 'repo/model'
    assert recorded['processor_id'] == 'repo/model'
    assert recorded['device'] == 'cpu'
    assert processor == 'processor'
    assert recorded.get('half')


def test_load_model_aliases_swin2sr(monkeypatch):
    recorded = {}

    def fake_model_pretrained(model_id):
        recorded['model_id'] = model_id

        class Dummy:
            def eval(self):
                return self

            def to(self, device):
                recorded['device'] = device
                return self

            def half(self):
                recorded['half'] = True
                return self

        return Dummy()

    def fake_processor_pretrained(model_id):
        recorded['processor_id'] = model_id
        return 'processor'

    tf_mod = types.ModuleType('transformers')
    tf_mod.Swin2SRForImageSuperResolution = types.SimpleNamespace(
        from_pretrained=fake_model_pretrained
    )
    tf_mod.AutoImageProcessor = types.SimpleNamespace(
        from_pretrained=fake_processor_pretrained
    )

    monkeypatch.setitem(sys.modules, 'transformers', tf_mod)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace())

    import importlib
    importlib.reload(fe)

    model, processor = fe._load_model('cpu', 'repo/model', fp16=True)

    assert recorded['model_id'] == 'repo/model'
    assert recorded['processor_id'] == 'repo/model'
    assert recorded['device'] == 'cpu'
    assert processor == 'processor'
    assert recorded.get('half')


# legacy tests removed since model loading now uses Hugging Face Transformers
