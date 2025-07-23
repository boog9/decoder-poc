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

import importlib
import src.frame_enhancer as fe


def test_parse_args_defaults() -> None:
    args = fe.parse_args(["--input-dir", "in", "--output-dir", "out"])
    assert isinstance(args, argparse.Namespace)
    assert args.batch_size == 4
    assert args.model_id == fe.DEFAULT_MODEL_ID

def test_load_model_uses_correct_name(monkeypatch, tmp_path):
    recorded = {}

    cfg = {"architecture": "swin"}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"architecture": "swin"}')

    def fake_hf_download(repo, filename):
        return str(cfg_path)

    def fake_create_model(name, pretrained=True, pretrained_cfg=None, scale=None):
        recorded['name'] = name
        recorded['scale'] = scale
        
        class Dummy:
            def eval(self):
                return self

            def to(self, device):
                return self

        return Dummy()

    monkeypatch.setitem(sys.modules, 'timm', types.SimpleNamespace(create_model=fake_create_model))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=fake_hf_download))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace())
    importlib.reload(fe)
    fe._load_model('cpu', fe.DEFAULT_MODEL_ID)
    assert recorded['name'] == 'swin'
    assert recorded['scale'] == 4


def test_load_model_with_missing_architecture(monkeypatch, tmp_path):
    recorded = {}

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"arch": "dummy_arch"}')

    def fake_hf_download(repo, filename):
        assert filename == "config.json"
        return str(cfg_path)

    def fake_create_model(name, pretrained=True, pretrained_cfg=None, scale=None):
        recorded['name'] = name
        recorded['scale'] = scale
        recorded['cfg'] = pretrained_cfg

        class Dummy:
            def eval(self):
                return self

            def to(self, device):
                return self

        return Dummy()

    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=fake_hf_download))
    monkeypatch.setitem(sys.modules, 'timm', types.SimpleNamespace(create_model=fake_create_model))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace())

    importlib.reload(fe)
    fe._load_model('cpu', fe.DEFAULT_MODEL_ID)

    assert recorded['name'] == 'dummy_arch'
    assert recorded['scale'] == 4
    assert recorded['cfg']['architecture'] == 'dummy_arch'


def test_load_model_with_architectures_list(monkeypatch, tmp_path):
    recorded = {}

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text('{"architectures": ["list_arch"]}')

    def fake_hf_download(repo, filename):
        assert filename == "config.json"
        return str(cfg_path)

    def fake_create_model(name, pretrained=True, pretrained_cfg=None, scale=None):
        recorded['name'] = name
        recorded['scale'] = scale
        recorded['cfg'] = pretrained_cfg

        class Dummy:
            def eval(self):
                return self

            def to(self, device):
                return self

        return Dummy()

    monkeypatch.setitem(sys.modules, 'huggingface_hub', types.SimpleNamespace(hf_hub_download=fake_hf_download))
    monkeypatch.setitem(sys.modules, 'timm', types.SimpleNamespace(create_model=fake_create_model))
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace())

    importlib.reload(fe)
    fe._load_model('cpu', fe.DEFAULT_MODEL_ID)

    assert recorded['name'] == 'list_arch'
    assert recorded['scale'] == 4
    assert recorded['cfg']['architecture'] == 'list_arch'
