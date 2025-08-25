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
"""Tests for :mod:`src.utils.checkpoint`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.checkpoint import verify_torch_ckpt


def test_verify_ckpt_missing(tmp_path: Path) -> None:
    """Raises when checkpoint does not exist."""

    with pytest.raises(FileNotFoundError):
        verify_torch_ckpt(str(tmp_path / "missing.pth"))


def test_verify_ckpt_too_small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises when checkpoint file is below minimum size."""

    fp = tmp_path / "w.pth"
    fp.write_bytes(b"123")

    import src.utils.checkpoint as ck

    monkeypatch.setattr(ck.torch, "load", lambda p, map_location=None: {}, raising=False)
    with pytest.raises(ValueError):
        verify_torch_ckpt(str(fp), min_bytes=1024)


def test_verify_ckpt_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Succeeds for a sufficiently large file."""

    fp = tmp_path / "w.pth"
    fp.write_bytes(b"0" * 2048)

    import src.utils.checkpoint as ck

    monkeypatch.setattr(ck.torch, "load", lambda p, map_location=None: {}, raising=False)
    assert verify_torch_ckpt(str(fp), min_bytes=1024) == "state_dict"


def test_verify_ckpt_jit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detects TorchScript checkpoints when torch.load fails."""

    fp = tmp_path / "w.pth"
    fp.write_bytes(b"0" * 2048)

    import src.utils.checkpoint as ck

    class DummyJIT:
        pass

    monkeypatch.setattr(ck.torch, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(ck.torch.jit, "load", lambda *a, **k: DummyJIT())
    assert verify_torch_ckpt(str(fp), min_bytes=1024) == "jit"

