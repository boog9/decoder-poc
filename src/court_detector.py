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
"""Tennis court detection wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image
from loguru import logger

_model = None
_model_device: Optional[str] = None
_model_weights: Optional[Path] = None
_logged_output_type = False


def _maybe_import_external_builder() -> Optional[Callable[..., Any]]:
    """Try to import a model builder from ``services.court_detector``."""

    try:  # Variant 1
        from services.court_detector.model import build as ext_build  # type: ignore

        return ext_build
    except Exception:
        pass
    try:  # Variant 2
        from services.court_detector.tcd import build_tcd_model as ext_build  # type: ignore

        return ext_build
    except Exception:
        return None


def build_tcd_model() -> Any:
    """Fallback builder if no external implementation is provided."""

    raise RuntimeError(
        "No external TCD builder found. Provide services.court_detector.* build function."
    )


def _get_model(device: str, weights: Path, weights_type: str) -> Any:
    """Load detector weights once and cache the model instance."""

    global _model, _model_device, _model_weights
    if _model is not None and _model_device == device and _model_weights == weights:
        return _model

    import torch  # pragma: no cover - heavy dependency

    # fmt: off
    logger.info("loading court detector ({}) from {} on {}", weights_type, weights, device)
    # fmt: on

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but not available. Use --device cpu or rebuild with CUDA base image."
        )

    if weights_type == "jit":
        _model = torch.jit.load(str(weights), map_location=device).eval()
    else:
        # 1) читаємо чекпойнт і готуємо sd
        state = torch.load(str(weights), map_location="cpu")
        sd = state.get("state_dict", state) if isinstance(state, dict) else state

        # 2) інференс базової ширини з першого шару (для tcd.pth це 64)
        try:
            base_ch = int(sd["conv1.block.0.weight"].shape[0])
        except Exception:
            base_ch = 16  # дефолт на випадок нестандартного ckpt

        # 3) будуємо мережу через external builder (якщо є) або fallback
        builder = _maybe_import_external_builder() or build_tcd_model

        import inspect

        sig = inspect.signature(builder)
        kwargs = {}
        for name in ("base_channels", "base_ch"):
            if name in sig.parameters:
                kwargs[name] = base_ch
                break
        net = builder(**kwargs) if kwargs else builder()

        # 4) shape-guard: залишаємо лише ті ваги, де форма збігається (страховка)
        model_sd = net.state_dict()
        sd_compatible = {
            k: v
            for k, v in sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        dropped = sorted(set(sd.keys()) - set(sd_compatible.keys()))
        if dropped:
            logger.warning(
                "Dropped {} keys due to shape mismatch: {}", len(dropped), dropped[:10]
            )

        info = net.load_state_dict(sd_compatible, strict=False)
        missing = getattr(info, "missing_keys", [])
        unexpected = getattr(info, "unexpected_keys", [])
        logger.info(
            "state_dict load (base_channels={}): missing={}, unexpected={}",
            base_ch,
            len(missing),
            len(unexpected),
        )

        net.to(device).eval()
        _model = net

    _model_device = device
    _model_weights = weights
    return _model


def detect_single_frame(
    img: Image.Image, *, device: str, weights: Path, min_score: float
) -> Dict[str, Any]:
    """Detect court geometry on a single frame.

    Args:
        img: Input frame.
        device: ``"cpu"`` or ``"cuda"`` for model execution.
        weights: Path to detector weights.
        min_score: Minimum confidence threshold.

    Returns:
        Dictionary with ``polygon``, ``lines``, ``homography`` and ``score``
        keys. Returns an empty dictionary when detection fails or ``score`` is
        below ``min_score``.
    """

    from .utils.checkpoint import verify_torch_ckpt

    wtype = verify_torch_ckpt(str(weights))
    model = _get_model(device, weights, wtype)

    try:  # pragma: no cover - preprocessing may fail if torch missing
        import torch  # heavy dependency

        try:
            from torchvision.transforms.functional import to_tensor  # type: ignore

            tensor = to_tensor(img).unsqueeze(0).to(device)
        except Exception:  # torchvision missing
            num_channels = len(img.getbands())
            data = torch.tensor(list(img.tobytes()), dtype=torch.uint8)
            tensor = (
                data.view(img.height, img.width, num_channels)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )
        with torch.no_grad():
            raw = model(tensor)
    except Exception:  # Fallback: pass PIL image directly
        raw = model(img)

    global _logged_output_type
    if not _logged_output_type:
        logger.debug("detector output type: {}", type(raw).__name__)
        _logged_output_type = True

    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if not isinstance(raw, dict):  # pragma: no cover - unexpected output
        logger.error("unexpected detector output: {}", type(raw).__name__)
        return {}

    polygon = raw.get("polygon") or raw.get("court_polygon") or []
    lines = raw.get("lines") or raw.get("court_lines") or {}
    homography = (
        raw.get("homography")
        or raw.get("H")
        or [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    score = float(raw.get("score", 0.0))
    if score < min_score or not polygon:
        return {}

    return {
        "polygon": polygon,
        "lines": lines,
        "homography": homography,
        "score": score,
    }
