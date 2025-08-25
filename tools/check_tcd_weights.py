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
"""Check that the tennis court detector weights match the model structure."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch


def main() -> None:  # pragma: no cover - used in Docker verification
    """Load the model and verify state_dict compatibility."""

    module = importlib.import_module("services.court_detector.tcd")
    model_static = getattr(module, "TennisCourtDetector")
    model_dynamic = getattr(module, "TennisCourtDetectorFromSD", None)
    try:
        sd_raw = torch.load("weights/tcd.pth", map_location="cpu", weights_only=True)
    except TypeError:
        sd_raw = torch.load("weights/tcd.pth", map_location="cpu")
    sd = sd_raw.get("state_dict", sd_raw) if isinstance(sd_raw, dict) else sd_raw
    if model_dynamic is not None:
        model = model_dynamic(sd)
    else:
        model = model_static()
    msd = model.state_dict()

    mismatched = [
        (k, tuple(sd[k].shape), tuple(msd[k].shape))
        for k in sd
        if k in msd and sd[k].shape != msd[k].shape
    ]
    missing = sorted([k for k in sd.keys() if k not in msd.keys()])
    unexpected = sorted([k for k in msd.keys() if k not in sd.keys()])

    print(f"shape_mismatches: {len(mismatched)}")
    for k, s1, s2 in mismatched[:20]:
        print(" ", k, s1, s2)
    print(f"missing: {len(missing)}")
    print(" ", missing[:20])
    print(f"unexpected: {len(unexpected)}")
    print(" ", unexpected[:20])

    print("has conv1.block.2.running_mean:", "conv1.block.2.running_mean" in sd)
    print("conv10.block.0.weight shape:", sd["conv10.block.0.weight"].shape)


if __name__ == "__main__":  # pragma: no cover
    main()

