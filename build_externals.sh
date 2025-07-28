#!/usr/bin/env bash
set -euo pipefail

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

# Build C++ extensions for vendored dependencies such as ByteTrack.

BYTE_DIR="$(dirname "$0")/externals/ByteTrack"

if [ ! -f "$BYTE_DIR/setup.py" ]; then
    echo "ByteTrack submodule not found." >&2
    echo "Run 'git submodule update --init --recursive' first." >&2
    echo "If the directory is empty you can clone it manually:" >&2
    echo "  rm -rf externals/ByteTrack" >&2
    echo "  git clone https://github.com/ifzhang/ByteTrack.git externals/ByteTrack" >&2
    echo "If cloning fails with 'CONNECT tunnel failed: 403', try again when" >&2
    echo "network access is available." >&2
    exit 1
fi

pushd "$BYTE_DIR" >/dev/null
# Use Python 3 explicitly in Docker since `python` may not be installed.
python3 -m pip install --user cython pybind11 packaging
python3 setup.py build_ext --inplace
popd >/dev/null

