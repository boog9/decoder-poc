#!/usr/bin/env bash
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
set -Eeuo pipefail
make draw
mkdir -p out/frames_viz

docker run --rm -v "$PWD:/app" decoder-draw:latest \
  --frames-dir /app/data/frames_min \
  --tracks-json /app/tracks.json \
  --output-dir /app/out/frames_viz --label --id --max-frames 10

if ! find out/frames_viz -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) | head -n1 | grep -q .; then
  echo "No frames produced" >&2
  exit 1
fi
echo "OK"
