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
set -euo pipefail

UIDGID="$(id -u):$(id -g)"
FRAMES_DIR="${FRAMES_DIR:-/app/frames}"
COURT_JSON="${COURT_JSON:-/app/court.json}"
OUT_JSON="${OUT_JSON:-/app/court_by_name.json}"

docker run --rm --user "$UIDGID" \
  -v "$(pwd)":/app \
  -e FRAMES_DIR="$FRAMES_DIR" \
  -e COURT_JSON="$COURT_JSON" \
  -e OUT_JSON="$OUT_JSON" \
  --entrypoint python \
  decoder-track:latest /app/tools/map_court_by_name.py
