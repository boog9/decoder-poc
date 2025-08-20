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
"""Common class ID mappings used across utilities."""

from __future__ import annotations

CLASS_NAME_TO_ID = {
    "person": 0,
    "sports ball": 32,
    "tennis_court": 100,
}

CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}
