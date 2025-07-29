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
"""Package for video processing utilities."""

__all__ = []


def draw_tracks_cli(*args, **kwargs):
    """Lazy import wrapper for :func:`src.draw_tracks.cli`."""

    from .draw_tracks import cli

    return cli(*args, **kwargs)


__all__.append("draw_tracks_cli")



