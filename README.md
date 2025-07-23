# decoder-poc

This repository contains experimental utilities for video processing. The first
module provides a command-line interface for extracting frames from a video
using FFmpeg.

## Frame Extractor

```
python -m src.frame_extractor --input input.mp4 --output frames --fps 30
```

The script logs extraction progress and writes frames as JPEG images in the
specified directory.
