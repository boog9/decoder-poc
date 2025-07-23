# decoder-poc

This project contains experimental utilities for video processing. The `frame_extractor` CLI provides a simple way to extract video frames using FFmpeg.

## Frame Extraction CLI

```
python src/frame_extractor.py -i <video.mp4> -o /path/to/output -f 30
```

- `-i`, `--input`: Path to input video.
- `-o`, `--output`: Output directory for JPEG frames.
- `-f`, `--fps`: Frames per second to extract (default: 30).
- `-v`, `--verbose`: Increase logging detail.

Example output logs:

```
2024-01-01 12:00:00 - INFO - Running command: ffmpeg -i input.mp4 -vf fps=30 output/frame_%06d.jpg -loglevel error -progress pipe:1
2024-01-01 12:00:01 - INFO - frame=1
...
2024-01-01 12:00:02 - INFO - progress=end
```

The script requires FFmpeg to be installed and available on the system path.

