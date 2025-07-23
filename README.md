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

Example output:

```
2024-01-01 12:00:05 - INFO - Completed extraction of 150 frames in 5.00 seconds
```

The script requires FFmpeg to be installed and available on the system path.

## Frame Enhancement CLI

Enhance extracted frames using the Swin2SR model. Requires a CUDA-enabled GPU.

```
python -m src.frame_enhancer --input-dir frames/ --output-dir frames_sr/ --batch-size 4
```

Before running the enhancement script, install the Python dependencies:

```
pip install -r requirements.txt
```

Alternatively, build the Docker image which installs everything via `make build`.
The Pillow and NumPy packages used by ``frame_enhancer.py`` come from
``requirements.txt``.

