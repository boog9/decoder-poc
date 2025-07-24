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

## Setup

Install the required Python packages and run tests:

```bash
python -m pip install -U -r requirements.txt
make test
```

Build the Docker image (requires NVIDIA GPU drivers):

```bash
make build
```

## Frame Enhancement CLI

Enhance extracted frames using the Swin2SR model
(`caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr`, scale=4).
Requires a CUDA-enabled GPU.

```
python -m src.frame_enhancer \
    --input-dir frames/ \
    --output-dir frames_sr/ \
    --batch-size 4 \
   --model-id caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr \
   --fp16
```

The `--fp16` flag converts the model and inputs to half precision,
reducing memory usage on supported GPUs.

If you encounter CUDA out-of-memory errors, reduce `--batch-size` or set the
environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```


Before running the enhancement script, install the Python dependencies. Ensure
that ``python`` and ``pip`` come from the same environment:

```
python -m pip install -U -r requirements.txt
```

The YOLOX models loaded via ``torch.hub`` also require the ``loguru`` package,
OpenCV, and ``tabulate``. These dependencies are included in
``requirements.txt`` (``loguru``, ``opencv-python-headless``, ``tabulate``). If
installing individually, add these packages before running the detection CLI.

Alternatively, build the Docker image which installs everything via `make build`.
The Pillow and NumPy packages used by ``frame_enhancer.py`` come from
``requirements.txt``.

## Object Detection CLI

Run YOLOX object detection on extracted frames. Only ``person`` detections are
saved. This command requires a CUDA-enabled GPU and YOLOX 0.3+.

```bash
python -m src.detect_objects \
    --frames-dir frames/ \
    --output-json detections.json \
    --model yolox-s \
    --img-size 640 \
    --conf-thres 0.30 \
    --nms-thres 0.45
```

To use the GPU-enabled Docker image, build it with ``Dockerfile.detect``:

```bash
docker build -f Dockerfile.detect -t decoder-detect:latest --progress=plain .
```

Run detection inside the container (assumes frames are in ``./frames``):

```bash
docker run --gpus all --rm -v $(pwd):/app decoder-detect:latest \
    --frames-dir frames/ \
    --output-json detections.json \
    --model yolox-s \
    --img-size 640
```

This image installs YOLOX and its dependencies using the official
``pytorch/pytorch`` CUDA runtime as the base image, which already includes
PyTorch with GPU support. The YOLOX package is installed directly from the
GitHub repository to avoid issues with the PyPI release.

## ROI Visualization CLI

Overlay detection bounding boxes on extracted frames.

```bash
python -m src.draw_roi \
    --frames-dir frames/ \
    --detections-json detections.json \
    --output-dir frames_roi/
```

The command reads detection results from ``detections.json`` and writes
annotated images to ``frames_roi`` using red rectangles by default.

## Detection Validation CLI

Run simple quality checks on detection results.

```bash
python -m src.validate_detections sanity-check \
    --detections detections.json \
    --frames-dir frames/
```

This prints the number of invalid bounding boxes and low-confidence detections.
