# decoder-poc

This project contains experimental utilities for video processing. The `frame_extractor` CLI provides a simple way to extract video frames using FFmpeg.

## Frame Extraction CLI

```
python src/frame_extractor.py -i <video.mp4> -o /path/to/output -f 30
```

- `-i`, `--input`: Path to input video.
- `-o`, `--output`: Output directory for PNG frames.
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

Install the following packages to run detection:

* ``torch`` (with CUDA support)
* ``opencv-python-headless``
* ``loguru``
* ``tabulate``

YOLOX 0.3+ is required and can be installed from GitHub or via the provided
Docker image. These dependencies are already included in ``requirements.txt``.

Alternatively, build the Docker image which installs everything via `make build`.
The Pillow and NumPy packages used by ``frame_enhancer.py`` come from
``requirements.txt``.

## Object Detection CLI

Run YOLOX object detection on extracted frames. A CUDA-enabled GPU and YOLOX
``v0.3`` or newer are required. The ``--classes`` option takes one or more
numeric class IDs. If omitted, detections for all classes are kept. For
example, ``--classes 0 32`` keeps only ``person`` and ``sports ball``.

```bash
python -m src.detect_objects \
    --frames-dir frames/ \
    --output-json detections.json \
    --model yolox-s \
    --img-size 640 \
    --conf-thres 0.30 \
    --nms-thres 0.45 \
    --classes 0 32
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
    --img-size 640 \
    --conf-thres 0.30 \
    --nms-thres 0.45 \
    --classes 0 32
```

Example ``detections.json`` output:

```json
[
  {
    "frame": "frame_000001.png",
    "detections": [
      {"bbox": [15, 30, 210, 330], "score": 0.91, "class": 0}
    ]
  },
  {
    "frame": "frame_000002.png",
    "detections": []
  }
]
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
    --output-dir frames_roi/ \
    --label
```

The command reads detection results from ``detections.json`` and writes
annotated PNG images to ``frames_roi``. Using ``--label`` draws the COCO class name
and confidence score above each box. The bounding box coordinates are expected
to match the original frame pixels, so no scaling is applied when drawing.

## Detection Validation CLI

Run simple quality checks on detection results.

```bash
python -m src.validate_detections sanity-check \
    --detections detections.json \
    --frames-dir frames/
```

This prints the number of invalid bounding boxes and low-confidence detections.

## Single Image Detection Demo

A small container is provided to run YOLOX on a single image. Build it with:

```bash
docker build -f Dockerfile.image -t decoder-image:latest .
```

Run detection on ``frame_000019.jpg`` using the GPU:

```bash
docker run --gpus all --rm -v $(pwd):/app decoder-image:latest \
    --image /app/frames01/frame_000019.jpg \
    --model yolox-x \
    --conf-thres 0.10 \
    --nms-thres 0.45 \
    --img-size 640 \
    --device gpu \
    --save-result /app/out.jpg
```

The command prints the detections as JSON and saves the annotated image to
``out.jpg``.
