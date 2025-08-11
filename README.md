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

When cloning the repository make sure to also fetch the ``ByteTrack``
submodule:

```bash
git clone --recursive <repo-url>
```

If the repository was cloned without ``--recursive`` run:

```bash
git submodule update --init --recursive
```

## Setup

Install the system dependencies and Python packages, then fetch the
``ByteTrack`` submodule and verify the vendored tracker.

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-dev
git submodule update --init --recursive
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install pytest
bash build_externals.sh  # sanity-check ByteTrack vendor only
make test
```

> **Note:** PyTorch is provided by the base image `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`.
> Do not install or upgrade `torch` or `torchvision` via pip—this increases the image size
> and may break CUDA compatibility.

### External Dependencies

Verify the vendored ByteTrack tracker:

```bash
git submodule update --init --recursive
bash build_externals.sh
# the script only checks that the ByteTrack vendor code is present
```

If `build_externals.sh` exits with the message:

```
ByteTrack vendor not found at /path/to/externals/ByteTrack/bytetrack_vendor
Run: git submodule update --init --recursive
```

the repository was cloned without the ByteTrack code. Re-fetch the submodule or clone it manually:

```bash
git submodule update --init --recursive
# or, if the folder is empty
rm -rf externals/ByteTrack
git clone https://github.com/ifzhang/ByteTrack.git externals/ByteTrack
```

If the clone fails with a `CONNECT tunnel failed: 403` error, it
is due to network restrictions. Re-run the command once internet
access is restored or fetch the code via another method.

Then run `bash build_externals.sh` again.

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
# 1) Явно з detect
python -m src.detect_objects detect \
  --frames-dir    ./frames \
  --output-json   detections.json \
  --model         yolox-x \
  --img-size      1280 \
  --conf-thres    0.1 \
  --nms-thres     0.1

# 2) Без сабкоманди (виконання detect за замовчуванням)
python -m src.detect_objects \
  --frames-dir    ./frames \
  --output-json   detections.json \
  --model         yolox-x \
  --img-size      1280 \
  --conf-thres    0.1 \
  --nms-thres     0.1

# 3) Трекінг
python -m src.detect_objects track \
  --detections-json detections.json \
  --output-json     tracks.json \
  --min-score       0.3
```

To use the GPU-enabled Docker image, build it with ``Dockerfile.detect``:

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.detect -t decoder-detect:latest \
    --build-arg YOLOX_REF=0.3.0 \
    --progress=plain .
```

The `YOLOX_REF` build argument accepts a tag, branch or commit. If the
specified ref is unavailable, the build falls back to ``main`` and prints a
warning.

Run detection inside the container (assumes frames are in ``./frames``):

```bash
docker run --gpus all --rm -v $(pwd):/app decoder-detect:latest \
    detect --frames-dir frames/ \
    --output-json detections.json \
    --model yolox-s \
    --img-size 640 \
    --conf-thres 0.30 \
    --nms-thres 0.45 \
    --classes 0 32
```

> **Note:** the image already has ENTRYPOINT `python -m src.detect_objects`.
> Do not prefix the command with `python`; the first argument must be the
> subcommand `detect` or `track`.

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

## Object Tracking CLI

After running detection you can generate consistent tracks for each
``person`` and ``ball`` class using ByteTrack. The tracker is vendored in
``externals/ByteTrack/bytetrack_vendor`` and is isolated from the official
YOLOX package. Verify its presence via ``build_externals.sh`` before running the
tracking step; the script now performs only a sanity check and no native
YOLOX modules need to be built.
Only detections with score above ``--min-score`` are considered.

```bash
python -m src.detect_objects track \
    --detections-json detections.json \
    --output-json tracks.json \
    --min-score 0.30
```

* ``--detections-json`` – input file produced by the detection step.
* ``--output-json`` – destination for the tracked results.
* ``--min-score`` – detection score threshold (default: ``0.3``).

The detection CLI relies on the official `yolox` package installed from the
GitHub repository, while the tracking CLI imports only
`bytetrack_vendor.*` from the vendored ByteTrack tree. There are no shared
imports between the two; running detection and tracking in separate
containers (`decoder-detect` and `decoder-track`) avoids dependency
conflicts.

The detections file may use either of the following schemas:

1. Nested per-frame structure:
   `[{"frame": "frame_000001.png", "detections": [{...}]}]`
2. Flat list of detections:
   `[{"frame": 1, "class": "person", "bbox": [..], "score": 0.9}]`

Both formats are parsed automatically.
Frame IDs are derived from the last group of digits in the `frame` value (e.g., `frame_000123.png` → `123`), eliminating 0/1-based ambiguity.

**BBox format:** each `bbox` must be `[x1, y1, x2, y2]` (XYXY, pixels).

**Class field:** accepts COCO IDs or strings like `"person"`, `"sports ball"` (alias `"ball"`).

The output ``tracks.json`` contains entries of the form:

```json
[
  {"frame": 1, "class": 0,  "track_id": 5, "bbox": [x1, y1, x2, y2], "score": 0.93},
  {"frame": 1, "class": 32, "track_id": 2, "bbox": [x1, y1, x2, y2], "score": 0.88}
]
```

The command logs the number of processed frames, the count of active tracks
and a per-class summary.

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

## Track visualisation (draw_tracks)

After running detection and tracking you can overlay tracking results either on individual frames or combine them into an MP4 video. The command reads ``tracks.json`` produced by ``detect_objects track`` and draws each track ID with a deterministic colour.

```bash
python -m src.draw_tracks \
    --frames-dir frames/ \
    --tracks-json tracks.json \
    --output-dir frames_tracks/ \
    --label
```

To create an MP4 instead of annotated images:

```bash
python -m src.draw_tracks \
    --frames-dir frames/ \
    --tracks-json tracks.json \
    --output-video out.mp4 \
    --fps 25
```

| Option | Description |
| ------ | ----------- |
| ``--frames-dir`` | Directory with input frame images |
| ``--tracks-json`` | ByteTrack output JSON |
| ``--output-dir`` | Destination folder for annotated frames |
| ``--output-video`` | Destination MP4 file |
| ``--fps`` | Frames per second for MP4 (default ``30``) |
| ``--label/--no-label`` | Draw text labels with class and ID |
| ``--palette`` | ``coco`` uses fixed COCO colours, ``random`` assigns random deterministic colours, ``track`` hashes the ID |
| ``--thickness`` | Bounding box thickness |
| ``--max-frames`` | Limit number of processed frames |

**Exactly one of `--output-dir` or `--output-video` must be provided.**

Use this step after ``detect_objects track`` and before any further processing that requires visual inspection.
Place this step right after ``detect_objects track`` in the makefile / bash-script flow.

## Detection Validation CLI

Run simple quality checks on detection results.

```bash
python -m src.validate_detections sanity-check \
    --detections detections.json \
    --frames-dir frames/
```

This prints the number of invalid bounding boxes and low-confidence detections.

## Detection Docker Image

- **Service name:** `decoder-detect`
- **Purpose:** Run YOLOX detection on a directory of frames using the official `yolox` package.
- **GPU:** Required; enable with `--gpus all`.
- **Volumes:** Mount project directory to `/app` to access frames and outputs.
- **Build:**

  ```bash
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.detect -t decoder-detect:latest \
      --build-arg YOLOX_REF=0.3.0 .
  ```

  The `YOLOX_REF` argument accepts a tag, branch or commit. If the ref is
  invalid, the build falls back to ``main`` and prints a non-fatal warning. A
  YOLOX smoke-check runs during the build; failures only issue a warning and do
  not stop the build.

- **Run:**

  ```bash
  docker run --gpus all --rm -v $(pwd):/app decoder-detect:latest \
      detect --frames-dir /app/frames --output-json /app/detections.json
  ```

> **Note:** The image sets `ENTRYPOINT ["python","-m","src.detect_objects"]`.
> For one-off Python commands use:
> `docker run --rm --entrypoint python decoder-detect:latest -c "import yolox; print(yolox.__file__)"`.

- **Parameters:**

  | Option | Description | Default |
  | ------ | ----------- | ------- |
  | `--frames-dir` | Input PNG/JPG frames directory | **required** |
  | `--output-json` | Path to save detection results | **required** |
  | `--model` | YOLOX model size (`yolox-s` ... `yolox-x`) | `yolox-s` |
  | `--img-size` | Inference image size | `640` |
  | `--conf-thres` | Confidence threshold | `0.3` |
  | `--nms-thres` | NMS threshold | `0.45` |
  | `--classes` | Filter by class IDs | none |


## Tracking Docker Image

- **Service name:** `decoder-track`
- **Purpose:** Run ByteTrack tracking on detection results using the vendored `bytetrack_vendor` package.
- **GPU:** Required; enable with `--gpus all`.
- **Volumes:** Mount project directory to `/app` to access inputs and outputs.
- **Build:**

  ```bash
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.track -t decoder-track:latest .
  ```

  The image sets ``PYTHONPATH`` before verifying the ByteTrack vendor to avoid
  ``ModuleNotFoundError`` during the build-time sanity check.

- **Run:**

  ```bash
  docker run --gpus all --rm -v $(pwd):/app decoder-track:latest \
      track --detections-json /app/detections.json --output-json /app/tracks.json
  ```

- **Parameters:**

  | Option | Description | Default |
  | ------ | ----------- | ------- |
  | `--detections-json` | Input JSON from detection step | **required** |
  | `--output-json` | Path to save tracked results | **required** |
  | `--min-score` | Detection score threshold | `0.3` |

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

## Troubleshooting

- **ImportError when loading YOLOX** – Ensure the official `yolox` package is
  installed in the detection environment. For tracking errors verify that the
  ByteTrack vendor is verified via `bash build_externals.sh`.

- **Missing weights** – Download the official YOLOX checkpoints and place them
  in the ``weights`` directory, e.g. ``weights/yolox_x.pth``.

## 4. Visualization (decoder-draw)

Render bounding boxes for detection or tracking results.

```bash
# 1) Build the image
make draw

# 2a) Visualise detections
make draw-run-detect

# 2b) Visualise tracks with IDs
make draw-run-track

# 2c) Assemble frames into MP4
make draw-run-mp4

# Show CLI help
docker run --rm decoder-draw:latest --help
```

The CLI reads frame images and one of `detections.json` or `tracks.json`.
Two JSON schemas are accepted:

* Nested per-frame: `[{"frame": "frame_000123.png", "detections": [{"bbox": [x1,y1,x2,y2], "class": <int|str>, "score": float}]}]`
* Flat list: `[{"frame": "frame_000123.png", "bbox": [...], "class": ..., "score": ..., "track_id": int?}]`

Supported keys: `frame`, `bbox`, `class`, `score`, `track_id`.
Default class mapping: `0: person`, `32: sports ball`.
Filter classes via `--only-class`, e.g. `--only-class person,sports ball`.

Notes on reliability:

* Falls back to Pillow if `cv2.imread` fails.
* Bounding boxes are clipped to frame boundaries.
* Output frames are sorted lexicographically before MP4 export.
* MP4 export uses ffmpeg **image2** demuxer with a staged numeric sequence for deterministic order.
* Frame names like `frame_%06d.png` are expected when resolving by index.

Pipeline usage:

```
detect -> detections.json
track  -> tracks.json
draw   -> overlays in out/frames_viz and optional MP4
```

Image name: `decoder-draw:latest` (CPU only). Mount the repository as `/app`.
Parameters: run `docker run --rm decoder-draw:latest --help`.
Modes: `--mode auto|detect|track` (default `auto` picks `track` if `--tracks-json` exists, otherwise `detect`).
