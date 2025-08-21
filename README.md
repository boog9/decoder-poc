# decoder-poc

This project contains experimental utilities for video processing. The `frame_extractor` CLI provides a simple way to extract video frames using FFmpeg.

> All command-line interfaces should be executed in Docker containers. Running them locally is intended for developers only.

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

### Prerequisites (Docker)

- Docker 24+ і NVIDIA Container Toolkit.
- GPU має бути видимим у контейнері:
  ```bash
  docker run --gpus all --rm nvidia/cuda:12.2.0-base nvidia-smi
  ```
- У прикладах нижче завжди монтуємо репозиторій як /app:
  `-v "$(pwd)":/app`

## Setup (developers only)

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

### Build images

Зберіть контейнер(и) один раз перед запуском пайплайна:

```bash
# Court detector
DOCKER_BUILDKIT=1 docker build -f Dockerfile.court  -t decoder-court:latest .

# YOLOX detect
DOCKER_BUILDKIT=1 docker build -f Dockerfile.detect -t decoder-detect:latest .

# ByteTrack track
DOCKER_BUILDKIT=1 docker build -f Dockerfile.track  -t decoder-track:latest .
```

Якщо у вас є Makefile-цілі (make court, make detect, make track), можете використовувати їх — але приклади нижче припускають явні docker build.

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

## Overlay Rendering CLI

Draw detection or tracking overlays on extracted frames. Colors are stable
either per class or per track ID depending on the selected mode.

```bash
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
  --frames-dir /app/frames_min \
  --tracks-json /app/tracks.json \
  --output-dir /app/frames_tracks \
  --mode track --label --id --draw-court --draw-court-lines

docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
  --frames-dir /app/frames_min \
  --detections-json /app/detections.json \
  --output-dir /app/frames_det \
  --mode class --label
```

Use `--draw-court=false` to hide the court polygon and
`--no-draw-court-lines` to omit internal lines.

`--roi-json` accepts either a single polygon `{ "polygon": [...] }` or a `court.json` file with per-frame polygons (the polygon from the first frame is used).

- `--only-court`: Draw only the court contour without boxes or IDs.
- `--palette-seed`: Stabilise the colour palette globally.
- `--class-map`: Optional JSON/YAML mapping of class IDs to names.
- `--confidence-thr`: Filter detections below this score.

Example rendering only the court outline:

```bash
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
  --frames-dir /app/frames --detections-json /app/detections.json \
  --output-dir /app/preview_court --only-court --draw-court-lines --roi-json /app/court.json
```

The legacy `src.draw_tracks` module is kept for backwards compatibility and
forwards all arguments to `src.draw_overlay --mode track`.


Before running the enhancement script, install the Python dependencies. Ensure
that ``python`` and ``pip`` come from the same environment:

```
python -m pip install -U -r requirements.txt
```

Install the following packages to run detection:

* ``torch`` (with CUDA support)
* ``opencv-python-headless``
* ``loguru``

## Tennis defaults (YOLOX + ByteTrack)

End-to-end приклад з тюнінгами під теніс (Docker-first):

```bash
# 1) court detection -> /app/court.json
docker run --rm -v "$(pwd)":/app decoder-court:latest \
  --frames-dir /app/frames --output-json /app/court.json \
  --weights /app/weights/tcd.pth --sample-rate 5 --stabilize ema

# 2) detect – class-aware NMS, ROI gate, ball interpolation up to 5 frames
docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest \
  detect --frames-dir /app/frames \
         --output-json /app/detections.json \
         --two-pass --nms-class-aware \
         --roi-json /app/court.json --roi-margin 8 \
         --detect-court

# 3) track – softer thresholds, wider buffers, stitching & smoothing
# IMPORTANT: pass --frames-dir to auto-enable appearance refine
docker run --gpus all --rm -v "$(pwd)":/app decoder-track:latest \
  track --detections-json /app/detections.json \
        --output-json /app/tracks.json \
        --frames-dir /app/frames \
        --fps 30 --min-score 0.28 \
        --pre-nms-iou 0.6 --pre-min-area-q 0.15 --pre-topk 3 --pre-court-gate \
        --p-match-thresh 0.55 --p-track-buffer 160 --reid-reuse-window 150 \
        --b-match-thresh 0.55 --b-track-buffer 150 \
        --stitch --stitch-iou 0.55 --stitch-gap 12 \
        --smooth ema --smooth-alpha 0.3
```
> `pre-court-gate` works only with a real court polygon. If `court.json` is a full-frame placeholder, the gate has no effect—either disable it (`--no-pre-court-gate`) or run `decoder-court` with weights to obtain real geometry.

If you see `pre-court-gate disabled: detected full-frame court polygon`, this is expected for placeholder polygons.


ROI note: --roi-json приймає або один полігон { "polygon": [...] }, або court.json (список полігонів по кадрах). У випадку списку використовується полігон з першого кадру. Якщо камера рухома — краще не використовувати ROI на етапі Detect, а покладатися на --pre-court-gate у Track.

ROI debug:

```bash
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay --only-court --draw-court-lines --roi-json /app/court.json \
  --frames-dir /app/frames --detections-json /app/detections.json \
  --output-dir /app/preview_court
```


Lower b-match-thresh + higher b-track-buffer стабілізують ID м’яча. Для гравців — зменшений p-match-thresh і збільшений p-track-buffer допомагають утримати ID біля сітки. pre-min-area-q=0.15 не ріже далекого гравця, а stitch=* зливає короткі обриви. appearance-refine активується автоматично, коли передано --frames-dir.

### Verification pipeline (quick check)

1) Візуалізація оверлею (детекції/треки → PNG + MP4):

```bash
# Tracks overlay to PNGs
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
  --mode track \
  --frames-dir /app/frames \
  --tracks-json /app/tracks.json \
  --output-dir /app/preview_tracks \
  --draw-court --draw-court-lines --roi-json /app/court.json

# Optional MP4 export (25 fps)
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
  --mode track \
  --frames-dir /app/frames \
  --tracks-json /app/tracks.json \
  --output-dir /app/preview_tracks \
  --export-mp4 /app/preview_tracks.mp4 --fps 25 --draw-court --draw-court-lines --roi-json /app/court.json
```

Запускати можна всередині будь-якого образу, де є Python + OpenCV. Найпростіше — у decoder-track:latest з примонтованим репозиторієм:

```bash
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
    --mode track --frames-dir /app/frames \
    --tracks-json /app/tracks.json \
    --output-dir /app/preview_tracks \
    --export-mp4 /app/preview_tracks.mp4 --fps 25 \
    --draw-court --draw-court-lines --roi-json /app/court.json
```

Саніті-метрики (скільки унікальних гравців, частка кадрів з м’ячем, середня довжина треку м’яча):

```bash
docker run --rm -v "$(pwd)":/app decoder-track:latest \
  python tools/verify_tennis_defaults.py --tracks-json /app/tracks.json
# очікувано: players≈2, ball_frame_frac > 0, avg_ball_track збільшився проти базових налаштувань
```

### Troubleshooting

- **YOLOX not found / torch CUDA issues:** запускайте детекцію через контейнер `decoder-detect:latest`:
  ```bash
  docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest detect ...
  ```
- GPU не видно у контейнері: перевірте nvidia-smi у контейнері (див. вище) і встановіть NVIDIA Container Toolkit.
- Порожні детекції для далекого гравця: спробуйте --pre-min-area-q 0.0 у трекері, або підвищіть --person-img-size у детекторі.
- М’яч часто втрачає ID: зменшіть --b-match-thresh (наприклад, до 0.5) і/або збільште --b-track-buffer (до 180).
- Контур корту не видно: переконайтесь, що court.json містить реальний полігон, а не «рамку кадру». Для дебагу:
  ```bash
  docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
    -m src.draw_overlay --only-court --roi-json /app/court.json \
    --frames-dir /app/frames --detections-json /app/detections.json \
    --output-dir /app/preview_court
  ```

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

Run detection inside the container (assumes frames are in `/app/frames`):

```bash
docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest \
    detect --frames-dir /app/frames \
           --output-json /app/detections.json \
           --model yolox-x \
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
docker run --gpus all --rm -v "$(pwd)":/app decoder-track:latest \
    track --detections-json /app/detections.json \
          --output-json /app/tracks.json \
          --min-score 0.30
```

* ``--detections-json`` – input file produced by the detection step.
* ``--output-json`` – destination for the tracked results.
* ``--min-score`` – detection score threshold (default: ``0.3``).

Note: We avoid accessing private STrack fields and support different ByteTrack
forks that expose `tlwh` as a property or method, and sometimes only provide
`tlbr`. This prevents crashes such as ``AttributeError: 'STrack' object has no
attribute '_tlwh'``.

The tracker initialisation automatically inspects the `BYTETracker` constructor
to support both the ``high_thresh/low_thresh`` and ``track_thresh`` variants,
removing the need for manual configuration when switching forks.

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
    --fps 30
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
  docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest \
      detect --frames-dir /app/frames --output-json /app/detections.json \
      --two-pass --detect-court

  # disable court detection explicitly
  docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest \
      detect --frames-dir /app/frames --output-json /app/detections.json \
      --two-pass --no-detect-court

  # single-pass mode
  docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest \
      detect --frames-dir /app/frames --output-json /app/detections.json \
      --two-pass=false --conf-thres 0.5 --nms-thres 0.45 --img-size 960
  ```

  Supported frame extensions: `.jpg`, `.jpeg`, `.png`.

  The ball pass uses a larger `img-size` and lower `conf` to catch the small,
  fast-moving ball.

> **Note:** The image sets `ENTRYPOINT ["python","-m","src.detect_objects"]`.
> For one-off Python commands use:
> `docker run --rm -v "$(pwd)":/app --entrypoint python decoder-detect:latest -c "import yolox; print(yolox.__file__)"`.

- **Parameters:**

  | Option | Description | Default |
  | ------ | ----------- | ------- |
  | `--frames-dir` | Input JPG/JPEG/PNG frames directory | **required** |
  | `--output-json` | Path to save detection results | **required** |
  | `--model` | YOLOX model size (`yolox-s` ... `yolox-x`) | `yolox-x` |
  | `--img-size` | Inference image size | `640` |
  | `--conf-thres` | Confidence threshold | `0.3` |
  | `--nms-thres` | NMS threshold | `0.45` |
  | `--classes` | Filter by class IDs | none |
  | `--two-pass` | Enable person/ball sequential detection | `true` |
  | `--detect-court` | Detect tennis court polygon | `true` |
  | `--court-device` | Device for court detector (`auto`, `cuda`, `cpu`) | `auto` |
  | `--court-use-homography` | Enable homography refinement (placeholder) | `false` |
  | `--court-refine-kps` | Enable keypoint refinement (placeholder) | `false` |
  | `--court-weights` | Optional path to court model weights | _none_ |
  | `--person-conf` | Person detection confidence (alias: --conf-person) | `0.55` |
  | `--person-nms` | Person NMS threshold | `0.45` |
  | `--person-img-size` | Person inference image size | `1280` |
  | `--person-classes` | Person classes | `person` |
  | `--ball-conf` | Ball detection confidence (alias: --conf-ball) | `0.15` |
  | `--ball-nms` | Ball NMS threshold | `0.30` |
  | `--ball-img-size` | Ball inference image size | `1536` |
  | `--ball-classes` | Ball classes | `"sports ball"` |
  | `--nms-class-aware` | Apply NMS per class | `true` |
  | `--roi-json` | Court polygon JSON | _none_ |
  | `--roi-margin` | ROI margin in pixels | `8` |
  | `--keep-outside-roi` | Keep detections outside ROI | `false` |
  | `--prelink-ball` | Interpolate ball gaps | `true` |
  | `--save-splits` | Save detections_person.json and detections_ball.json | `false` |


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
  docker run --gpus all --rm -v "$(pwd)":/app decoder-track:latest \
      track --detections-json /app/detections.json \
            --output-json /app/tracks.json \
            --fps 30 --min-score 0.10
  ```

  **Enhanced example with pre/post processing:**

  ```bash
  docker run --gpus all --rm -v "$(pwd)":/app decoder-track:latest \
      track --detections-json /app/detections.json \
            --output-json /app/tracks.json \
            --fps 30 --min-score 0.28 \
            --pre-nms-iou 0.6 --pre-min-area-q 0.5 --pre-topk 3 --pre-court-gate \
            --p-match-thresh 0.60 --p-track-buffer 125 --reid-reuse-window 125 \
            --stitch --stitch-iou 0.55 --stitch-gap 5 \
            --smooth ema --smooth-alpha 0.3
  ```

  *Optional colour check:* add `--appearance-refine --appearance-lambda 0.3 --frames-dir /app/frames`.

  > Note: `--fps` must match the source frame rate. Default 30 FPS.

- **Parameters:**

  | Option | Description | Default |
  | ------ | ----------- | ------- |
  | `--detections-json` | Input JSON from detection step | **required** |
  | `--output-json` | Path to save tracked results | **required** |
  | `--min-score` | Detection score threshold | `0.3` |
  | `--fps` | Video frame rate | `30` |
  | `--reid-reuse-window` | Frames to keep IDs for reuse | `125` |
  | `--pre-nms-iou` | Greedy NMS IoU for persons | `0.0` |
  | `--pre-min-area-q` | Quantile filter for small boxes | `0.0` |
  | `--pre-topk` | Keep top-K persons per frame | `0` |
  | `--pre-court-gate` | Enable court polygon gating | `False` |
  | `--court-json` | Path to court polygons | `None` |
  | `--p-track-thresh` | Person track threshold | `0.50` |
  | `--p-high-thresh` | Person high detection threshold | `0.60` |
  | `--p-match-thresh` | Person match threshold | `0.60` |
  | `--p-track-buffer` | Person track buffer | `125` |
  | `--b-track-thresh` | Ball track threshold | `0.15` |
  | `--b-high-thresh` | Ball high detection threshold | `0.30` |
  | `--b-match-thresh` | Ball match threshold | `0.85` |
  | `--b-track-buffer` | Ball track buffer | `90` |
  | `--b-min-box-area` | Minimum ball box area | `4` |
  | `--b-max-aspect-ratio` | Maximum ball aspect ratio | `2.0` |
  | `--stitch` | Enable predictive ID stitching | `False` |
  | `--smooth` | Trajectory smoothing method | `none` |
  | `--appearance-refine` | Enable HSV appearance matching | `False` |

  **Quick sanity check:**

  ```bash
  docker run --gpus all --rm -v "$(pwd)":/app decoder-track:latest \
    track --detections-json /app/detections.json \
          --output-json /app/tracks.json \
          --fps 30 --min-score 0.28 \
          --pre-nms-iou 0.6 --pre-min-area-q 0.5 --pre-topk 3 --pre-court-gate \
          --p-match-thresh 0.60 --p-track-buffer 125 --reid-reuse-window 125 \
          --stitch --stitch-iou 0.55 --stitch-gap 5 \
          --smooth ema --smooth-alpha 0.3
  ```

## Tuning for Tennis

Recommended thresholds for tennis court videos:

| Flag | Person | Ball |
| ---- | ------ | ---- |
| `--person-conf` | 0.55 | - |
| `--ball-conf` | - | 0.10 |
| `--p-track-buffer` | 125 | - |
| `--b-track-buffer` | - | 90 |

Example commands:

```bash
docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest \
  detect --frames-dir /app/frames --output-json /app/detections.json \
  --two-pass --model yolox-x \
  --person-conf 0.55 --ball-conf 0.10 --nms-class-aware \
  --roi-json /app/court_meta.json --roi-margin 8

docker run --gpus all --rm -v "$(pwd)":/app decoder-track:latest \
  track --detections-json /app/detections.json --output-json /app/tracks.json \
  --fps 30 --reid-reuse-window 125 \
  --p-track-thresh 0.50 --p-high-thresh 0.60 --p-match-thresh 0.60 --p-track-buffer 125 \
  --b-track-thresh 0.15 --b-high-thresh 0.30 --b-match-thresh 0.85 --b-track-buffer 90 \
  --b-min-box-area 4 --b-max-aspect-ratio 2.0

docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
  --frames-dir /app/frames --tracks-json /app/tracks.json \
  --output-dir /app/frames_tracks --mode track --label --id --only-court
```

## Single Image Detection Demo

A small container is provided to run YOLOX on a single image. Build it with:

```bash
docker build -f Dockerfile.image -t decoder-image:latest .
```

Run detection on ``frame_000019.jpg`` using the GPU:

```bash
docker run --gpus all --rm -v "$(pwd)":/app decoder-image:latest \
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

To render overlays, use the tracking image:

```bash
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay --help
```

Logging: loguru (>=0.7.0)

## Court Detection

Service/Docker image name: `decoder-court:latest`

Purpose: detect the tennis court polygon for each input frame.

### Startup example

```bash
docker build -t decoder-court:latest -f Dockerfile.court .

docker run --rm -v "$(pwd)":/app decoder-court:latest \
  --frames-dir /app/frames --output-json /app/court.json \
  --weights /app/weights/tcd.pth
```

- Mount `/app` to access frames and outputs
- GPU not required
- Includes `loguru` (>=0.7.0) for logging
- Provide `--weights /app/weights/tcd.pth`; without weights, add `--allow-placeholder` to emit a full-frame placeholder
  (real geometry is recommended for gating and line rendering)

### Parameters

| Option | Description | Default |
| ------ | ----------- | ------- |
| `--frames-dir` | Input directory with frame images | **required** |
| `--output-json` | Output file for court polygons | **required** |
| `--weights` | Path to `TennisCourtDetector` weights (e.g. `/app/weights/tcd.pth`) | _none_ |
| `--sample-rate` | Process every Nth frame and interpolate between | `5` |
| `--stabilize` | `ema` or `median` coordinate smoothing | `ema` |
| `--allow-placeholder` | Emit full-frame placeholder when detection fails | `false` |
| `--help` | Show CLI help | - |

The output `court.json` has one entry per frame with optional lines and
homography:

```json
[
  {
    "frame": "frame_000001.png",
    "polygon": [[0,0],[639,0],[639,359],[0,359]],
    "lines": {"service_center": [[319,0],[319,359]]},
    "homography": [[1,0,0],[0,1,0],[0,0,1]],
    "score": 0.92,
    "placeholder": false
  }
]
```
Frames without a confident detection are omitted unless
`--allow-placeholder` is enabled, in which case a full-frame polygon with
`"placeholder": true` is returned.

## Пайплан на перевірку (копіпаст і вперед)

> Припускаємо: у корені репо є `frames/` з кадрами `frame_000001.png...`

```bash
# 0) build images (один раз)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.court  -t decoder-court:latest .
DOCKER_BUILDKIT=1 docker build -f Dockerfile.detect -t decoder-detect:latest .
DOCKER_BUILDKIT=1 docker build -f Dockerfile.track  -t decoder-track:latest .

# 1) court
docker run --rm -v "$(pwd)":/app decoder-court:latest \
  --frames-dir /app/frames --output-json /app/court.json \
  --weights /app/weights/tcd.pth --sample-rate 5 --stabilize ema

# 2) detect
docker run --gpus all --rm -v "$(pwd)":/app decoder-detect:latest \
  detect --frames-dir /app/frames \
         --output-json /app/detections.json \
         --two-pass --nms-class-aware \
         --roi-json /app/court.json --roi-margin 8 \
         --detect-court

# 3) track (тенісні дефолти + appearance-refine автоматично)
docker run --gpus all --rm -v "$(pwd)":/app decoder-track:latest \
  track --detections-json /app/detections.json \
        --output-json /app/tracks.json \
        --frames-dir /app/frames \
        --fps 30 --min-score 0.28 \
        --pre-nms-iou 0.6 --pre-min-area-q 0.15 --pre-topk 3 --pre-court-gate \
        --p-match-thresh 0.55 --p-track-buffer 160 --reid-reuse-window 150 \
        --b-match-thresh 0.55 --b-track-buffer 150 \
        --stitch --stitch-iou 0.55 --stitch-gap 12 \
        --smooth ema --smooth-alpha 0.3

# 4) overlay preview (PNG + MP4)
docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
  -m src.draw_overlay \
    --mode track \
    --frames-dir /app/frames \
    --tracks-json /app/tracks.json \
    --output-dir /app/preview_tracks \
    --export-mp4 /app/preview_tracks.mp4 --fps 25 \
    --draw-court --draw-court-lines --roi-json /app/court.json

# 5) sanity metrics
docker run --rm -v "$(pwd)":/app decoder-track:latest \
  python tools/verify_tennis_defaults.py --tracks-json /app/tracks.json
```
