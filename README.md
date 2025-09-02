# Decoder PoC

This repository provides Dockerized utilities for frame extraction, object detection, tracking and overlay rendering for racket-sport videos.

## Images

### `decoder-detect`
- **Purpose:** run YOLOX-based player and ball detection.
- **GPU:** optional (`--gpus all` to enable).
- **Example:**
  ```bash
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.detect -t decoder-detect:latest .
  docker run --rm -v "$(pwd)":/app decoder-detect:latest \
    detect --frames-dir /app/frames --output-json /app/detections.json \
    --img-size 1536 --p-conf 0.30 --b-conf 0.05 \
    --p-nms 0.60 --b-nms 0.70 --two-pass --nms-class-aware --multi-scale on
  ```

### `decoder-track`
- **Purpose:** build player and ball tracks and render overlays.
- **GPU:** optional (`--gpus all` to enable).
- **Example:**
  ```bash
  DOCKER_BUILDKIT=1 docker build -f Dockerfile.track -t decoder-track:latest .
  docker run --rm -v "$(pwd)":/app decoder-track:latest \
    track --detections-json /app/detections.json --output-json /app/tracks.json \
    --fps 30 --min-score 0.10

  docker run --rm -v "$(pwd)":/app --entrypoint python decoder-track:latest \
    -m src.draw_overlay --mode track --frames-dir /app/frames \
    --tracks-json /app/tracks.json --output-dir /app/preview_tracks \
    --export-mp4 /app/preview_tracks.mp4 --fps 30 --crf 18 --id --label \
    --roi-json /app/roi.json
  ```

  The optional `roi.json` describes polygons to overlay for visualization only. Example:

  ```json
  {
    "polygons": [
      {
        "name": "ROI-1",
        "points": [[120,80],[1280,80],[1280,640],[120,640]],
        "color": "#00FF00",
        "thickness": 2,
        "fill_alpha": 0.12
      }
    ]
  }
  ```

## Development
- `make build` – build helper image.
- `make test` – run unit tests.
