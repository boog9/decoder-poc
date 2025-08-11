.PHONY: detect-image track-image detect track pipeline clean-images smoke \
        build test enhance lint-imports

# Default YOLOX ref and Docker build settings.
YOLOX_REF ?= 0.3.0
DOCKER_BUILDKIT ?= 1
PWD_ABS := $(shell pwd)

detect-image:
	@docker image inspect decoder-detect:latest >/dev/null 2>&1 || \
	  (echo "==> Building decoder-detect (YOLOX_REF=$(YOLOX_REF))..." && \
   DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -f Dockerfile.detect \
     --build-arg YOLOX_REF=$(YOLOX_REF) -t decoder-detect:latest .)

track-image:
	@docker image inspect decoder-track:latest >/dev/null 2>&1 || \
	  (echo "==> Building decoder-track..." && \
   DOCKER_BUILDKIT=$(DOCKER_BUILDKIT) docker build -f Dockerfile.track \
     -t decoder-track:latest .)

detect: detect-image
	@echo "==> Running detection..."
	docker run --gpus all --rm -v $(PWD_ABS):/app decoder-detect:latest \
  detect --frames-dir /app/frames --output-json /app/detections.json \
  --model yolox-x --img-size 1280

track: track-image
	@echo "==> Running tracking..."
	docker run --gpus all --rm -v $(PWD_ABS):/app decoder-track:latest \
  track --detections-json /app/detections.json --output-json /app/tracks.json

pipeline: detect track
	@echo "==> Done. Outputs: detections.json, tracks.json"

smoke: detect-image track-image
	@echo "==> Smoke: building only (no GPU run in CI)."
	@echo "   decoder-detect and decoder-track images exist."

clean-images:
	@echo "==> Removing local decoder images..."
	-@docker rmi -f decoder-detect:latest decoder-track:latest 2>/dev/null || true

# Legacy targets retained for development.
build:
	docker build -t decoder-poc .

test:
	pytest -vv

enhance:
	python -m src.frame_enhancer $(ARGS)

lint-imports:
	@! git grep -nE '(^|[^A-Za-z_])(from|import)\s+yolox(\.|$$)' externals/ByteTrack | grep -v 'bytetrack_vendor' || \
 (echo "ERROR: found stray 'yolox' imports inside vendor"; exit 1)

