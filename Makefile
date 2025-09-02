.PHONY: build test enhance lint-imports draw draw-run-detect draw-run-track draw-run-mp4

build:
	DOCKER_BUILDKIT=1 docker build -t decoder-poc .

test:
	pytest -vv

enhance:
	python -m src.frame_enhancer $(ARGS)

lint-imports:
	@! git grep -nE '(^|[^A-Za-z_])(from|import)\s+yolox(\.|$$)' externals/ByteTrack | grep -v 'bytetrack_vendor' || \
	 (echo "ERROR: found stray 'yolox' imports inside vendor"; exit 1)

draw:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.draw -t decoder-draw:latest .

draw-run-detect:
	# example: detections only
	docker run --rm -v $(CURDIR):/app decoder-draw:latest \
	  --frames-dir /app/data/frames_min \
	  --detections-json /app/detections.json \
	  --output-dir /app/out/frames_viz --label --confidence-thr 0.1

draw-run-track:
	# example: tracks ByteTrack
	docker run --rm -v $(CURDIR):/app decoder-draw:latest \
	  --frames-dir /app/data/frames_min \
	  --tracks-json /app/tracks.json \
	  --output-dir /app/out/frames_viz --label --id

draw-run-mp4:
	# example: build mp4
	docker run --rm -v $(CURDIR):/app decoder-draw:latest \
	  --frames-dir /app/data/frames_min \
	  --tracks-json /app/tracks.json \
	  --output-dir /app/out/frames_viz \
	  --label --id --export-mp4 /app/out/tracks_viz.mp4 --fps 25 --crf 23

