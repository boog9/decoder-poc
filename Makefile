.PHONY: test enhance build lint-imports

build:
	docker build -t decoder-poc .

test:
	pytest -vv

enhance:
	python -m src.frame_enhancer $(ARGS)

lint-imports:
	@! git grep -nE '(^|[^A-Za-z_])(from|import)\s+yolox(\.|$$)' externals/ByteTrack | grep -v 'bytetrack_vendor' || \
	 (echo "ERROR: found stray 'yolox' imports inside vendor"; exit 1)
