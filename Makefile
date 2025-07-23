.PHONY: test enhance build

build:
	docker build -t decoder-poc .

test:
	pytest -vv

enhance:
	python -m src.frame_enhancer $(ARGS)