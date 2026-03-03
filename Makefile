PYTHON ?= python3

.PHONY: format lint test pipeline demo

format:
	$(PYTHON) -m pip install -q ruff
	ruff format src tests

lint:
	$(PYTHON) -m pip install -q ruff
	ruff check src tests

test:
	$(PYTHON) -m pytest -q

pipeline:
	$(PYTHON) -m refinery.cli ingest data/*.pdf
	$(PYTHON) -m refinery.cli build-index data/*.pdf

demo:
	bash scripts/demo_protocol.sh
