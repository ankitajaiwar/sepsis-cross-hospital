PY ?= python
export PYTHONPATH := src

.PHONY: install data run run-seeds run-fast active test serve docker

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m sepsis.download

run:
	$(PY) -m sepsis.pipeline --raw data/raw --out artifacts

run-seeds:
	$(PY) -m sepsis.pipeline --from-omop artifacts/omop --out artifacts --seeds 0,1,2

run-fast:
	$(PY) -m sepsis.pipeline --raw data/raw --out artifacts_fast --limit 2000 --fast

active:
	$(PY) -m sepsis.active --omop artifacts/omop --out artifacts_active

test:
	$(PY) -m pytest -q

serve:
	MODEL_DIR=artifacts/model uvicorn sepsis.api:app --port 8000

docker:
	docker build -t sepsis-api .
	docker run --rm -p 8000:8000 sepsis-api
