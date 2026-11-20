PYTHON ?= python3
EVAL_ARGS ?=

.PHONY: install corpus test smoke eval baseline agreement up clean

install:
	$(PYTHON) -m pip install -r requirements.txt

corpus:
	$(PYTHON) scripts/fetch_corpus.py

test:
	$(PYTHON) -m pytest -q

# Execução barata: 3 casos, para checar que a chave e o pipeline funcionam.
smoke:
	$(PYTHON) -m evals.run_eval --name smoke --limit 3

# Avalia a configuração atual (ou a que você passar em EVAL_ARGS) e aplica o gate.
# Exemplo: make eval EVAL_ARGS="--chunk-size 400 --chunk-overlap 40"
eval:
	$(PYTHON) -m evals.run_eval --name candidate $(EVAL_ARGS)
	$(PYTHON) -m evals.gate results/candidate.json --report results/report.md

# Regrava a baseline. Rode na main depois de um merge que mudou a qualidade de propósito.
baseline:
	$(PYTHON) -m evals.run_eval --name baseline
	cp results/baseline.json evals/baseline.json

agreement:
	$(PYTHON) -m evals.agreement results/candidate_claims.csv

up:
	docker compose up --build

clean:
	rm -rf results .cache .pytest_cache
