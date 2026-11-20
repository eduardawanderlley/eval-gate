"""Gate de regressão sobre a última execução real.

Pula quando não existe results/candidate.json, ou seja, quando ninguém rodou
`make eval` (caso típico do `make test` local e dos jobs de PR sem secret).
"""
import json
from pathlib import Path

import pytest
import yaml

from evals.gate import evaluate_gate

RESULT = Path("results/candidate.json")
BASELINE = Path("evals/baseline.json")
THRESHOLDS = Path("thresholds.yaml")


def test_candidate_passes_gate():
    if not RESULT.exists():
        pytest.skip("Sem results/candidate.json. Rode `make eval` antes.")
    candidate = json.loads(RESULT.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else None
    thresholds = yaml.safe_load(THRESHOLDS.read_text(encoding="utf-8"))
    failures = evaluate_gate(candidate, baseline, thresholds)
    assert not failures, "\n".join(failures)
