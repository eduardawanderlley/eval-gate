from evals.gate import evaluate_gate

THRESHOLDS = {
    "floors": {
        "faithfulness": 0.80,
        "answer_relevance": 0.70,
        "context_precision": 0.50,
        "refusal_accuracy": 0.85,
    },
    "max_regression": {
        "faithfulness": 0.03,
        "answer_relevance": 0.05,
        "context_precision": 0.05,
        "refusal_accuracy": 0.03,
    },
}


def _run(**metrics) -> dict:
    base = {"faithfulness": 0.9, "answer_relevance": 0.9, "context_precision": 0.9, "refusal_accuracy": 0.9}
    base.update(metrics)
    return {"metrics": base}


def test_passes_when_all_above_floor_and_no_baseline():
    assert evaluate_gate(_run(), None, THRESHOLDS) == []


def test_fails_below_absolute_floor():
    failures = evaluate_gate(_run(faithfulness=0.5), None, THRESHOLDS)
    assert len(failures) == 1
    assert "faithfulness" in failures[0]


def test_fails_on_regression_even_above_floor():
    baseline = _run(faithfulness=0.90)
    candidate = _run(faithfulness=0.84)  # acima do piso de 0.80, mas caiu 0.06 (limite 0.03)
    failures = evaluate_gate(candidate, baseline, THRESHOLDS)
    assert any("caiu" in f for f in failures)


def test_missing_metric_fails():
    failures = evaluate_gate(_run(context_precision=None), None, THRESHOLDS)
    assert any("context_precision" in f for f in failures)
