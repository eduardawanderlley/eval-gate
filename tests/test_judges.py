import pytest

from app.ingest import Chunk
from evals.judges import judge_context_precision, judge_relevance, parse_json
from tests.fakes import FakeLLM


def _chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"d#{i}", source="d", text=f"trecho {i}")


def test_parse_json_extracts_object_from_prose():
    assert parse_json('Claro! {"score": 4, "reason": "ok"} Fim.')["score"] == 4


def test_parse_json_fails_loudly_without_json():
    with pytest.raises(ValueError):
        parse_json("não sei responder")


def test_relevance_is_normalized_to_unit_interval():
    llm = FakeLLM(responses=['{"score": 4, "reason": "boa"}'])
    assert judge_relevance(llm, "pergunta", "resposta") == 0.8


def test_relevance_rejects_out_of_range_score():
    llm = FakeLLM(responses=['{"score": 9, "reason": "?"}'])
    with pytest.raises(ValueError):
        judge_relevance(llm, "pergunta", "resposta")


def test_context_precision_is_fraction_of_useful_chunks():
    llm = FakeLLM(responses=['{"relevant": [0, 2]}'])
    chunks = [_chunk(i) for i in range(4)]
    assert judge_context_precision(llm, "pergunta", chunks) == 0.5
