import json

from app.ingest import Chunk
from evals.hallucination import check_faithfulness, split_claims
from tests.fakes import FakeLLM

CHUNK = Chunk(
    chunk_id="lgpd.txt#0001",
    source="lgpd.txt",
    text=(
        "Art. 5º Dado pessoal é informação relacionada a pessoa natural "
        "identificada ou identificável."
    ),
)


def test_claim_with_real_evidence_is_supported():
    llm = FakeLLM(responses=[
        json.dumps({"claims": ["Dado pessoal é informação relacionada a pessoa natural."]}),
        json.dumps({
            "supported": True,
            "chunk_index": 0,
            "evidence": "Dado pessoal é informação relacionada a pessoa natural",
        }),
    ])
    report = check_faithfulness(llm, "Dado pessoal é informação relacionada a pessoa natural.", [CHUNK])
    assert report.score == 1.0
    assert report.verdicts[0].source_id == "lgpd.txt#0001"


def test_fabricated_evidence_is_rejected_even_if_judge_says_supported():
    llm = FakeLLM(responses=[
        json.dumps({"claims": ["Qualquer coisa."]}),
        json.dumps({"supported": True, "chunk_index": 0, "evidence": "trecho que não existe"}),
    ])
    report = check_faithfulness(llm, "Qualquer coisa.", [CHUNK])
    assert report.score == 0.0
    assert report.verdicts[0].rejected_evidence is True


def test_refusal_has_no_claims_and_no_score():
    llm = FakeLLM(responses=[json.dumps({"claims": []})])
    report = check_faithfulness(llm, "Não encontrei essa informação.", [CHUNK])
    assert report.score is None


def test_split_claims_ignores_blank_entries():
    llm = FakeLLM(responses=[json.dumps({"claims": ["A.", "   ", ""]})])
    assert split_claims(llm, "x") == ["A."]
