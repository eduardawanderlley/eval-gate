"""Verificador de sustentação (detector de alucinação).

Três etapas, cada uma com saída verificável:

1. Decomposição: a resposta vira uma lista de afirmações atômicas (LLM).
2. Evidência: para cada afirmação, o juiz escolhe o chunk que a sustenta e
   copia um trecho literal. Sem evidência, a afirmação não conta como sustentada.
3. Checagem determinística: o trecho citado precisa existir no chunk apontado,
   comparando texto normalizado. Juízes LLM às vezes "confirmam" uma afirmação
   e inventam a citação; esta etapa pega exatamente esse caso.

faithfulness = afirmações sustentadas / afirmações totais.
Resposta sem afirmações (recusa, por exemplo) não tem score: devolvemos None,
e quem agrega decide como tratar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.ingest import Chunk
from app.llm import LLM
from app.retrieval import normalize
from evals.judges import parse_json

SPLIT_SYSTEM = (
    "Quebre a resposta em afirmações atômicas: cada uma deve conter uma única "
    "informação verificável. Ignore citações entre colchetes, como [lgpd.txt#0003]. "
    "Se a resposta for uma recusa, retorne lista vazia.\n"
    "Responda somente com JSON: {\"claims\": [\"...\", \"...\"]}"
)

SUPPORT_SYSTEM = (
    "Você verifica se uma afirmação é sustentada por algum dos trechos numerados. "
    "A afirmação só é sustentada se um trecho contém a informação, sem exigir "
    "conhecimento externo.\n"
    "Se for sustentada, copie em \"evidence\" um trecho LITERAL do trecho escolhido, "
    "com até 30 palavras.\n"
    "Responda somente com JSON: "
    "{\"supported\": true|false, \"chunk_index\": <número ou null>, \"evidence\": \"<trecho literal ou vazio>\"}"
)


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    source_id: str | None = None
    evidence: str = ""
    rejected_evidence: bool = False  # juiz disse "sustentada" mas a citação não existe no chunk


@dataclass
class FaithfulnessReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        if not self.verdicts:
            return None
        return sum(v.supported for v in self.verdicts) / len(self.verdicts)


def split_claims(llm: LLM, answer: str) -> list[str]:
    data = parse_json(llm.complete(SPLIT_SYSTEM, answer, max_tokens=400).text)
    return [c.strip() for c in data.get("claims", []) if c.strip()]


def _evidence_in(evidence: str, chunk_text: str) -> bool:
    """A citação precisa aparecer no chunk, ignorando caixa, acentos e espaços extras."""
    if not evidence:
        return False
    needle = " ".join(normalize(evidence).split())
    haystack = " ".join(normalize(chunk_text).split())
    return needle in haystack


def verify_claim(llm: LLM, claim: str, chunks: list[Chunk]) -> ClaimVerdict:
    numbered = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))
    user = f"Afirmação: {claim}\n\nTrechos:\n{numbered}"
    data = parse_json(llm.complete(SUPPORT_SYSTEM, user, max_tokens=300).text)

    if not data.get("supported"):
        return ClaimVerdict(claim=claim, supported=False)

    idx = data.get("chunk_index")
    evidence = (data.get("evidence") or "").strip()
    if idx is None or not 0 <= int(idx) < len(chunks):
        return ClaimVerdict(claim=claim, supported=False, rejected_evidence=True)

    chunk = chunks[int(idx)]
    if not _evidence_in(evidence, chunk.text):
        return ClaimVerdict(claim=claim, supported=False, rejected_evidence=True)
    return ClaimVerdict(claim=claim, supported=True, source_id=chunk.chunk_id, evidence=evidence)


def check_faithfulness(llm: LLM, answer: str, chunks: list[Chunk]) -> FaithfulnessReport:
    report = FaithfulnessReport()
    if not chunks:
        return report
    for claim in split_claims(llm, answer):
        report.verdicts.append(verify_claim(llm, claim, chunks))
    return report
