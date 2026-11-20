"""Juízes baseados em LLM, escritos do zero.

Cada juiz tem critério explícito no prompt, responde em JSON e tem a saída
validada. Se o JSON não parsear ou a nota vier fora do intervalo, o juiz
levanta erro. Prefiro quebrar a execução a registrar um número inventado.
"""
from __future__ import annotations

import json
import re

from app.ingest import Chunk
from app.llm import LLM

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict:
    """Extrai o primeiro objeto JSON da resposta, mesmo que venha cercado de texto."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"Juiz não retornou JSON: {text[:200]!r}")
    return json.loads(match.group(0))


RELEVANCE_SYSTEM = (
    "Você avalia se uma resposta trata da pergunta feita.\n"
    "Escala de 0 a 5:\n"
    "5 = responde diretamente e de forma completa\n"
    "3 = responde parcialmente ou com desvios\n"
    "0 = não tem relação com a pergunta\n"
    "Uma recusa correta (\"Não encontrei essa informação...\") vale 5 quando a "
    "pergunta de fato não pode ser respondida com os documentos.\n"
    "Responda somente com JSON: {\"score\": <inteiro de 0 a 5>, \"reason\": \"<frase curta>\"}"
)

CONTEXT_SYSTEM = (
    "Você recebe uma pergunta e trechos numerados. Indique quais trechos contêm "
    "informação útil para responder à pergunta.\n"
    "Responda somente com JSON: {\"relevant\": [<índices dos trechos úteis>]}"
)

REFUSAL_SYSTEM = (
    "Você decide se a resposta é uma recusa, isto é, diz que a informação não "
    "está disponível ou que não consegue responder à pergunta.\n"
    "Responda somente com JSON: {\"refused\": true} ou {\"refused\": false}"
)


def judge_relevance(llm: LLM, question: str, answer: str) -> float:
    """Nota de 0.0 a 1.0 para o quanto a resposta trata da pergunta."""
    user = f"Pergunta: {question}\n\nResposta: {answer}"
    data = parse_json(llm.complete(RELEVANCE_SYSTEM, user, max_tokens=200).text)
    score = int(data["score"])
    if not 0 <= score <= 5:
        raise ValueError(f"Nota fora do intervalo 0-5: {score}")
    return score / 5


def judge_context_precision(llm: LLM, question: str, chunks: list[Chunk]) -> float:
    """Fração dos chunks recuperados que são úteis para a pergunta."""
    if not chunks:
        raise ValueError("Sem chunks para avaliar; trate o caso antes de chamar o juiz.")
    numbered = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))
    user = f"Pergunta: {question}\n\nTrechos:\n{numbered}"
    data = parse_json(llm.complete(CONTEXT_SYSTEM, user, max_tokens=150).text)
    relevant = {int(i) for i in data.get("relevant", []) if 0 <= int(i) < len(chunks)}
    return len(relevant) / len(chunks)


def judge_refusal(llm: LLM, question: str, answer: str) -> bool:
    user = f"Pergunta: {question}\n\nResposta: {answer}"
    data = parse_json(llm.complete(REFUSAL_SYSTEM, user, max_tokens=50).text)
    return bool(data["refused"])
