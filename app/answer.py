"""Geração de respostas, com recusa explícita para o que não está nos documentos."""
from __future__ import annotations

from dataclasses import dataclass

from app.ingest import Chunk
from app.llm import LLM, LLMResponse

REFUSAL = "Não encontrei essa informação nos documentos disponíveis."

# Versões de prompt são chaves estáveis: o results/<name>.json registra qual foi usada.
PROMPTS = {
    "v1": (
        "Você responde perguntas usando somente os trechos fornecidos.\n"
        "- Cite a fonte entre colchetes, por exemplo [lgpd.txt#0003].\n"
        "- Se os trechos não contêm a resposta, responda exatamente: "
        f"\"{REFUSAL}\"\n"
        "- Não use conhecimento externo nem complete lacunas com suposições."
    ),
    "v2": (
        "Responda em no máximo três frases, usando somente os trechos fornecidos.\n"
        "- Cite a fonte entre colchetes, por exemplo [lgpd.txt#0003].\n"
        "- Se os trechos não contêm a resposta, responda exatamente: "
        f"\"{REFUSAL}\"\n"
        "- Não use conhecimento externo."
    ),
}


@dataclass
class Answer:
    question: str
    text: str
    contexts: list[Chunk]
    generation: LLMResponse


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)


def answer_question(question: str, contexts: list[Chunk], llm: LLM, prompt_version: str) -> Answer:
    if prompt_version not in PROMPTS:
        raise KeyError(f"Versão de prompt desconhecida: {prompt_version}")
    if not contexts:
        # Sem trechos recuperados não há o que sustentar a resposta. Recusa sem chamar o modelo.
        return Answer(question, REFUSAL, [], LLMResponse(text=REFUSAL))

    user = f"Trechos:\n{build_context(contexts)}\n\nPergunta: {question}"
    resp = llm.complete(PROMPTS[prompt_version], user, max_tokens=500)
    return Answer(question, resp.text.strip(), contexts, resp)
