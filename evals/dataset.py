"""Carregamento e validação do golden set (evals/golden.jsonl).

Cada linha é um caso. Validamos na leitura para que um typo na categoria ou
um id repetido quebre a execução cedo, e não distorça as métricas em silêncio.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = {"resposta_direta", "multi_trecho", "fora_do_escopo"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_answer: str
    category: str
    source_docs: tuple[str, ...]

    @property
    def out_of_scope(self) -> bool:
        return self.category == "fora_do_escopo"


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        case = EvalCase(
            id=raw["id"],
            question=raw["question"],
            expected_answer=raw.get("expected_answer", ""),
            category=raw["category"],
            source_docs=tuple(raw.get("source_docs", [])),
        )
        if case.category not in CATEGORIES:
            raise ValueError(f"{path}:{lineno}: categoria inválida '{case.category}'")
        if case.id in seen:
            raise ValueError(f"{path}:{lineno}: id duplicado '{case.id}'")
        if not case.out_of_scope and not case.expected_answer:
            raise ValueError(f"{path}:{lineno}: caso dentro do escopo sem expected_answer")
        seen.add(case.id)
        cases.append(case)
    return cases
