"""Concordância entre o juiz LLM e a anotação humana sobre sustentação.

Fluxo: `make eval` gera results/candidate_claims.csv. Você preenche a coluna
human_supported (1 ou 0) em uma amostra de 30 a 50 linhas e roda:

    python -m evals.agreement results/candidate_claims.csv

Reporto concordância simples e kappa de Cohen. Kappa desconta a concordância
esperada por acaso, que é alta quando quase todas as afirmações são sustentadas.
Linhas sem human_supported são ignoradas.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    p_a = sum(a) / n
    p_b = sum(b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def load_pairs(path: Path) -> tuple[list[int], list[int]]:
    judge, human = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["human_supported"].strip():
                continue
            judge.append(int(row["judge_supported"]))
            human.append(int(row["human_supported"]))
    return judge, human


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("uso: python -m evals.agreement <arquivo_claims.csv>", file=sys.stderr)
        return 2

    judge, human = load_pairs(Path(args[0]))
    if not judge:
        print("Nenhuma linha anotada (preencha human_supported).", file=sys.stderr)
        return 1

    agree = sum(j == h for j, h in zip(judge, human)) / len(judge)
    kappa = cohen_kappa(judge, human)
    print(f"Afirmações anotadas: {len(judge)}")
    print(f"Concordância simples: {agree:.3f}")
    print(f"Kappa de Cohen: {kappa:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
