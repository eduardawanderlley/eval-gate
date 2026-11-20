"""Gate de qualidade: decide se uma execução pode seguir para merge.

Duas regras:

1. Piso absoluto. Cada métrica gated precisa estar acima do valor em thresholds.yaml.
2. Sem regressão. Nenhuma métrica pode cair mais que `max_regression` em relação
   à baseline commitada em evals/baseline.json (atualizada na main).

Código de saída 0 se passou, 1 se reprovou. O relatório vai para stdout e,
opcionalmente, para um arquivo que o workflow publica como comentário no PR.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from evals.report import render_markdown

GATED_METRICS = ["faithfulness", "answer_relevance", "context_precision", "refusal_accuracy"]


def evaluate_gate(candidate: dict, baseline: dict | None, thresholds: dict) -> list[str]:
    failures: list[str] = []
    cand = candidate["metrics"]
    floors = thresholds["floors"]
    max_drop = thresholds["max_regression"]

    for metric in GATED_METRICS:
        value = cand.get(metric)
        if value is None:
            failures.append(f"{metric}: sem valor (nenhum caso avaliável)")
            continue
        if value < floors[metric]:
            failures.append(f"{metric} = {value:.3f}, abaixo do piso {floors[metric]:.3f}")
        if baseline:
            base = baseline["metrics"].get(metric)
            if base is not None and base - value > max_drop[metric]:
                failures.append(
                    f"{metric} caiu {base - value:.3f} em relação à baseline "
                    f"({base:.3f} para {value:.3f}; limite {max_drop[metric]:.3f})"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avalia um resultado contra piso e baseline.")
    parser.add_argument("candidate", help="ex.: results/candidate.json")
    parser.add_argument("--baseline", default="evals/baseline.json")
    parser.add_argument("--thresholds", default="thresholds.yaml")
    parser.add_argument("--report", help="grava o markdown neste caminho")
    args = parser.parse_args(argv)

    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    base_path = Path(args.baseline)
    baseline = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else None
    thresholds = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))

    failures = evaluate_gate(candidate, baseline, thresholds)
    markdown = render_markdown(candidate, baseline, thresholds, failures)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
