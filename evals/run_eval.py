"""Executa o pipeline sobre o golden set e grava as métricas em results/<name>.json.

Exemplos:
    python -m evals.run_eval --name smoke --limit 3
    python -m evals.run_eval --name chunk400 --chunk-size 400 --chunk-overlap 40

Além do JSON, grava results/<name>_claims.csv com as afirmações julgadas,
para a medição de concordância humano x juiz (ver evals/agreement.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

from app.answer import REFUSAL, Answer, answer_question
from app.config import PipelineConfig
from app.ingest import ingest
from app.llm import AnthropicLLM, CachedLLM
from app.retrieval import BM25Retriever
from evals.dataset import EvalCase, load_dataset
from evals.hallucination import check_faithfulness
from evals.judges import judge_context_precision, judge_refusal, judge_relevance


def score_case(case: EvalCase, answer: Answer, judge: CachedLLM) -> dict:
    # Recusa literal não precisa de juiz. Qualquer outra resposta passa pelo classificador.
    refused = answer.text == REFUSAL or judge_refusal(judge, case.question, answer.text)

    faith = None if refused else check_faithfulness(judge, answer.text, answer.contexts)
    # Sem chunks recuperados, precisão de contexto não se aplica (fica fora da média).
    ctx_precision = judge_context_precision(judge, case.question, answer.contexts) if answer.contexts else None

    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "answer": answer.text,
        "refused": refused,
        "refusal_correct": refused == case.out_of_scope,
        "faithfulness": faith.score if faith else None,
        "answer_relevance": judge_relevance(judge, case.question, answer.text),
        "context_precision": ctx_precision,
        "claims": [
            {"claim": v.claim, "supported": v.supported, "source_id": v.source_id}
            for v in (faith.verdicts if faith else [])
        ],
        "generation_cached": answer.generation.cached,
        "generation_latency_s": answer.generation.latency_s,
    }


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = round(pct / 100 * (len(ordered) - 1))
    return round(ordered[idx], 3)


def aggregate(rows: list[dict], llms: list[CachedLLM], cfg: PipelineConfig) -> dict:
    # Latência só das chamadas de geração que realmente foram ao modelo.
    latencies = [r["generation_latency_s"] for r in rows if not r["generation_cached"]]
    tokens_in = sum(llm.input_tokens for llm in llms)
    tokens_out = sum(llm.output_tokens for llm in llms)
    cost = (tokens_in * cfg.price_input_per_mtok + tokens_out * cfg.price_output_per_mtok) / 1_000_000
    hits = sum(llm.hits for llm in llms)
    misses = sum(llm.misses for llm in llms)

    return {
        "faithfulness": _mean(r["faithfulness"] for r in rows),
        "answer_relevance": _mean(r["answer_relevance"] for r in rows),
        "context_precision": _mean(r["context_precision"] for r in rows),
        "refusal_accuracy": _mean(float(r["refusal_correct"]) for r in rows),
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost, 4),
        "cache_hit_rate": round(hits / (hits + misses), 3) if hits + misses else None,
    }


def run(cfg: PipelineConfig, dataset_path: str | None = None, limit: int | None = None) -> dict:
    chunks = ingest(cfg.corpus_dir, cfg.chunk_size, cfg.chunk_overlap)
    retriever = BM25Retriever(chunks)
    cases = load_dataset(dataset_path or cfg.dataset_path)[:limit]

    gen = CachedLLM(AnthropicLLM(cfg.model), model=cfg.model, cache_dir=cfg.cache_dir)
    judge = CachedLLM(AnthropicLLM(cfg.judge_model), model=cfg.judge_model, cache_dir=cfg.cache_dir)

    rows = []
    for i, case in enumerate(cases, start=1):
        hits = [chunk for chunk, _ in retriever.search(case.question, cfg.top_k)]
        answer = answer_question(case.question, hits, gen, cfg.prompt_version)
        rows.append(score_case(case, answer, judge))
        print(f"[{i}/{len(cases)}] {case.id}", file=sys.stderr)

    return {
        "n_cases": len(cases),
        "config": cfg.to_dict(),
        "metrics": aggregate(rows, [gen, judge], cfg),
        "cases": rows,
    }


def write_claims_csv(result: dict, path: Path) -> None:
    """Exporta as afirmações para anotação humana. A coluna human_supported é preenchida à mão."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "claim", "judge_supported", "human_supported"])
        for case in result["cases"]:
            for claim in case["claims"]:
                writer.writerow([case["id"], claim["claim"], int(claim["supported"]), ""])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="candidate", help="vira results/<name>.json")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--prompt", dest="prompt_version", help="v1 ou v2")
    parser.add_argument("--dataset")
    parser.add_argument("--limit", type=int, help="avalia só os N primeiros casos")
    args = parser.parse_args(argv)

    overrides = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "prompt_version": args.prompt_version,
    }
    cfg = replace(PipelineConfig(name=args.name), **{k: v for k, v in overrides.items() if v is not None})

    result = run(cfg, dataset_path=args.dataset, limit=args.limit)

    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_claims_csv(result, out_dir / f"{cfg.name}_claims.csv")

    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
