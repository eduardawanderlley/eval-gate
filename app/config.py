"""Configuração do pipeline.

Os parâmetros que mudam entre experimentos (chunking, top-k, versão do prompt)
ficam no dataclass. Modelos e preços vêm do ambiente, para que o repositório
não precise conhecer chaves nem tabelas de preço, que mudam com frequência.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "candidate"
    chunk_size: int = 800          # alvo em caracteres
    chunk_overlap: int = 120       # sobreposição entre chunks consecutivos
    top_k: int = 4                 # chunks que entram no contexto
    prompt_version: str = "v1"
    model: str = field(default_factory=lambda: os.getenv("EVAL_MODEL", "claude-haiku-5-5"))
    judge_model: str = field(default_factory=lambda: os.getenv("EVAL_JUDGE_MODEL", "claude-haiku-5-5"))
    price_input_per_mtok: float = field(default_factory=lambda: float(os.getenv("EVAL_PRICE_INPUT_PER_MTOK", "0")))
    price_output_per_mtok: float = field(default_factory=lambda: float(os.getenv("EVAL_PRICE_OUTPUT_PER_MTOK", "0")))
    corpus_dir: str = "data/corpus"
    dataset_path: str = "evals/golden.jsonl"
    cache_dir: str = ".cache/llm"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)
