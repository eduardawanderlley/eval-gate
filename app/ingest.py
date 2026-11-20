"""Carregamento e chunking dos documentos.

O corte é guiado por parágrafos: juntamos parágrafos até atingir o tamanho
alvo. Um parágrafo maior que o limite é quebrado em janelas com sobreposição.
Não é a melhor estratégia para todo tipo de documento (leis com artigos
numerados se beneficiariam de cortes por artigo), mas é previsível e serve
de baseline para comparar outras estratégias.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


def load_corpus(corpus_dir: str | Path) -> dict[str, str]:
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus não encontrado em {root}. Rode `make corpus`.")
    docs = {}
    for path in sorted(root.iterdir()):
        if path.suffix in {".txt", ".md"}:
            docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise ValueError(f"Nenhum .txt ou .md em {root}.")
    return docs


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Quebra um trecho maior que `size` em janelas que se sobrepõem."""
    step = size - overlap
    pieces = []
    for start in range(0, len(text), step):
        window = text[start:start + size]
        if window.strip():
            pieces.append(window)
        if start + size >= len(text):
            break
    return pieces


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk_overlap precisa ser menor que chunk_size")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long(para, size, overlap))
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
            continue

        chunks.append(current)
        # Os últimos caracteres do chunk anterior seguem como contexto do próximo.
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{para}".strip()
        if len(current) > size:
            current = para

    if current:
        chunks.append(current)
    return chunks


def ingest(corpus_dir: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in load_corpus(corpus_dir).items():
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            chunks.append(Chunk(chunk_id=f"{source}#{i:04d}", source=source, text=piece))
    return chunks
