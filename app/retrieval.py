"""Recuperação lexical com BM25.

BM25 é o baseline deliberadamente simples: determinístico, sem custo de
embeddings e difícil de bater em texto jurídico com termos muito específicos
("controlador", "operador", "anonimização"). A limitação é conhecida: não
entende sinônimos. A interface `search` foi pensada para que trocar por busca
vetorial não obrigue a mexer no resto do pipeline.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from app.ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "o", "as", "os", "de", "do", "da", "dos", "das", "e", "em", "no", "na",
    "nos", "nas", "um", "uma", "por", "para", "com", "que", "ou", "se", "ao",
    "pelo", "pela", "como", "quais", "qual", "quando", "onde", "ser",
}


def normalize(text: str) -> str:
    """Minúsculas e sem acentos, para que 'Titular' e 'titular' casem."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in _STOPWORDS and len(t) > 1]


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._docs = [Counter(tokenize(c.text)) for c in chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avgdl = sum(self._lengths) / max(len(self._lengths), 1)

        df: Counter = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        q_terms = tokenize(query)
        scored = []
        for idx, doc in enumerate(self._docs):
            score = 0.0
            dl = self._lengths[idx]
            for term in q_terms:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += self._idf[term] * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.chunks[idx], score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
