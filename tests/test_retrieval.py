from app.ingest import Chunk
from app.retrieval import BM25Retriever, tokenize


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(chunk_id=f"d#{i}", source="d", text=t) for i, t in enumerate(texts)]


def test_tokenize_drops_accents_and_stopwords():
    assert tokenize("O Titular de Dados é Pessoa") == ["titular", "dados", "pessoa"]
    assert tokenize("Anonimização") == ["anonimizacao"]


def test_most_relevant_chunk_ranks_first():
    chunks = _chunks(
        "O prazo de guarda dos registros é de seis meses.",
        "Dado pessoal sensível inclui dado genético e biométrico.",
        "A ANPD fiscaliza o cumprimento da lei.",
    )
    top = BM25Retriever(chunks).search("dado genético biométrico", top_k=2)
    assert top[0][0].chunk_id == "d#1"


def test_query_without_matches_returns_nothing():
    retriever = BM25Retriever(_chunks("texto sobre gatos e cachorros"))
    assert retriever.search("imposto de renda", top_k=3) == []
