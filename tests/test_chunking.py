import pytest

from app.ingest import chunk_text


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Frase curta número {i}." for i in range(50))
    chunks = chunk_text(text, size=120, overlap=20)
    assert chunks
    assert all(len(c) <= 120 for c in chunks)


def test_short_paragraphs_stay_together():
    text = "Primeiro parágrafo.\n\nSegundo parágrafo."
    assert chunk_text(text, size=500, overlap=0) == ["Primeiro parágrafo.\n\nSegundo parágrafo."]


def test_oversized_paragraph_is_split_with_overlap():
    chunks = chunk_text("a" * 1000, size=300, overlap=50)
    # passo de 250 caracteres: inícios em 0, 250, 500 e 750
    assert len(chunks) == 4
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a" * 20, size=10, overlap=10)
