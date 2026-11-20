"""Baixa o texto da LGPD (Lei 13.709/2018) do Planalto e salva em data/corpus/lgpd.txt.

Uso:
    python scripts/fetch_corpus.py

Se o endereço mudar ou a rede bloquear o acesso, basta colocar o texto da lei
em data/corpus/lgpd.txt manualmente. O pipeline só lê arquivos .txt e .md.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

URL = "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/L13709.htm"
OUT = Path("data/corpus/lgpd.txt")

BLOCK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "tr", "li"}


class _TextExtractor(HTMLParser):
    """Extrai texto de HTML, ignorando scripts e estilos e preservando quebras de bloco."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    # Páginas do Planalto costumam vir em cp1252 mesmo sem declarar charset.
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def clean(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> int:
    try:
        html = fetch(URL)
    except OSError as exc:
        print(f"Falha ao baixar {URL}: {exc}", file=sys.stderr)
        print("Coloque o texto manualmente em data/corpus/lgpd.txt.", file=sys.stderr)
        return 1

    parser = _TextExtractor()
    parser.feed(html)
    text = clean("".join(parser.parts))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT}: {len(text):,} caracteres")
    return 0


if __name__ == "__main__":
    sys.exit(main())
