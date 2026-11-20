"""LLM falso para os testes. Devolve respostas enfileiradas, na ordem das chamadas."""
from __future__ import annotations

from app.llm import LLMResponse


class FakeLLM:
    def __init__(self, responses: list[str] | None = None, default: str = ""):
        self._responses = list(responses or [])
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        self.calls.append((system, user))
        text = self._responses.pop(0) if self._responses else self._default
        return LLMResponse(text=text)
