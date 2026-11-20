"""Cliente de LLM com cache em disco.

Toda chamada passa pelo `CachedLLM`. A chave do cache é o hash de modelo,
system prompt, user prompt e max_tokens. Logo, reexecutar a suíte sem mudar
nada não gasta tokens, e uma mudança de prompt invalida só o que mudou.
Em CI o diretório de cache é persistido com actions/cache.

O SDK só é importado quando `AnthropicLLM` é instanciado, para que os testes
rodem sem a dependência e sem chave de API.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Dependências ausentes. Rode `make install`.") from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY não definida. Veja .env.example.")
        self._client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        start = time.perf_counter()
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,  # avaliação precisa ser reproduzível
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency = time.perf_counter() - start
        text = "".join(block.text for block in msg.content if block.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            latency_s=latency,
        )


class CachedLLM:
    def __init__(self, inner: LLM, model: str, cache_dir: str):
        self._inner = inner
        self._model = model
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.input_tokens = 0   # só chamadas reais contam; cache não custa nada
        self.output_tokens = 0

    def _key(self, system: str, user: str, max_tokens: int) -> str:
        payload = json.dumps([self._model, system, user, max_tokens], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete(self, system: str, user: str, max_tokens: int = 800) -> LLMResponse:
        path = self._dir / f"{self._key(system, user, max_tokens)}.json"
        if path.exists():
            self.hits += 1
            return LLMResponse(cached=True, **json.loads(path.read_text(encoding="utf-8")))

        self.misses += 1
        resp = self._inner.complete(system, user, max_tokens)
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        path.write_text(json.dumps({
            "text": resp.text,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_s": resp.latency_s,
        }, ensure_ascii=False), encoding="utf-8")
        return resp
