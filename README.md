<div align="center">

# Eval Gate

**Automated RAG Evaluation & CI Quality Gate Pipeline**

*Versioned evaluation datasets, LLM-as-a-Judge metrics, deterministic hallucination verification, and automated Pull Request gating via GitHub Actions.*

[![CI](https://github.com/eduardawanderlley/eval-gate/actions/workflows/eval.yml/badge.svg)](https://github.com/eduardawanderlley/eval-gate/actions/workflows/eval.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[🇺🇸 English](#english-documentation) • [🇧🇷 Português](#documentação-em-português)

</div>

---

## English Documentation

### Overview

Building a Retrieval-Augmented Generation (RAG) prototype is straightforward; guaranteeing that chunking tweaks, prompt rewrites, or retrieval parameter changes do not silently degrade answer quality is the real engineering challenge.

**Eval Gate** treats AI evaluation as first-class software engineering: versioned golden datasets in Git, deterministic metric calculation on every change, and a **CI Quality Gate** in GitHub Actions that automatically blocks Pull Requests whenever quality regresses beyond configured thresholds.

While the reference corpus uses Brazilian data privacy legislation (**LGPD — Law 13.709/2018**), the pipeline is completely domain-agnostic—swapping the corpus and golden dataset adapts the framework to any knowledge base.

---

### Architecture & Evaluation Pipeline

Every query runs through a lexical BM25 retriever and a constrained generator with explicit refusal instructions. The output is evaluated across four orthogonal quality dimensions before hitting the CI gate:

```text
Query ──► BM25 (top-k chunks) ──► Constrained Generation (with refusal) ──► Response
                                            │
          ┌─────────────────────────────────┼──────────────────────────────┐
          ▼                                 ▼                              ▼
   Faithfulness                    Answer Relevance               Context Precision
   (Claim × Literal Evidence)       (LLM Judge, 0–5)              (Useful Chunks / top-k)
          │
          └──────────── Refusal Accuracy (Refuses iff out-of-scope) ───────┐
                                                                            ▼
                        CI Gate: Absolute Floor + Baseline Regression ──► Pass / Fail
```

---

### Evaluation Metrics & Telemetry

| Metric | Core Question | Calculation Methodology | Known Limitations |
| :--- | :--- | :--- | :--- |
| **`faithfulness`** | Is the answer strictly grounded in retrieved chunks? | Supported claims / total extracted claims. Every claim requires a verbatim quote verified deterministically in the source chunk. | Sensitive to LLM claim decomposition granularity; poorly split claims may receive lenient verdicts. |
| **`answer_relevance`** | Does the response directly address the user query? | LLM-as-a-Judge score on a `0–5` scale, normalized to `[0, 1]`. | LLM judges exhibit verbosity bias (favoring longer answers), requiring human-judge calibration. |
| **`context_precision`** | Did retrieval surface relevant context? | Useful chunks / total retrieved chunks (`top-k`). Queries returning zero chunks are excluded from the mean. | Does not capture *recall* (cannot penalize relevant documents missed by the retriever). |
| **`refusal_accuracy`** | Does the system abstain when—and only when—appropriate? | Proportion of cases where model refusal aligns with the dataset ground-truth label (`fora_do_escopo`). | Partial refusals (answering part of a query while abstaining on another) depend on classifier sensitivity. |

**Operational Telemetry:** Alongside quality scores, every report tracks **p50/p95 generation latency** (excluding cache hits), **token consumption**, **estimated API cost** (configured via `.env`), and **disk cache hit rate**.

---

### The CI Quality Gate (`thresholds.yaml`)

The gate enforces two non-negotiable policies prior to merging code:

1. **Absolute Floor:** Every gated metric must exceed its minimum threshold, regardless of prior baseline performance.
2. **Regression Tolerance:** No metric may drop beyond the allowed delta compared to the `main` branch reference stored in `evals/baseline.json`.

> **Engineering Principle:** Initial thresholds are conservative and calibrated on the first `main` run. Never lower a threshold merely to unblock a PR—investigate the root cause of the regression instead.

---

### Key Architectural Decisions

- **Deterministic Lexical Retrieval (BM25):** Chosen over dense embeddings for zero-cost CPU execution, full determinism, and high precision on exact legal terminology (e.g., *"controlador"*, *"operador"*, *"anonimização"*). The retriever interface is decoupled to allow seamless A/B benchmarking against dense vector embeddings.
- **Role Separation (Generator vs. Judge):** Configurable via `EVAL_JUDGE_MODEL` to prevent self-enhancement bias (models rating their own outputs higher).
- **Anti-Hallucination Quote Verification:** When a judge marks a claim as supported, it must output a literal substring from the context. The verifier deterministically checks substring existence in the chunk—fabricated citations are automatically downgraded to *unsupported*.
- **Zero-Context Short-Circuit Refusal:** When BM25 returns zero chunks, the pipeline returns a deterministic refusal without invoking the LLM, eliminating hallucination risk and reducing inference cost.
- **Content-Addressed Disk Cache & `temperature=0`:** LLM calls are cached by `(model, prompt_hash, max_tokens)`. Re-running unchanged suites incurs zero API cost, while prompt edits invalidate only affected calls.
- **API-Free Unit & Regression Test Suite:** `FakeLLM` (`tests/fakes.py`) serves deterministic queued responses, enabling fast CI verification of judges, claim verifiers, and gate logic without external network calls.

---

### Quickstart

Requires **Python 3.12+** and a POSIX shell (or **Docker**).

```bash
make install          # Install dependencies
make corpus           # Download LGPD text to data/corpus/lgpd.txt
make test             # Run unit test suite (offline, no API key required)

cp .env.example .env  # Configure ANTHROPIC_API_KEY and optional token pricing
make smoke            # Run 3-case smoke test to validate credentials & pipeline
make baseline         # Evaluate current state and persist evals/baseline.json
make eval             # Run full evaluation and enforce CI gate
```

#### Experimenting with Hyperparameters

Pass runtime overrides via `EVAL_ARGS` to benchmark variants against `baseline.json`:

```bash
make eval EVAL_ARGS="--chunk-size 400 --chunk-overlap 40"
make eval EVAL_ARGS="--prompt v2 --top-k 6"
```

#### Running with Docker

```bash
docker compose up --build
```

Executes `make eval` inside a reproducible container while persisting `.cache/` and `results/` to the host filesystem. Summary reports are written to `results/report.md` and granular per-case traces to `results/<run_name>.json`.

---

### Judge-Human Alignment (Cohen's Kappa)

An LLM judge is only trustworthy if its verdicts align with domain experts. Each `make eval` run exports `results/candidate_claims.csv` containing all evaluated claims. Annotate 30–50 rows in the `human_supported` column (`1` or `0`) blindly (without viewing `judge_supported`), then compute inter-annotator agreement:

```bash
make agreement
```

The script reports raw percentage agreement alongside **Cohen's Kappa ($\kappa$)**, accounting for chance agreement on imbalanced claim distributions.

---

### Benchmark Results & Regression Case Study

<!-- Populate strictly with outputs from `make baseline` and `make eval`. -->

| Metric | Baseline | Candidate | Gate Floor |
| :--- | :---: | :---: | :---: |
| `faithfulness` | — | — | `0.80` |
| `answer_relevance` | — | — | `0.70` |
| `context_precision` | — | — | `0.50` |
| `refusal_accuracy` | — | — | `0.85` |

- **Latency & Cost per Query:** *To be populated from `results/report.md`.*
- **Judge × Human Agreement:** *To be populated from `make agreement` ($N$ annotated claims).*

#### Reproducing a Controlled Regression

To observe the gate catching a real degradation, reduce chunk size aggressively so legal context is fragmented:

```bash
make eval EVAL_ARGS="--chunk-size 150 --chunk-overlap 20"
```

Splitting clauses across tiny chunks severs claims from supporting evidence, driving `faithfulness` and `context_precision` below the gate floor and failing the build with a detailed diff report.

---

### Known Limitations & Roadmap

- **Golden Dataset Scale:** The initial seed contains 12 curated cases suitable for pipeline validation. Production statistical significance requires expanding to **100–150 cases** covering single-hop, multi-hop synthesis, and out-of-scope queries, plus confidence intervals.
- **Lexical Vocabulary Mismatch:** BM25 does not capture semantic synonyms (e.g., *"dado de saúde"* vs. *"dado referente à saúde"*), impacting `context_precision` and motivating a hybrid BM25 + dense embedding experiment.
- **Structure-Aware Chunking:** Currently splits by paragraph; legal statutes and contracts benefit from article/section-aware hierarchical chunking.

---

### Repository Structure

```text
eval-gate/
├── app/                      # System Under Test (SUT)
│   ├── config.py             # Experiment configuration & hyperparameters
│   ├── ingest.py             # Document loading & chunking
│   ├── retrieval.py          # BM25 lexical retriever
│   ├── llm.py                # API client & content-addressed disk cache
│   └── answer.py             # Constrained generation, refusal logic & prompts
├── evals/                    # Evaluation & CI Gating Layer
│   ├── golden.jsonl          # Versioned ground-truth dataset
│   ├── dataset.py            # Dataset loader & schema validation
│   ├── judges.py             # Answer relevance, context & refusal judges
│   ├── hallucination.py      # Claim decomposition & deterministic quote verifier
│   ├── run_eval.py           # Evaluation runner & metric aggregator
│   ├── gate.py               # Floor/regression evaluator & CI exit codes
│   ├── report.py             # Markdown & JSON report generator
│   ├── agreement.py          # Cohen's Kappa human-judge alignment script
│   └── baseline.json         # Reference metrics from main branch
├── tests/                    # Offline unit & regression tests
├── scripts/fetch_corpus.py   # Corpus ingestion script
├── data/corpus/              # Source documents
├── thresholds.yaml           # CI gate floors & regression tolerances
├── .github/workflows/eval.yml
├── Makefile
├── Dockerfile
└── docker-compose.yml
```

### License

Distributed under the **MIT License**. See `LICENSE` for details.

---
---

## Documentação em Português

### Visão Geral

Construir um protótipo de RAG (*Retrieval-Augmented Generation*) que responde perguntas sobre documentos é relativamente simples. O verdadeiro desafio de engenharia é saber, após alterar o tamanho dos *chunks* ou reescrever um *prompt*, se o sistema melhorou ou piorou — e impedir que mudanças aparentemente inofensivas degradem a qualidade das respostas em produção.

O **Eval Gate** trata a avaliação de IA como código de primeira classe: um *dataset* de casos versionado no repositório, métricas calculadas a cada alteração e um **Quality Gate no GitHub Actions** que bloqueia automaticamente o *Pull Request* sempre que alguma métrica ultrapassa os limites tolerados.

O domínio de referência utiliza a **LGPD (Lei 13.709/2018)**, mas o pipeline é totalmente agnóstico ao domínio — basta substituir o corpus e o *golden set* para aplicá-lo a qualquer base de conhecimento.

---

### Arquitetura e Pipeline de Avaliação

Cada pergunta passa por uma busca lexical BM25 e por um gerador restrito ao contexto, com instrução explícita de recusa quando a evidência não estiver presente. Em seguida, a resposta é avaliada por métricas independentes antes de passar pelo gate de CI:

```text
pergunta ──► BM25 (top-k chunks) ──► geração (com recusa) ──► resposta
                                            │
          ┌─────────────────────────────────┼──────────────────────────────┐
          ▼                                 ▼                              ▼
   faithfulness                    answer relevance              context precision
   (afirmação × evidência)           (juiz LLM, nota 0–5)      (chunks úteis / top-k)
          │
          └──────────── refusal accuracy (recusa só quando deve) ──────────┐
                                                                            ▼
                             gate: piso absoluto + regressão vs baseline ──► CI
```

---

### Métricas e Telemetria

| Métrica | Pergunta que responde | Como é calculada | Limitação conhecida |
| :--- | :--- | :--- | :--- |
| **`faithfulness`** | A resposta é sustentada pelos trechos recuperados? | Afirmações sustentadas divididas pelo total de afirmações. Cada afirmação exige citação literal verificada deterministicamente no chunk. | Depende da granularidade da decomposição feita pelo LLM; afirmações mal divididas podem ser julgadas com folga. |
| **`answer_relevance`** | A resposta trata da pergunta feita? | Juiz LLM em escala de `0` a `5`, normalizada para `[0, 1]`. | Juízes LLM tendem a favorecer respostas longas (*verbosity bias*), exigindo calibração com humanos. |
| **`context_precision`** | O retrieval trouxe contexto útil? | Chunks julgados úteis divididos pelos chunks recuperados (`top-k`). Casos sem chunks ficam fora da média. | Não mede *recall*: não penaliza trechos relevantes que deixaram de ser recuperados. |
| **`refusal_accuracy`** | O sistema recusa quando deve, e apenas quando deve? | Proporção de casos em que a recusa coincide com o rótulo esperado (`fora_do_escopo`). | Recusas parciais (que respondem parte da pergunta e negam outra) dependem da sensibilidade do classificador. |

**Telemetria Operacional:** Além das métricas de qualidade, o relatório registra **latência (p50 e p95)** das chamadas que foram ao modelo (excluindo cache), **tokens consumidos**, **custo estimado** (via `.env`) e **taxa de acerto do cache**.

---

### O Gate de Qualidade (`thresholds.yaml`)

O gate aplica duas regras obrigatórias antes de permitir o *merge* de um PR:

1. **Piso Absoluto:** Cada métrica monitorada deve estar acima do valor mínimo configurado, independentemente do histórico anterior.
2. **Tolerância de Regressão:** Nenhuma métrica pode cair além da margem permitida em relação à *baseline* da branch `main` salva em `evals/baseline.json`.

> **Regra de ouro:** Os limites iniciais são conservadores e calibrados na primeira execução real na `main`. Nunca rebaixe um limite apenas para destravar um PR — investigue a causa da regressão.

---

### Decisões Arquiteturais

- **Busca Lexical Determinística (BM25):** Escolhida no lugar de busca vetorial por ter custo zero de infraestrutura, determinismo total e alta precisão em termos jurídicos exatos (ex.: *"controlador"*, *"operador"*, *"anonimização"*). A interface de retrieval foi isolada para permitir experimentação futura com *embeddings*.
- **Separação de Papéis (Gerador vs. Juiz):** Configurável via `EVAL_JUDGE_MODEL` para evitar viés de autoavaliação (*self-enhancement bias*), no qual um modelo atribui notas maiores às próprias respostas.
- **Verificação Determinística de Citações:** Quando o juiz afirma que uma declaração é sustentada, ele precisa apontar um trecho literal do contexto. O verificador valida deterministicamente se a *substring* existe no chunk — citações inventadas pelo juiz são reclassificadas como não sustentadas.
- **Recusa em Caminho Curto (*Short-Circuit*):** Se nenhum chunk for recuperado, o pipeline retorna a recusa diretamente sem acionar o LLM, economizando custo e eliminando o risco de alucinação sem contexto.
- **Cache em Disco por Hash e `temperature=0`:** Todas as chamadas ao LLM passam por cache indexado pelo hash de `(modelo, prompt, max_tokens)`. Reexecutar suítes sem alterações tem custo zero, e mudanças de prompt invalidam apenas os casos afetados.
- **Testes Unitários Offline (`FakeLLM`):** O componente `FakeLLM` em `tests/fakes.py` devolve respostas enfileiradas, permitindo validar juízes, verificador de sustentação e gate no CI sem depender de chaves de API.

---

### Como Rodar

Os comandos assumem **Python 3.12+** e shell POSIX (ou **Docker**).

```bash
make install          # Instala as dependências
make corpus           # Baixa a LGPD do Planalto para data/corpus/lgpd.txt
make test             # Executa os testes unitários (offline, sem API)

cp .env.example .env  # Preencha ANTHROPIC_API_KEY e, opcionalmente, os preços por token
make smoke            # Roda 3 casos para validar credenciais e pipeline
make baseline         # Avalia a versão atual e grava evals/baseline.json
make eval             # Executa avaliação completa e aplica o gate
```

#### Comparando Variantes com a Baseline

Passe parâmetros em `EVAL_ARGS` para testar hipóteses contra `baseline.json`:

```bash
make eval EVAL_ARGS="--chunk-size 400 --chunk-overlap 40"
make eval EVAL_ARGS="--prompt v2 --top-k 6"
```

#### Execução via Docker

```bash
docker compose up --build
```

Executa `make eval` em container isolado, persistindo `.cache/` e `results/` no host. O relatório resumido é gerado em `results/report.md` e o detalhamento completo caso a caso em `results/<nome>.json`.

---

### Concordância entre Juiz e Humano (Kappa de Cohen)

Um juiz LLM só é confiável se concordar com avaliadores humanos. A execução de `make eval` gera `results/candidate_claims.csv` com todas as afirmações julgadas. Preencha a coluna `human_supported` (`1` ou `0`) em uma amostra de 30 a 50 linhas às cegas (sem consultar `judge_supported`) e execute:

```bash
make agreement
```

O script calcula a concordância simples e o **Kappa de Cohen ($\kappa$)**, descontando a concordância esperada ao acaso.

---

### Resultados e Caso de Regressão

<!-- Preencher exclusivamente com saídas de `make baseline` e `make eval`. -->

| Métrica | Baseline | Candidato | Piso do Gate |
| :--- | :---: | :---: | :---: |
| `faithfulness` | — | — | `0.80` |
| `answer_relevance` | — | — | `0.70` |
| `context_precision` | — | — | `0.50` |
| `refusal_accuracy` | — | — | `0.85` |

- **Custo e latência por consulta:** *A preencher a partir de `results/report.md`.*
- **Concordância juiz × humano:** *A preencher com o resultado de `make agreement` ($N$ afirmações anotadas).*

#### Reproduzindo uma Regressão Controlada

Para observar o gate bloqueando uma degradação real, reduza drasticamente o tamanho dos chunks para fragmentar o contexto jurídico:

```bash
make eval EVAL_ARGS="--chunk-size 150 --chunk-overlap 20"
```

Chunks excessivamente pequenos separam as afirmações de suas evidências, derrubando `faithfulness` e `context_precision` abaixo do piso configurado e reprovando o CI.

---

### Limitações Conhecidas

- **Tamanho do Golden Set:** O conjunto inicial possui 12 casos curados, adequado para validar o pipeline, mas insuficiente para significância estatística entre prompts (recomenda-se expandir para **100–150 casos** com intervalos de confiança).
- **Descasa­mento Lexical (BM25):** Perguntas que usam sinônimos ausentes no texto (ex.: *"dado de saúde"* vs. *"dado referente à saúde"*) podem falhar no retrieval, motivando experimentos futuros com busca híbrida (BM25 + *embeddings*).
- **Chunking Estrutural:** O corte atual é por parágrafo; textos legais e contratos se beneficiam de *chunking* hierárquico guiado por artigos e seções.

---

### Licença

Distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais informações.
