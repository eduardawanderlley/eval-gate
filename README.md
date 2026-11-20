# Eval Gate

Pipeline de avaliação para sistemas RAG, com bloqueio automático de pull request quando a qualidade cai.

Montar um RAG que responde perguntas sobre documentos é relativamente fácil. Difícil é saber, depois de trocar o tamanho dos chunks ou reescrever o prompt, se o sistema ficou melhor ou pior, e impedir que uma mudança aparentemente inocente degrade as respostas sem que ninguém perceba. Este projeto trata a avaliação como parte do código: um conjunto de casos versionado no repositório, métricas calculadas a cada mudança e um gate no GitHub Actions que reprova o PR quando uma métrica passa do limite.

O domínio de exemplo é a LGPD (Lei 13.709/2018), mas nada no pipeline depende dela. Trocar o corpus e o golden set é suficiente para aplicá-lo a outro domínio.

![CI](https://github.com/SEU_USUARIO/eval-gate/actions/workflows/eval.yml/badge.svg)

## Como funciona

A pergunta passa por uma busca lexical BM25, que devolve os chunks mais relevantes. O modelo gera a resposta usando apenas esses trechos, com instrução explícita para recusar quando a informação não estiver lá. Em seguida, cada resposta é avaliada por métricas independentes, e o conjunto das métricas alimenta o gate.

```
pergunta ──► BM25 (top-k chunks) ──► geração (com recusa) ──► resposta
                                            │
          ┌─────────────────────────────────┼──────────────────────────────┐
          ▼                                 ▼                              ▼
   faithfulness                    answer relevance              context precision
   (afirmação × evidência)           (juiz, nota 0–5)          (chunks úteis / top-k)
          │
          └──────────── refusal accuracy (recusa só quando deve) ──────────┐
                                                                            ▼
                             gate: piso absoluto + regressão vs baseline ──► CI
```

## Métricas

Cada métrica foi escolhida para responder a uma pergunta específica, e cada uma tem limitações que convém conhecer antes de confiar nela.

| Métrica | Pergunta que responde | Como é calculada | Limitação conhecida |
|---|---|---|---|
| `faithfulness` | A resposta é sustentada pelos trechos recuperados? | Afirmações sustentadas dividido pelo total de afirmações. Cada afirmação precisa de uma citação literal que é verificada no chunk indicado. | Depende da decomposição feita pelo LLM. Uma afirmação mal quebrada pode ser julgada com folga. |
| `answer_relevance` | A resposta trata da pergunta feita? | Juiz LLM com escala de 0 a 5, normalizada para 0 a 1. | Juízes LLM tendem a ser generosos com respostas longas. Por isso medimos a concordância com humanos (ver abaixo). |
| `context_precision` | O retrieval trouxe coisa útil? | Chunks julgados úteis dividido pelos chunks recuperados. Casos sem nenhum chunk ficam de fora da média. | Não mede recall: não diz o que deveria ter sido recuperado e não foi. |
| `refusal_accuracy` | O sistema recusa quando deve, e só quando deve? | Proporção de casos em que recusa coincide com a categoria do caso (`fora_do_escopo`). | Uma recusa parcial, que responde uma parte e nega outra, depende do classificador de recusa. |

Além das métricas de qualidade, o relatório registra latência (p50 e p95 das chamadas de geração que foram ao modelo, sem contar cache), tokens consumidos, custo estimado e taxa de acerto do cache. O custo só é calculado se você definir os preços por milhão de tokens em `.env`.

## O gate

O gate aplica duas regras, ambas definidas em `thresholds.yaml`.

A primeira é o piso absoluto. Cada métrica gated precisa estar acima do valor configurado, mesmo que a execução anterior tenha sido pior. A segunda é a regressão. Nenhuma métrica pode cair mais que a tolerância configurada em relação à baseline, que é o resultado da main gravado em `evals/baseline.json`.

Os limites iniciais são conservadores e devem ser calibrados com a primeira execução real na main. A regra de ouro é não rebaixar um limite só para destravar um PR. Se o gate reprovou algo legítimo, a resposta certa é entender a queda, não mexer no número.

## Decisões de projeto

A busca é lexical (BM25), não vetorial. A escolha foi deliberada: BM25 é determinístico, não custa nada além de CPU e funciona bem com termos jurídicos específicos, como "controlador", "operador" e "anonimização". A limitação é que ele não entende sinônimos. A interface de recuperação foi isolada para que comparar com embeddings seja uma troca de implementação, e essa comparação é o próximo experimento natural do projeto.

O juiz pode e deve usar um modelo diferente do gerador, configurado por `EVAL_JUDGE_MODEL`. Um modelo tende a dar notas melhores às próprias respostas, e separar os papéis reduz esse viés. Com um único modelo, o viés continua presente, e é preciso dizer isso ao avaliar os resultados.

O verificador de sustentação não confia só na palavra do juiz. Quando o juiz diz que uma afirmação é sustentada, ele precisa apontar um trecho literal do chunk, e esse trecho é procurado de forma determinística no texto recuperado. Juízes LLM podem confirmar uma afirmação e inventar a citação que a justificaria. Esse caso é contado como não sustentado.

A recusa é tratada como caminho próprio. Se nenhum chunk é recuperado, a resposta é a recusa, sem chamar o modelo de geração. Isso é mais barato e evita que o modelo invente contexto quando não há contexto nenhum.

Todas as chamadas ao LLM passam por um cache em disco indexado pelo hash de modelo, prompts e limite de tokens. Reexecutar a suíte sem mudanças não custa nada, e uma mudança de prompt invalida apenas as chamadas afetadas. A temperatura é zero, para que a comparação entre execuções reflita mudanças no sistema, e não ruído de amostragem.

Os testes não dependem de API. O `FakeLLM` em `tests/fakes.py` devolve respostas enfileiradas, o que permite testar o verificador de sustentação, os juízes e o gate de forma determinística. A única parte que exige a API é a execução de avaliação real.

## Como rodar

Os comandos abaixo assumem Python 3.12 e um shell POSIX. A forma mais simples de começar é com Docker, mas o fluxo local também funciona.

```bash
make install          # instala as dependências
make corpus           # baixa a LGPD do Planalto para data/corpus/lgpd.txt
make test             # roda os testes unitários (não precisa de API)

cp .env.example .env  # preencha ANTHROPIC_API_KEY e, se quiser, os preços
make smoke            # 3 casos, para checar a chave e o pipeline
make baseline         # avalia a versão atual e grava evals/baseline.json
make eval             # avalia de novo e aplica o gate
```

Para comparar uma variante com a baseline, passe os parâmetros em `EVAL_ARGS`:

```bash
make eval EVAL_ARGS="--chunk-size 400 --chunk-overlap 40"
make eval EVAL_ARGS="--prompt v2 --top-k 6"
```

Com Docker, `docker compose up --build` executa `make eval` dentro de um container com as dependências fixas. O cache e os resultados ficam no host, em `.cache/` e `results/`.

O relatório de cada execução fica em `results/report.md`, e o JSON completo, com cada caso, resposta, afirmações e veredictos, fica em `results/<nome>.json`. Esse JSON é o lugar para investigar uma regressão caso a caso.

## Concordância entre juiz e humano

Um juiz LLM só é útil se concordar com quem entende do assunto. Para medir isso, `make eval` gera `results/candidate_claims.csv`, com todas as afirmações julgadas. Preencha a coluna `human_supported` com 1 ou 0 em uma amostra de 30 a 50 linhas, sem olhar a coluna `judge_supported` para não contaminar a anotação, e rode:

```bash
make agreement
```

O script reporta a concordância simples e o kappa de Cohen. O kappa é o número que importa, porque desconta a concordância que se espera por acaso, o que é relevante quando a maioria das afirmações é sustentada.

## Resultados

<!-- Preencher somente com saídas de `make baseline` e `make eval` executadas no repositório. -->

| Métrica | Baseline | Candidato | Piso |
|---|---|---|---|
| faithfulness | — | — | 0.80 |
| answer_relevance | — | — | 0.70 |
| context_precision | — | — | 0.50 |
| refusal_accuracy | — | — | 0.85 |

Custo e latência por consulta: a preencher a partir do relatório.

Concordância juiz × humano: a preencher com o resultado de `make agreement`, incluindo o número de afirmações anotadas.

### Caso de regressão

<!-- Documentar aqui uma queda real: mudança feita, métrica afetada, o que o gate reportou e a causa encontrada. -->

Para reproduzir o experimento, troque o tamanho do chunk para um valor muito pequeno, por exemplo `--chunk-size 150 --chunk-overlap 20`, e rode `make eval`. Um chunk pequeno demais separa a informação do contexto que a sustenta, e espera-se que a `faithfulness` ou a `context_precision` caiam o suficiente para reprovar o gate. O relatório mostra qual métrica caiu e em quanto.

## Limitações

O golden set inicial tem apenas 12 casos. A recomendação é chegar a 100 a 150 casos bem escolhidos, cobrindo recuperação direta, perguntas que exigem combinar trechos e perguntas fora do escopo. Casos gerados em massa sem revisão medem pouco. Os `expected_answer` foram escritos a partir do conhecimento sobre a lei e precisam ser conferidos contra o texto oficial antes de qualquer publicação dos números.

Com 12 casos, uma métrica pode variar bastante por causa de um único caso. Isso é adequado para validar o pipeline, mas não para tirar conclusões sobre qual prompt é melhor. Conclusões exigem um conjunto maior e intervalos de confiança.

Os juízes são modelos e erram. A concordância com humanos é a medida de quanto se pode confiar neles, e precisa ser reportada junto com qualquer número.

A busca é lexical. Perguntas que usam palavras diferentes das do documento, como "dado de saúde" para "dado referente à saúde", podem não recuperar o chunk certo. Isso aparece na `context_precision` e no `refusal_accuracy`, e é o principal motivo para testar embeddings.

O corpus é uma única lei, com chunking por parágrafo. Documentos com estrutura forte, como contratos ou normas com artigos numerados, se beneficiariam de cortes guiados pela estrutura.

## Estrutura do repositório

```
eval-gate/
├── app/                    # o sistema avaliado
│   ├── config.py           # parâmetros de cada experimento
│   ├── ingest.py           # carregamento e chunking
│   ├── retrieval.py        # BM25
│   ├── llm.py              # cliente da API e cache em disco
│   └── answer.py           # geração com recusa; versões de prompt
├── evals/                  # a camada de avaliação
│   ├── golden.jsonl        # casos de avaliação, versionados
│   ├── dataset.py          # leitura e validação do golden set
│   ├── judges.py           # juízes de relevância, contexto e recusa
│   ├── hallucination.py    # verificador de sustentação
│   ├── run_eval.py         # executa o pipeline e agrega as métricas
│   ├── gate.py             # piso, regressão e código de saída
│   ├── report.py           # relatório em markdown
│   ├── agreement.py        # concordância juiz × humano
│   └── baseline.json       # resultado de referência da main
├── tests/                  # testes unitários e de regressão
├── scripts/fetch_corpus.py # baixa o corpus
├── data/corpus/            # documentos de origem
├── thresholds.yaml         # pisos e tolerâncias do gate
├── .github/workflows/eval.yml
├── Makefile
├── Dockerfile
└── docker-compose.yml
```

## Licença

MIT. Veja o arquivo LICENSE.
