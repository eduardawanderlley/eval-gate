"""Relatório em markdown: usado no comentário do PR e salvo em results/report.md."""
from __future__ import annotations


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(candidate: dict, baseline: dict | None, thresholds: dict, failures: list[str]) -> str:
    cm = candidate["metrics"]
    bm = baseline["metrics"] if baseline else {}
    cfg = candidate["config"]
    status = "❌ Reprovado" if failures else "✅ Aprovado"

    lines = [
        f"## Eval Gate: {status}",
        "",
        (
            f"Execução `{cfg['name']}` · {candidate['n_cases']} casos · modelo `{cfg['model']}` · "
            f"chunk {cfg['chunk_size']}/{cfg['chunk_overlap']} · top-k {cfg['top_k']} · prompt {cfg['prompt_version']}"
        ),
        "",
        "| Métrica | Baseline | Candidato | Piso | Status |",
        "|---|---|---|---|---|",
    ]
    for metric, floor in thresholds["floors"].items():
        value = cm.get(metric)
        ok = value is not None and value >= floor
        mark = "✅" if ok else "❌"
        lines.append(f"| {metric} | {_fmt(bm.get(metric))} | {_fmt(value)} | {floor:.2f} | {mark} |")

    lines += [
        "",
        "| Operacional | Valor |",
        "|---|---|",
        f"| Latência p50 / p95 (s) | {_fmt(cm.get('latency_p50_s'))} / {_fmt(cm.get('latency_p95_s'))} |",
        f"| Custo estimado (USD) | {cm.get('cost_usd', 0):.4f} |",
        f"| Tokens entrada / saída | {cm.get('tokens_in', 0)} / {cm.get('tokens_out', 0)} |",
        f"| Cache hit rate | {_fmt(cm.get('cache_hit_rate'))} |",
    ]
    if failures:
        lines += ["", "### Motivos da reprovação", ""] + [f"- {item}" for item in failures]
    return "\n".join(lines) + "\n"
