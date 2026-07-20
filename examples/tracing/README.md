# Tracing — runnable example

Mostra **o que um tracer captura de um agente aixon** — a árvore
`[chain:LangGraph] > [chain:model] > [tool:...]` com inputs/outputs e timing
por nó — usando a rota mais barata (debug tracer de console) com um modelo
scriptado: **sem API key, sem rede, sem conta**.

```bash
cd examples/tracing
PYTHONPATH=../.. python main.py
```

O ponto: o aixon fala LangChain/LangGraph por baixo, então o tracing engancha
no `langchain-core` — **nenhuma mudança no framework ou no agente**. A mesma
árvore que aparece no console aqui é o que LangSmith/Langfuse recebem:

| Rota | Custo | Quando |
|---|---|---|
| `set_debug(True)` (este exemplo) | zero | depurar um run local |
| **LangSmith** — só env vars (`LANGSMITH_TRACING=true` + key) | free tier | dev/experimentos (⚠️ SaaS: traces contêm as conversas — não usar com dados sensíveis) |
| **Langfuse self-hosted** — OTel (`opentelemetry-instrumentation-langchain` → endpoint OTLP) | sua infra | produção (LGPD: dados não saem de casa) |

Guia completo das três rotas: [docs/tracing.md](../../docs/tracing.md).
