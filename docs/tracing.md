# Tracing e observabilidade

O aixon fala LangChain/LangGraph por baixo do boundary neutro — e o tracing
desses ecossistemas engancha no nível do `langchain-core`. Consequência
prática: **qualquer tracer do ecossistema funciona com agentes aixon sem
mudar uma linha do framework ou dos agentes**. Verificado em demo real: a
árvore de execução completa (chains → model → tools), os prompts enviados,
as respostas e os tokens (incluindo `cached_tokens` do prompt caching e
`accepted/rejected_prediction_tokens` dos Predicted Outputs) aparecem no
tracer.

Há três rotas, da mais barata à mais completa:

## 1. Console (zero conta, zero infra — depurar um run)

```python
from langchain_core.globals import set_debug
set_debug(True)   # despeja cada chain/model/tool no console, com tokens
```

Demo executável offline: [examples/tracing](../examples/tracing).

## 2. LangSmith (SaaS — dev/experimentos)

Zero código; só variáveis de ambiente antes de subir o servidor:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...      # smith.langchain.com (free tier)
export LANGSMITH_PROJECT=meu-projeto
```

Toda execução de todo agente passa a aparecer na UI do LangSmith: árvore por
request, prompts por modelo, latência e custo por nó, replay. O LangGraph
Studio soma depuração visual do grafo (passo a passo, time-travel).

> ⚠️ **Privacidade/LGPD**: traces contêm as CONVERSAS (prompts e respostas
> inteiras). LangSmith é SaaS — em produção com dados de clientes/pessoais,
> use a rota self-hosted abaixo. LangSmith fica para dev com dados neutros.

## 3. Langfuse self-hosted (produção — dados não saem da sua infra)

O Langfuse expõe um endpoint OTLP; a instrumentação OpenTelemetry do
LangChain captura globalmente (sem tocar no aixon):

```bash
pip install opentelemetry-instrumentation-langchain opentelemetry-exporter-otlp
export OTEL_EXPORTER_OTLP_ENDPOINT="https://SEU-LANGFUSE/api/public/otel"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64(pk:sk)>"
```

```python
# uma vez, no boot (ex.: main.py, antes de servir):
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
LangchainInstrumentor().instrument()
```

Alternativa: o `CallbackHandler` do SDK do Langfuse, quando você controla a
invocação diretamente.

## O que você enxerga de graça

- Árvore por request: agente → (ReflectiveAgent: worker/juiz por rodada) →
  tools → chamadas de modelo, com timing por nó.
- Prompts e respostas completos por chamada de modelo.
- Tokens por chamada: prompt/completion, `cached_tokens` (efetividade do
  prompt caching entre rodadas), `accepted/rejected_prediction_tokens`
  (efetividade dos Predicted Outputs no retry).
- Tool calls com argumentos e resultados (o `TOOL ERROR` do shield aparece
  como resultado da tool — dá para medir taxa de falha de cada serviço).

## O que NÃO vem pronto (camada sua)

Análises de negócio — assuntos, sentimento, satisfação por conversa — não
existem em nenhuma dessas ferramentas de fábrica. O caminho recomendado é um
job batch que lê os traces (API do Langfuse), classifica com um modelo barato
e grava scores de volta + numa tabela para dashboard (ver as issues de
adoção nos consumidores). Construir captura própria de traces para isso é
reinventar a roda: a fundação já existe, como mostrado acima.
