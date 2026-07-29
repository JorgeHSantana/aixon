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

## 3. Langfuse — integração de primeira classe (0.1.27+, #26)

O Server já vem instrumentado: instale o extra e configure as envs — nada
de código.

```bash
pip install "aixon[langfuse]"
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://SEU-LANGFUSE"   # omitido = Langfuse Cloud
```

Com as DUAS chaves presentes, cada POST de chat vira **um trace** com o nome
do agente resolvido; sem elas, custo zero (nenhum import do SDK). Por dentro:
um span raiz por request agrupa TODOS os turnos internos (workers,
roteamento do orquestrador, juiz do reflective — que sem o span virariam
traces separados), e o `CallbackHandler` do Langfuse é anexado via
*configure hook* do langchain-core — o mesmo mecanismo do LangSmith, cobrindo
grafos, modelos e forks do langgraph sem tocar em nenhum call-site.

- **Por agente**: nome do trace = agente. **Por modelo**: cada generation
  registra o modelo REAL do provider que atendeu aquele turno (worker/juiz
  podem usar modelos diferentes); cadastre os preços no Langfuse para ver
  custo além de tokens. **Por usuário**: se o frontend encaminhar
  identidade (Open WebUI: `ENABLE_FORWARD_USER_INFO_HEADERS=true`), os
  headers `X-OpenWebUI-User-Email`/`-Id` viram `user_id` do trace e
  `X-OpenWebUI-Chat-Id` vira `session_id`.
- **Falha nunca derruba request**: SDK ausente/mal configurado desliga a
  integração com um warning; Langfuse inalcançável só atrasa spans.
- **Serverless**: por padrão há um `flush()` ao fim de cada request (Cloud
  Run congela a CPU pós-resposta; sem o flush, spans em batch podem nunca
  sair). Deploy com CPU always-on pode desligar:
  `AIXON_LANGFUSE_FLUSH_PER_REQUEST=0`.

Exemplo runnable: [examples/langfuse](../examples/langfuse/).

Onde hospedar: VM com docker-compose (o stack v3 tem ClickHouse — não cabe
em Cloud Run) ou Langfuse Cloud; para o aixon é indiferente, são as 3 envs.

Rota manual (pré-0.1.27, segue válida fora do Server): a instrumentação
OpenTelemetry do LangChain (`opentelemetry-instrumentation-langchain` +
endpoint OTLP do Langfuse) captura globalmente, ou o `CallbackHandler` do
SDK quando você controla a invocação diretamente.

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
