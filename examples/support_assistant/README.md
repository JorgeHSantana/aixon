# Acme Support Assistant — a complete aixon example

A runnable customer-support assistant that **routes** each request to the right
specialist and serves it over an **OpenAI-compatible API**, from one codebase.
It walks the full stack — `Provider → LLM → LLMAgent / ToolAgent → Orchestrator
→ Server` plus `Retriever`, `Embedding` and `Connector` — and exercises every
element of the framework with **no API key and no external services**: a bundled
offline `demo` provider answers, an in-memory FAQ backs the retriever, and the
orders connector falls back to a fixture. So `python main.py` just works.

Set `OPENAI_API_KEY` and the very same code uses `gpt-4o-mini` and OpenAI
embeddings instead — nothing else changes.

## Architecture

```
                       ┌──────────────────────────────┐
   POST /v1/chat/...   │      SupportOrchestrator      │   (Tier 2 graph)
  ───────────────────▶ │            "triage"           │
                       │              │                │
                       │   route_triage(verdict)       │
                       │        ┌─────┴─────┐           │
                       │        ▼           ▼           │
                       │   "orders"     "knowledge"     │
                       └──────┬──────────────┬──────────┘
                              │              │
                  OrdersAgent │              │ KnowledgeAgent
                 (ToolAgent)  │              │  (ToolAgent)
                              ▼              ▼
                     OrdersConnector   KnowledgeRetriever
                     (HTTP / fixture)  (Embedding search)
```

`triage` is an `LLMAgent` that classifies the message into one word; the
orchestrator's `route_triage` sends it to the `orders` or `knowledge`
specialist `ToolAgent`, whose answer is the final reply.

## What it demonstrates

| Element | Where |
|---|---|
| Custom `Provider` + `register_provider` (offline LLM) | [providers/demo.py](providers/demo.py) |
| `Embedding` ABC (offline) + `OpenAIEmbedding` swap | [providers/demo.py](providers/demo.py), [knowledge/faq_retriever.py](knowledge/faq_retriever.py) |
| `LLM` declarative handle (one place to pick the model) | [llm_config.py](llm_config.py) |
| `Retriever` + `TypeAccess.READ` + `as_tool()` | [knowledge/faq_retriever.py](knowledge/faq_retriever.py) |
| `Connector` (env config, `get`, Bearer auth, fixture fallback) | [connectors/orders.py](connectors/orders.py) |
| `LLMAgent` (pure LLM, no tools) | [agents/triage.py](agents/triage.py) |
| `ToolAgent` (tool-calling loop) ×2 | [agents/knowledge_agent.py](agents/knowledge_agent.py), [agents/orders_agent.py](agents/orders_agent.py) |
| `emit_reasoning` surfacing in `invoke()`/`stream()` | [agents/orders_agent.py](agents/orders_agent.py) |
| `Orchestrator` Tier 2 (nodes + `entry` + `route_<node>`) | [agents/support.py](agents/support.py) |
| `hidden` workers + `aliases` on the public entry point | [agents/support.py](agents/support.py) |
| `autodiscover` (drop a file in `agents/`, it goes live) | [main.py](main.py) |
| `Server` — OpenAI wire protocol + Bearer auth | [main.py](main.py) |
| Dependency-injection / offline testing | [test_support_assistant.py](test_support_assistant.py) |

## Run it

```bash
cd examples/support_assistant
pip install -r requirements.txt
python main.py            # http://localhost:8000  (set PORT to change)
```

## Call it (OpenAI wire format)

```bash
# Public, no auth:
curl http://localhost:8000/health
curl http://localhost:8000/v1/models          # lists 'support' (+ aliases)

# An orders question -> routed to the orders specialist -> connector lookup:
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"support","messages":[{"role":"user","content":"where is my order 1002?"}]}'

# A product question -> routed to the knowledge specialist -> FAQ search:
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"support","messages":[{"role":"user","content":"how do I enable SSO?"}]}'

# Stream it (SSE) — reasoning lines arrive before the answer:
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"support","messages":[{"role":"user","content":"cancel my order 1003"}],"stream":true}'
```

The assistant is also reachable by its aliases — use `"model":"assistant"` or
`"model":"help"`. Any OpenAI SDK works too; point `base_url` at
`http://localhost:8000/v1`.

## Use the CLI

```bash
# The CLI autodiscovers the local agents/ package — put the CWD on the path:
PYTHONPATH=. aixon list                 # -> support  [SupportOrchestrator]  ...
PYTHONPATH=. aixon chat                 # interactive, in-process
PYTHONPATH=. aixon chat --url http://localhost:8000   # against the running server
```

> Note: the `PYTHONPATH=.` prefix is a current workaround — see
> [`../../ISSUES.md`](../../ISSUES.md) #3. `python main.py` needs no such prefix.

## Turn on auth

```bash
AUTH_API_KEY=dev-secret python main.py
```

Now every route requires `Authorization: Bearer dev-secret` **except**
`/health` and `/v1/models`, which stay public. Multiple keys:
`AUTH_API_KEY=key-a,key-b`. See [.env.example](.env.example).

## Use real models

```bash
OPENAI_API_KEY=sk-... python main.py
```

Agents switch to `gpt-4o-mini` and the retriever to `OpenAIEmbedding`
(`text-embedding-3-small`) — automatically, via [llm_config.py](llm_config.py).
No other change.

## Test it

```bash
cd examples/support_assistant
python -m pytest          # forces offline mode — no server, no network, no key
```

## Make it real

* **Knowledge:** replace the in-memory FAQ in
  [knowledge/corpus.py](knowledge/corpus.py) (or swap `KnowledgeRetriever` for
  one backed by a vector DB). The agent layer doesn't change.
* **Orders:** set `ORDERS_API_URL` (and `ORDERS_API_TOKEN`) and
  `OrdersConnector.lookup_order` issues a real HTTP `GET` — same method, no code
  change.
* **Model:** set `OPENAI_API_KEY`, or edit [llm_config.py](llm_config.py) to use
  Anthropic/Google (`LLM("claude-3-5-haiku-20241022")`, etc.).
