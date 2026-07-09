# aixon

> One framework. Declarative agents, multi-agent orchestration, and a protocol-decoupled server.

[![Test & Publish](https://github.com/JorgeHSantana/aixon/actions/workflows/publish.yml/badge.svg)](https://github.com/JorgeHSantana/aixon/actions/workflows/publish.yml)
[![codecov](https://codecov.io/gh/JorgeHSantana/aixon/branch/main/graph/badge.svg)](https://codecov.io/gh/JorgeHSantana/aixon)
[![PyPI](https://img.shields.io/pypi/v/aixon?logo=pypi&logoColor=white)](https://pypi.org/project/aixon/)
[![Python](https://img.shields.io/pypi/pyversions/aixon?logo=python&logoColor=white)](https://pypi.org/project/aixon/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`aixon` is a Python framework for building AI-agent systems. Subclass an agent type,
declare your LLM and tools as class attributes, and the agent self-registers — no
wiring, no routing table. Connect agents into multi-agent graphs with
`Orchestrator`. Serve them over any wire format through a pluggable
`ProtocolAdapter`. Run the whole thing locally or expose it as an API that any
OpenAI-compatible client can reach.

---

## Architecture

```mermaid
graph LR
    Client["🤖 Client / LLM"] -->|"HTTP · SSE"| SA["Server<br/>(ProtocolAdapter)"]
    SA --> R["Registry"]
    R --> A["Agent<br/>(LLMAgent · ToolAgent · Orchestrator)"]
    A --> LLM["LLM<br/>(Provider)"]
    A --> T["Tools<br/>(Retriever · Connector · Agent)"]
    A --> O["Orchestrator<br/>→ nodes (Agents)"]
    A --> RF["ReflectiveAgent<br/>→ agent + judge_llm"]

    style SA fill:#4f46e5,color:#fff,stroke:none
    style A fill:#7c3aed,color:#fff,stroke:none
    style LLM fill:#9333ea,color:#fff,stroke:none
    style T fill:#a855f7,color:#fff,stroke:none
    style RF fill:#a855f7,color:#fff,stroke:none
```

The **neutral boundary** is the key design principle: every agent speaks only
`Message[]` in and `Message`/`Chunk` out — no provider type, no wire type ever
crosses into the agent runtime. Protocol adapters translate on the outside;
provider SDKs stay hidden inside `LLM`.

---

## Installation

```bash
pip install aixon          # core: langchain + langgraph — agents work out of the box
```

`langchain`/`langchain-core`/`langgraph` are mandatory core dependencies. The
optional extras add the outer layers:

```bash
pip install "aixon[server]"            # FastAPI + uvicorn + httpx — serve agents as an API
pip install "aixon[cli]"               # click + openai — the `aixon` command + remote chat
pip install "aixon[openai]"            # OpenAI provider binding (langchain-openai)
pip install "aixon[anthropic]"         # Anthropic provider binding
pip install "aixon[google]"            # Google provider binding
pip install "aixon[zai]"               # z.AI provider binding (GLM via langchain-openai)
pip install "aixon[retrieval]"         # httpx — Connector / HttpToolConnector
pip install "aixon[openai-embedding]"  # langchain-openai — OpenAIEmbedding
pip install "aixon[weaviate]"          # Weaviate vector-store Retriever
pip install "aixon[ragie]"             # Ragie managed-RAG Retriever
pip install "aixon[tavily]"            # Tavily web-search Retriever
pip install "aixon[rerank]"            # flashrank reranking (for Weaviate)
pip install "aixon[tiktoken]"          # token counting for the server `usage` field
pip install "aixon[all]"               # everything above
```

---

## 60-second quickstart

```bash
# 1. Scaffold a consumer project
aixon new my-agents
cd my-agents
pip install -e ".[all]"

# 2. Start the interactive chat
aixon chat

# 3. Or serve the OpenAI-compatible API
aixon serve
```

Or inline — no scaffolding needed:

```python
# agents/hello.py
from aixon import LLMAgent, LLM

class HelloAgent(LLMAgent):
    llm = LLM("gpt-4o-mini", temperature=0.2)
    description = "Greets the user"
    prompt = "You are a concise greeter. Reply in one sentence."
```

```python
# main.py
from aixon import autodiscover, Message
from aixon.registry import get_registry

autodiscover("agents")
agent = get_registry().resolve("helloagent")
reply = agent.invoke([Message(role="user", content="Hi!")])
print(reply.content)
```

```bash
python main.py
# → Hi there! How can I help you today?
```

---

## The Agent model

Everything in `aixon` is an `Agent` — a single callable unit with a uniform
interface:

```python
agent.invoke(messages: list[Message])  -> Message
agent.stream(messages: list[Message])  -> Iterator[Chunk]
agent.ainvoke(messages: list[Message]) -> Message            # async
agent.astream(messages: list[Message]) -> AsyncIterator[Chunk]
agent.as_tool(name=None, description=None) -> AgentTool
```

Sync is the default; `ainvoke`/`astream` are **additive** and run non-blocking
(native for `LLMAgent`/`ToolAgent`/`Orchestrator`, a thread-bridge for custom
sync agents). The server `await`s them, so concurrent requests don't serialize.

Four concrete subtypes cover the common cases. Pick the one that matches what
you need:

| Subtype | When to use | Suffix required |
|---|---|---|
| `LLMAgent` | Direct LLM call — no tools, no loop | `*Agent` |
| `ToolAgent` | LLM + tool-calling loop (LangGraph `create_agent`) | `*Agent` |
| `Orchestrator` | Multiple agents coordinated by a graph | `*Orchestrator` |
| `ReflectiveAgent` | Wrap a worker agent in a generate → judge → retry review loop | `*Agent` |

**Suffix rule:** every concrete subclass name must end with its declared suffix.
Violating it raises `NamingError` at import time — before the server starts.

```python
class Greeter(LLMAgent):    # ← raises NamingError: missing 'Agent' suffix
    ...

class GreeterAgent(LLMAgent):  # ← correct
    ...
```

---

## LLMAgent — direct LLM call

```python
from aixon import LLMAgent, LLM

class PlannerAgent(LLMAgent):
    llm         = LLM("gpt-4o-mini", temperature=0.2)
    description = "Strategic planner"
    prompt      = "You plan step-by-step actions for complex goals."
```

Attributes:

| Attribute | Type | Description |
|---|---|---|
| `llm` | `LLM` | **Required.** The language model to use. |
| `prompt` | `str` | Optional system prompt prepended to every conversation. |
| `description` | `str` | Human-readable purpose (shown in `aixon list`). |
| `name` | `str` | Registry name (defaults to lowercased class name). |
| `aliases` | `list[str]` | Alternate names for registry resolution. |
| `hidden` | `bool` | Exclude from `aixon chat` menu and `public()` listing. |

`aixon list` output (no header, one agent per line):

```
greeteragent  [LLMAgent]  Friendly greeter
```

See [docs/agents.md](docs/agents.md) for `ToolAgent` and full API reference.

---

## Orchestrator — multi-agent graphs

```python
from aixon import Orchestrator, LLM

class SupportOrchestrator(Orchestrator):
    supervisor = LLM("gpt-4o-mini")          # routes each turn to a worker
    agents     = [BillingAgent, TechAgent, PlannerAgent]   # your own LLMAgent/ToolAgent classes
```

Three tiers — pick by complexity. See [docs/orchestrator.md](docs/orchestrator.md).

---

## ReflectiveAgent — evaluator-optimizer loop

```python
from aixon import ReflectiveAgent, LLM

class ReviewedWriterAgent(ReflectiveAgent):
    agent         = WriterAgent                  # worker: any Agent, class or instance
    judge_llm     = LLM("gpt-4o-mini", temperature=0)
    judge_rubric  = "The answer must cite a source for every fact it states."
    max_rounds    = 3
```

Generate → judge against an objective rubric → on rejection, retry with the
critique — up to `max_rounds`. Exhausting the rounds returns the last attempt
rather than raising. See [docs/agents.md](docs/agents.md#reflectiveagent--evaluator-optimizer-loop)
and the runnable [examples/reflective_review](examples/reflective_review).

---

## Retrieval — Retriever, Connector, Embedding

`Retriever` is the base class for any agent that fetches context. Name it with a
`*Retriever` suffix and declare `type_access` to control read/write permissions
(`TypeAccess.READ`, `TypeAccess.WRITE`, or `TypeAccess.ALL`).

```python
from aixon import Retriever, TypeAccess

class LibraryRetriever(Retriever):
    description = "Fetches relevant documents from the knowledge base"
    type_access = TypeAccess.READ
```

Every `Retriever` has a sync `search` and a native async `asearch`; `as_tool()`
returns a dual tool that works on both the sync and async agent paths.

`Connector` is an HTTP base class for wrapping external APIs (`base_url_env` /
`auth_token_env`, sync `get`/`post` and async `aget`/`apost`). `HttpToolConnector`
builds on it for HTTP-JSON tool servers.

`Embedding` is the vector-embedding ABC; the built-in implementation is
`OpenAIEmbedding`.

**Vendor retrievers** ship behind optional extras — each a neutral `Retriever`,
lazy (the vendor SDK is imported only on use) and async-native:

| Retriever | Extra | Backend |
|---|---|---|
| `WeaviateRetriever` | `aixon[weaviate]` | Weaviate vector store (+ `aixon[rerank]` for flashrank) |
| `RagieRetriever` | `aixon[ragie]` | Ragie managed RAG |
| `TavilyRetriever` | `aixon[tavily]` | Tavily web search |

```python
from aixon import Retriever, TypeAccess, Connector, Embedding, OpenAIEmbedding
```

See [docs/retrieval.md](docs/retrieval.md) (ABCs) and
[docs/retrievers.md](docs/retrievers.md) (vendor backends).

---

## Server

```python
from aixon import Server, autodiscover

autodiscover("agents")
server = Server()
server.serve(host="0.0.0.0", port=8000)
```

Any OpenAI-compatible client works out of the box:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="test")
response = client.chat.completions.create(
    model="planneragent",
    messages=[{"role": "user", "content": "Plan a trip to Tokyo."}],
)
```

See [docs/server.md](docs/server.md) for Anthropic adapter, auth, and SSE streaming.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Framework log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `AUTH_API_KEY` | _(disabled)_ | Bearer token for the server. Unset = no auth. Multiple keys comma-separated. |
| `OPENAI_API_KEY` | _(required for OpenAI)_ | API key for the OpenAI provider. |
| `ANTHROPIC_API_KEY` | _(required for Anthropic)_ | API key for the Anthropic provider. |
| `GOOGLE_API_KEY` | _(required for Google)_ | API key for the Google provider. |
| `ZAI_API_KEY` | _(required for z.AI)_ | API key for the z.AI provider (GLM models). `ZAI_BASE_URL` overrides the default `https://api.z.ai/api/paas/v4`. |

---

## Naming conventions

Suffix violations raise `NamingError` at import time — the server never starts
with a mis-named class.

| Base class | Required suffix | Example |
|---|---|---|
| `LLMAgent` | `*Agent` | `PlannerAgent` |
| `ToolAgent` | `*Agent` | `DiagnosisAgent` |
| `Orchestrator` | `*Orchestrator` | `SupportOrchestrator` |
| `ReflectiveAgent` | `*Agent` | `ReviewedWriterAgent` |
| `Retriever` | `*Retriever` | `LibraryRetriever` |
| `Connector` | `*Connector` | `CRMConnector` |

Abstract intermediate classes (declared with `abstract=True`) are exempt and
never registered.

---

## Documentation

- [Architecture](docs/architecture.md) — layers, neutral boundary, protocol decoupling
- [Agents](docs/agents.md) — `LLMAgent`, `ToolAgent`, `ReflectiveAgent`, declarative API, `as_tool`, async
- [Orchestrator](docs/orchestrator.md) — three tiers, entry/topology, branching, recursion guards
- [Server](docs/server.md) — `ProtocolAdapter`, adapters, auth, SSE
- [Retrieval](docs/retrieval.md) — `Retriever`, `Embedding`, `Connector`
- [Vendor retrievers](docs/retrievers.md) — `Weaviate`, `Ragie`, `Tavily`
- [CLI](docs/cli.md) — `chat`, `new`, `serve`, `list`
- [Quickstart](docs/quickstart.md) — consumer project walkthrough
- [Example](examples/support_assistant) — a complete multi-agent support assistant, runnable offline
- [Example: Reflective Review](examples/reflective_review) — the `ReflectiveAgent` loop, runnable offline

---

## Dependencies

```
langchain             >= 1.0    (core)
langchain-core        >= 1.0    (core)
langgraph             >= 1.0    (core)
fastapi / uvicorn / pydantic    (server extra)
httpx                 >= 0.27   (server / retrieval extra)
click / openai                  (cli extra — `aixon` command + remote chat)
langchain-openai      >= 0.2    (openai / openai-embedding / zai extra)
langchain-anthropic   >= 0.2    (anthropic extra)
langchain-google-genai >= 2.0   (google extra)
weaviate-client / langchain-weaviate  (weaviate extra)
ragie                 >= 2      (ragie extra)
tavily-python         >= 0.7    (tavily extra)
flashrank             >= 0.2    (rerank extra)
tiktoken              >= 0.7    (tiktoken extra)
```

---

## Author

**Jorge Henrique Moreira Santana**  
Electrical Engineer, Postgraduate in Artificial Intelligence  
[LinkedIn](https://www.linkedin.com/in/jorge-santana-b246874a/) · ti@zeusagro.com

---

## License

[MIT](LICENSE)
