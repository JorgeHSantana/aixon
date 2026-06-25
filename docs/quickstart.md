# Quickstart — Build a consumer project

This guide builds a small but complete `aixon` consumer project: a support
system with three agents and an orchestrator, served over the OpenAI-compatible
API and accessible via `aixon chat`.

**Time:** ~15 minutes. Assumes Python 3.11+ and an OpenAI API key.

---

## 1. Scaffold the project

```bash
pip install "aixon[all]"
aixon new my-support
cd my-support
pip install -e ".[all]"
```

Project layout (generated):

```
my-support/
├── agents/
│   └── hello.py       # delete this; we'll write our own
├── main.py
└── pyproject.toml
```

---

## 2. Add a knowledge base retriever

```python
# agents/retriever.py
from aixon import Retriever, TypeAccess

class KnowledgeRetriever(Retriever):
    description = "Searches the product knowledge base"
    type_access = TypeAccess.READ

    # Minimal in-memory implementation for the quickstart.
    _docs = [
        {"text": "Battery life is 8 hours.", "metadata": {"topic": "battery"}},
        {"text": "Reset by holding power for 10 seconds.", "metadata": {"topic": "reset"}},
    ]

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        limit = k or len(self._docs)
        return self._docs[:limit]
```

---

## 3. Write the specialist agents

```python
# agents/billing.py
from aixon import LLMAgent, LLM

class BillingAgent(LLMAgent):
    llm         = LLM("gpt-4o-mini", temperature=0.1)
    description = "Handles billing and account questions"
    prompt      = "You are a billing specialist. Be concise and factual."
```

```python
# agents/tech.py
from aixon import ToolAgent, LLM
from agents.retriever import KnowledgeRetriever

class TechAgent(ToolAgent):
    llm         = LLM("gpt-4o-mini", temperature=0.1)
    description = "Handles technical issues using the knowledge base"
    prompt      = "Use the knowledge base. Cite your sources."
    tools       = [KnowledgeRetriever()]
```

---

## 4. Write the orchestrator

```python
# agents/support.py
from aixon import Orchestrator, LLM
from agents.billing import BillingAgent
from agents.tech import TechAgent

class SupportOrchestrator(Orchestrator):
    description = "Routes support tickets to the right specialist"
    supervisor  = LLM("gpt-4o-mini")
    agents      = [BillingAgent, TechAgent]
```

---

## 5. Verify the registry

```bash
aixon list --package agents
```

Expected output — one line per agent, `name  [Type]  description`:

```
billingagent  [LLMAgent]  Handles billing and account questions
techagent  [ToolAgent]  Handles technical issues using the knowledge base
supportorchestrator  [Orchestrator]  Routes support tickets to the right specialist
```

`KnowledgeRetriever` does not appear — `Retriever` subclasses are not `Agent`
subclasses and are never registered in the agent registry.

---

## 6. Chat interactively

```bash
OPENAI_API_KEY=sk-... aixon chat --package agents
```

Select `supportorchestrator`. Ask: "My battery dies after 3 hours — is this normal?"

The orchestrator routes to `TechAgent`, which searches the knowledge base and
replies: "According to the manual, battery life should be 8 hours. A 3-hour life
suggests the battery may need replacement."

---

## 7. Serve the API

```bash
OPENAI_API_KEY=sk-... aixon serve --package agents
```

Test with curl:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
# → {"object": "list", "data": [{"id": "billingagent", ...}, ...]}

curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "supportorchestrator",
    "messages": [{"role": "user", "content": "How do I reset the device?"}],
    "stream": false
  }' | python -m json.tool
```

Or with any OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="any")
response = client.chat.completions.create(
    model="supportorchestrator",
    messages=[{"role": "user", "content": "How do I reset the device?"}],
)
print(response.choices[0].message.content)
```

---

## 8. Add a Connector (optional)

Add a Connector if your agents need to call an external microservice:

```python
# agents/crm.py
from aixon import Connector

class CRMConnector(Connector):
    base_url_env   = "CRM_API_URL"
    auth_token_env = "CRM_API_KEY"

    def get_ticket(self, ticket_id: str) -> dict:
        return self.get(f"/tickets/{ticket_id}")
```

```python
# agents/tech.py (extended)
from agents.crm import CRMConnector

class TechAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini", temperature=0.1)
    tools = [
        KnowledgeRetriever(),
        CRMConnector().get_ticket,   # plain callable → coerced to StructuredTool
    ]
```

---

## 9. Enable auth (production)

```bash
AUTH_API_KEY=my-production-key OPENAI_API_KEY=sk-... aixon serve --package agents
```

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-production-key",
)
```

---

## Next steps

- [Architecture](architecture.md) — understand the neutral boundary and layers
- [Agents](agents.md) — full `LLMAgent` / `ToolAgent` attribute reference
- [Orchestrator](orchestrator.md) — Tier 2 graphs, branching, recursion guards
- [Server](server.md) — Anthropic adapter, custom adapters
- [Retrieval](retrieval.md) — `TypeAccess.ALL`, `Embedding`, write operations
- [CLI](cli.md) — remote mode, `/menu`, `Ctrl+C`
