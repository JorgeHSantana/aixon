# CLI

`aixon` ships a command-line interface for interactive development and
deployment.

```
Usage: aixon [OPTIONS] COMMAND [ARGS]...

Commands:
  chat   Interactive agent chat (in-process or remote)
  list   List registered agents
  new    Scaffold a new consumer project
  serve  Start the API server
```

Install the CLI extra to get the `aixon` command in your PATH:

```bash
pip install 'aixon[cli]'
```

---

## aixon list

List all registered agents in the current project.

```bash
# List agents from the default package
aixon list

# List from a specific package
aixon list --package myagents
aixon list -p myagents
```

```
Options:
  --package, -p  TEXT  Package to autodiscover before listing. [default: agents]
```

Output — one line per agent, `name  [Type]  description`:

```
planneragent  [LLMAgent]  Breaks complex goals into step-by-step plans
researchagent  [ToolAgent]  Researches topics using web search and the knowledge base
supportorchestrator  [Orchestrator]  Routes support tickets to the right specialist
```

Hidden agents (`.hidden = True`) are excluded. Use `get_registry().all()`
programmatically to include them.

---

## aixon chat

Start an interactive session with a registered agent.

```bash
# In-process (default): agents run directly in the CLI process
aixon chat

# In-process with explicit package
aixon chat --package myagents

# Remote: connects to a running `aixon serve` instance
aixon chat --url http://localhost:8000
```

```
Options:
  --package, -p  TEXT  Package to autodiscover (ignored when --url is set). [default: agents]
  --url          TEXT  Remote server URL (e.g. http://localhost:8000).
                       When set, routes messages via the OpenAI wire format.
```

**Two modes, same command:**

| Mode | How | When to use |
|---|---|---|
| In-process | Imports and invokes agents directly | Development, no server running |
| Remote | OpenAI client against `aixon serve` | Test against a live server; same UX |

**In-process flow:**

1. `autodiscover(package)` imports all non-hidden agents (default package: `"agents"`).
2. A numbered menu lists all non-hidden agents with their type and description.
3. Select an agent by number.
4. Type messages. The agent streams its response — `reasoning` is shown dimmed
   (grey on ANSI-capable terminals), `content` is shown normally.
5. In-session commands:
   - `/menu` — return to the agent selection menu (conversation history is reset).
   - `/exit` — quit the CLI.
   - `Ctrl+C` — interrupt the current generation; press again at an empty prompt
     to return to the agent selection menu.

**Remote flow:**

1. `--url` connects to a running `aixon serve` instance.
2. Available agents are fetched from the server's `/v1/models` endpoint.
3. Select an agent by number.
4. Chat proceeds identically to in-process mode — `/menu`, `/exit`, and `Ctrl+C`
   all work.

The remote mode uses the `OpenAIAdapter` wire format — any `aixon serve` instance
is compatible. The `openai` package is required; it is included in `aixon[cli]`.

---

## aixon new

Scaffold a new consumer project.

```bash
aixon new my-agents
cd my-agents
pip install -e .
python main.py
```

Generated structure:

```
my-agents/
├── agents/
│   ├── __init__.py     # bare package marker — drop .py files here
│   └── greeter.py      # example LLMAgent
├── main.py             # autodiscover + serve entry point
└── pyproject.toml
```

`main.py` (generated):

```python
from aixon import Server, autodiscover

# Import every module in agents/, registering each Agent at startup.
autodiscover("agents")

# OpenAI-compatible API server. Set AUTH_API_KEY to require a Bearer token.
server = Server()
app = server.app  # ASGI app — for production: `uvicorn main:app --workers 4`

if __name__ == "__main__":
    server.serve(host="0.0.0.0", port=8000)
```

`agents/greeter.py` (generated):

```python
from aixon import LLMAgent, LLM


class GreeterAgent(LLMAgent):
    description = "Friendly greeter"
    # Replace 'gpt-4o-mini' with any supported model.
    llm = LLM("gpt-4o-mini", temperature=0.7)
    prompt = "You are a friendly assistant. Greet the user warmly."
```

The `pyproject.toml` pins `aixon[server,cli]` as a dependency. Add your own
agents to `agents/` — `autodiscover` registers them automatically; no list to
maintain.

---

## aixon serve

Start the API server.

```bash
# Basic
aixon serve

# Custom host/port
aixon serve --host 127.0.0.1 --port 9000

# Load agents from a specific package
aixon serve --package myagents
aixon serve -p myagents

# With auth
AUTH_API_KEY=my-secret-key aixon serve
```

```
Options:
  --host         TEXT     Host to bind to. [default: 0.0.0.0]
  --port         INTEGER  Port to listen on. [default: 8000]
  --package, -p  TEXT     Package to autodiscover before serving. [default: agents]
```

The server requires the `server` extra:

```bash
pip install 'aixon[server]'
```

It starts `uvicorn` and mounts the `OpenAIAdapter` at `/v1`. See
[server.md](server.md) for mounting additional adapters programmatically.

---

## Environment variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Required when agents use an OpenAI LLM. |
| `ANTHROPIC_API_KEY` | Required when agents use an Anthropic LLM. |
| `GOOGLE_API_KEY` | Required when agents use a Google LLM. |
| `AUTH_API_KEY` | Bearer token required to call `aixon serve` endpoints. |
| `LOG_LEVEL` | Framework log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
