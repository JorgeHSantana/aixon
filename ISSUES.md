# aixon — known issues

Incongruities found while building/using the framework. Tracked here to fix
incrementally. Each entry: symptom, location, suggested fix, status.

---

## 1. Fictional model `gpt-5.4` lingering in source docstrings and error messages

**Status:** open

**Symptom:** The living docs were corrected to real models (`gpt-4o-mini`), but
several source files still reference the fictional `gpt-5.4` in docstrings and —
worse — in a **user-facing error message**.

**Locations:**
- `aixon/llm.py` — module docstring (`llm = LLM("gpt-5.4", temperature=0.2)`)
- `aixon/agents/llm_agent.py` — class/module docstrings (×2) and the
  `_validate_subclass` error message (`e.g. llm = LLM('gpt-5.4')`)
- `aixon/agents/tool_agent.py` — class docstring and the `_validate_subclass`
  `AixonError` message (`e.g. \`llm = LLM("gpt-5.4")\``)

**Why it matters:** The error messages are shown to consumers who forget to
declare `llm`. Suggesting a fictional model in an error is confusing.

**Suggested fix:** Replace every `gpt-5.4` with `gpt-4o-mini` across these files.

---

## 2. Built-in OpenAI and Anthropic adapters both claim `GET /v1/models`

**Status:** open

**Symptom:** `OpenAIAdapter.routes()` returns `GET /v1/models` and
`AnthropicAdapter.routes()` also returns `GET /v1/models`. Mounting both on one
`Server(adapters=[OpenAIAdapter(), AnthropicAdapter()])` registers the same
`GET /v1/models` path twice — the first wins, the second is shadowed, and
`POST /v1/chat/completions` vs `POST /v1/messages` coexist only by luck of
distinct paths.

**Why it matters:** A user wanting to expose both dialects at once cannot do so
cleanly. The default `Server()` mounts OpenAI only, so this is latent, but any
multi-dialect deployment hits it.

**Suggested fix:** Give each adapter a mount prefix (e.g. Anthropic under
`/anthropic`) or let `Server` namespace adapter routes so two adapters never
collide on a shared path. Until then, the support_assistant example uses the
default OpenAI-only `Server()`.

---

## 3. CLI does not put the current directory on `sys.path`

**Status:** open

**Symptom:** Running `aixon list` / `aixon serve` / `aixon chat` from a project
directory reports **"No agents registered."** even when an importable `agents/`
package is right there. `autodiscover("agents")` does
`importlib.import_module("agents")`, but the installed `aixon` console-script
has its launcher dir on `sys.path[0]`, not the CWD — so the import fails and the
CLI swallows it (`except (ImportError, ModuleNotFoundError, ValueError): pass`).

**Repro:**
```
cd examples/support_assistant
aixon list                 # -> "No agents registered."
PYTHONPATH=. aixon list     # -> "support  [SupportOrchestrator]  ..."
```

**Why it matters:** `python main.py` works (CWD is on the path for scripts), but
the CLI does not — a confusing inconsistency. The silent swallow makes it look
like nothing is wrong.

**Suggested fix:** In the CLI commands, insert `os.getcwd()` at `sys.path[0]`
before calling `autodiscover()` (mirrors how `python main.py` behaves). The
support_assistant README documents the `PYTHONPATH=.` workaround until then.

---
