# Tool Shield + Memoization — runnable 0.1.19 example

A minimal, **fully offline** demonstration of the two tool-robustness features
every `ToolAgent` gets since 0.1.19 (see
[docs/agents.md](../../docs/agents.md#toolagent--the-tool-calling-loop)):

1. **Error shield** (`shield_tool_errors`, default `True`) — a tool that
   raises (here, a fake database whose connection times out) hands the model
   a readable `TOOL ERROR (...)` result instead of killing the whole
   run/stream with an opaque server error. The agent answers explaining the
   outage. `str(e) or repr(e)` guarantees a named error even for exceptions
   with an empty `str()` (e.g. `httpx.ReadTimeout`).
2. **Request-scoped memoization** (`aixon.toolcache`) — inside one activated
   cache scope (the aixon Server activates one per request; `ReflectiveAgent`
   activates one per run), a tool called again with the SAME arguments
   returns the first result without re-executing. Errors are never cached.
   Opt out per tool: `as_tool(memoize=False)` or `fn.aixon_memoize = False`.

No API key, no network call: the driving model is scripted.

## Run

```bash
cd examples/tool_shield_memo
PYTHONPATH=../.. python main.py
```

## Expected output

```
> DB status and Recife weather?

Final answer: The database is unavailable right now (connection timeout), but
the weather in Recife is sunny.

query_database raised TimeoutError — and the run SURVIVED (shield).
get_weather was asked twice with the same args but executed 1 time(s) — the
second call was memoized.
```

(A `WARNING aixon.tools — tool 'query_database' failed: ...` line also appears
on stderr — the shield logs every converted failure.)

## What to look at in [main.py](main.py)

| Element | Where |
|---|---|
| Failing tool → `TOOL ERROR` result (run survives) | `query_database` + the scripted `AIMessage` sequence |
| Same-args call memoized (counter shows 1 execution) | `get_weather` + `CALLS` |
| Scope activation (`with tool_call_cache():`) — what the Server does per request | `main()` |
| The strict opt-out (`shield_tool_errors = False`) | `FieldAssistantAgent` (commented default shown) |
