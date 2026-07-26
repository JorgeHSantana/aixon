# Client Tools — the client brings its own tools

Agentic clients (editors, IDEs — e.g. ONLYOFFICE's AI agent) POST `tools`
on the chat-completions request and expect `tool_calls` back, which they
execute **on the client side** and report via a `role: "tool"` message. This
example runs that whole handshake in-process — no API key, no network:

```
pip install -r requirements.txt
python main.py
```

## What it demonstrates

- **`ParsedRequest.tools`** — the OpenAI adapter extracts the client's tool
  definitions from the request body (they are transport, not a generation
  param).
- **`aixon.runtime.current_client_tools()`** — the Server publishes the tools
  per request (same pattern as generation params); the agent reads them and
  decides to call `open_file`.
- **`Message.tool_calls` / `Chunk.tool_calls` on the wire** — the adapter
  emits `finish_reason: "tool_calls"` and OpenAI-shaped `tool_calls`, both
  non-stream and as split stream deltas.
- **The second turn** — the client posts the tool result back
  (`role: "tool"`); the adapter parses the history's `tool_calls` into
  neutral form and the agent answers in text.

## Expected output

```
== turn 1: editor -> server (with tools) ==
finish_reason: tool_calls
tool_call: open_file {"path": "/home/user/report.docx"}

== the editor executes the call locally ==
result: {"status": "success", "opened": "/home/user/report.docx"}

== turn 2: editor -> server (with the tool result) ==
finish_reason: stop
answer: Done — the client reported: {"status": "success", "opened": "/home/user/report.docx"}
```

The `FileButlerAgent` here is scripted so the example is deterministic and
offline; in a real deployment the same routing lives around an LLM — read
`current_client_tools()`, bind them to the model, surface the model's calls
as `tool_calls`, and let the client execute.

## Modo merge (#18c) — `ToolAgent(client_tools="merge")`

`main.py` above shows the RAW passthrough (`LLMAgent(client_tools=True)`,
#18a): the agent itself decides when to answer with `tool_calls`. `ToolAgent`
has a first-class alternative — `client_tools="merge"` (or `"replace"`) puts
the client's declared tools in the SAME tool-calling loop as the agent's own
tools, with the model free to call either kind in one run:

```
PYTHONPATH=../.. python merge_demo.py
```

**What it demonstrates**

- An INTERNAL tool (`buscar_orcamento`) executes server-side inside the run,
  same as any normal `ToolAgent` tool call.
- A CLIENT tool (`inserir_no_documento`, declared via
  `aixon.runtime.client_tools(...)`, same contextvar the Server publishes
  per request) ends the turn immediately — the run returns
  `Message(role="assistant", content="", tool_calls=[...])` instead of
  trying to execute it.
- The **resume** round-trip: the "editor" executes the call itself and posts
  a second request with `assistant(tool_calls=[...])` + `role="tool"` (the
  result) appended to the history; the model sees the result and concludes
  in text.

Request/response diagram (see [docs/server.md](../../docs/server.md) for the
full `client_tools` × `client_tools_conflict` reference table):

```
request 1  → user: "busque o orçamento e insira no documento"
         (internal tool buscar_orcamento runs here, server-side)
         ←  assistant, tool_calls=[inserir_no_documento(...)]   # finish_reason: tool_calls

[the editor executes inserir_no_documento itself]

request 2  → ...history..., assistant(tool_calls=[...]), tool(result="ok")
         ←  assistant: "Inserido com sucesso no documento."      # finish_reason: stop
```

**Expected output**

```
== request 1: editor -> agent (com tools=[inserir_no_documento]) ==
role: assistant
tool_calls: [{'name': 'inserir_no_documento', 'args': {'texto': 'Orçamento de licenças: R$ 4.200,00'}, 'id': 'c2'}]

== o editor executa a call localmente ==
resultado: ok — texto inserido

== request 2: editor -> agent (histórico + role=tool) ==
role: assistant
content: Inserido com sucesso no documento.
```
