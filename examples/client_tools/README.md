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
