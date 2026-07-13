# Reasoning — a runnable `LLM(reasoning=...)` example

A minimal, live demonstration of the declarative reasoning knob documented in
[docs/agents.md](../../docs/agents.md#reasoning-extended-thinking--reasoning-effort):
`LLM("claude-haiku-4-5", reasoning={"budget_tokens": 2048})` turns on Claude's
extended thinking, and the model's real chain-of-thought arrives on the
neutral `Chunk.reasoning` / `Message.reasoning` fields — separate from the
answer, never mixed into `content`.

Unlike the other examples, this one calls a **real provider** (extended
thinking is a provider feature — there is nothing meaningful to fake), so it
needs `ANTHROPIC_API_KEY`. It uses `claude-haiku-4-5` with a 2048-token
budget to stay cheap.

## What it demonstrates

| Element | Where |
|---|---|
| The knob: `LLM("claude-haiku-4-5", reasoning={"budget_tokens": 2048})` | [main.py](main.py) — `REASONING_LLM` |
| `LLMAgent` streaming: `Chunk.reasoning` (thinking) before `Chunk.content` | [main.py](main.py) — `RiddlerAgent` / `stream_and_print` |
| `ToolAgent`: the model's thinking in the live channel BEFORE the `"Calling multiply..."` label | [main.py](main.py) — `CalculatorAgent` |
| `Message.usage` on a non-stream `invoke` — thinking bills as output tokens | [main.py](main.py) — `invoke_and_print_usage` |
| Per-request `reasoning_effort` override over the server (commented, not executed) | [main.py](main.py) — section 3 |

## Run it

```bash
pip install -r requirements.txt   # aixon + langchain-anthropic
export ANTHROPIC_API_KEY=sk-ant-...
cd examples/reasoning
python main.py
```

## What to expect

Model output varies run to run (thinking doubly so); the shape is:

```
── 1. LLMAgent: extended thinking streams before the answer ──

> I speak without a mouth and hear without ears. What am I?

[thinking] This is a classic riddle. Something that "speaks" without a
[thinking] mouth and "hears" without ears... an echo repeats sounds back...
An echo.

> What has keys but can't open locks?
  answer: A piano.
  usage:  {'prompt_tokens': 21, 'completion_tokens': 187, 'total_tokens': 208}  (thinking bills as output tokens)

── 2. ToolAgent: thinking arrives BEFORE the tool-call label ──

> What is 1337 times 42?

[thinking] The user wants 1337 × 42. I should use the multiply tool.
[thinking] Calling multiply...
1337 × 42 = 56,154.
```

Note the ordering in part 2: the model's own thinking is emitted into the
`ReasoningChannel` *before* the tool-call step label — the model reasoned
before deciding to call the tool, and the channel preserves that order.

The `completion_tokens` count is noticeably larger than the visible answer:
**thinking tokens bill as output tokens** and are already included there —
`Message.usage` carries the provider's real counts, no separate meter.

## Honesty notes — which providers show what

The knob translates per provider (full table in
[docs/agents.md](../../docs/agents.md#reasoning-extended-thinking--reasoning-effort)),
but *visible* reasoning text differs:

* **Anthropic** (this example) and **Gemini** (`include_thoughts`) return
  real thinking text — it lands on `Message.reasoning` / `Chunk.reasoning`.
* **OpenAI** never returns raw chain-of-thought: `reasoning_effort` makes the
  model think harder and improves the answer, but `Message.reasoning` stays
  `None` — the improvement is silent.
* **z.AI (GLM)**: thinking IS enabled on the wire, but the installed
  `langchain-openai` does not populate the `reasoning_content` response field
  today, so GLM reasoning text does not surface yet (a provider-side gap;
  aixon's extraction already handles the convention the moment it appears).

## Make it real

* **Effort instead of a budget:** `reasoning={"effort": "high"}` (16384
  tokens) — or just `reasoning=True` for the medium default (4096).
* **Per request:** serve the agents (`aixon serve`) and send
  `"reasoning_effort": "high"` in the request body — it overrides the class
  knob for that one request (see section 3 in [main.py](main.py) and
  [docs/server.md](../../docs/server.md)).
* **Streaming over the wire:** `thought_stream_mode` on the OpenAI adapter
  picks how reasoning reaches the client — a separate `delta.reasoning`
  field, inline `<think>` tags, or hidden ([docs/server.md](../../docs/server.md)).
