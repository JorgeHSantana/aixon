# Reflective Review — a runnable `ReflectiveAgent` example

A minimal, **fully offline** demonstration of `ReflectiveAgent`, the
evaluator-optimizer loop documented in [docs/agents.md](../../docs/agents.md#reflectiveagent--evaluator-optimizer-loop):
a judge LLM checks a generated answer against an objective rubric and, if it
falls short, sends the critique back to the generator for another attempt —
up to `max_rounds` times.

No API key, no network call: both the generator and the judge are scripted
(deterministic, offline chat-model/agent doubles), so `python main.py` just
works.

## What it demonstrates

| Element | Where |
|---|---|
| A custom `Provider` + scripted `BaseChatModel` (the offline judge) | [main.py](main.py) — `ScriptedChatModel` / `ScriptedProvider` |
| A plain scripted `Agent` as the worker (`ReflectiveAgent.agent` accepts any `Agent`, class or instance) | [main.py](main.py) — `DraftWriterAgent` |
| `class ReviewedWriterAgent(ReflectiveAgent)` with an objective `judge_rubric` | [main.py](main.py) |
| `stream()` surfacing the loop's reasoning labels (`judge_label` / `retry_label`) | [main.py](main.py) — `main()` |
| A real reject-then-approve round trip (not just the happy path) | the printed output below |
| `revision_mode = "patch"` (0.1.19): the retry emits a SEARCH/REPLACE block, applied programmatically — raw patch text never reaches the stream | [main.py](main.py) — `PatchWriterAgent` / `PatchReviewedWriterAgent` |

## Run it

```bash
cd examples/reflective_review
python main.py
```

Install the single dependency first (or use the repo venv, where `aixon`
is already importable):

```bash
pip install -r requirements.txt   # just `aixon` — everything else is inline
```

## What to expect

`DraftWriterAgent` is scripted to answer "Fortaleza is the capital of Ceará."
the first time (no source) and "Fortaleza is the capital of Ceará (source:
IBGE)." the second time. The judge is scripted to reject the first answer
("does not cite a source") and approve the second (`APROVADO`). Running the
example drives that loop for real and prints the reasoning as it happens:

```
> What is the capital of Ceará?

[reasoning] Avaliando a resposta…
[reasoning] Refinando a resposta (rodada 2/3)…
[reasoning] Avaliando a resposta…

Final answer: Fortaleza is the capital of Ceará (source: IBGE).

DraftWriterAgent was called 2 time(s) — the judge rejected round 1 (no source) and approved round 2.
```

A second section then runs the SAME loop with `revision_mode = "patch"`
(`PatchReviewedWriterAgent`): the retry answers with a `<<<<<<< SEARCH /
======= / >>>>>>> REPLACE` block, the agent applies it over the previous
answer, and the judge approves the PATCHED text — same final answer, but the
worker only paid for the edit, not a full rewrite. A patch that doesn't apply
falls back to full regeneration automatically.

(The reasoning labels are Portuguese — `ReflectiveAgent`'s defaults; override
`judge_label`/`retry_label`/`exhausted_label`/`patch_fallback_label` on your
own subclass to customize or translate them.)

## Retries got cheaper in 0.1.19 (automatic)

Beyond `revision_mode`, three optimizations apply to every `ReflectiveAgent`
without code changes: the prompt prefix is byte-stable across rounds (OpenAI
prompt caching applies; `LLM(..., cache=True)` adds Anthropic breakpoints),
tool calls repeated with identical args are memoized for the run
(`aixon.toolcache` — see [examples/tool_shield_memo](../tool_shield_memo)),
and OpenAI workers receive the previous attempt as a Predicted Output on
retries (latency win). Details: CHANGELOG 0.1.19 and
[docs/agents.md](../../docs/agents.md#reflectiveagent--evaluator-optimizer-loop).

## Make it real

* **Generator:** swap `DraftWriterAgent` for a real `LLMAgent`/`ToolAgent` —
  `ReflectiveAgent.agent` just needs an `Agent` (class or instance); nothing
  else in `ReviewedWriterAgent` changes.
* **Judge:** swap `judge_llm = scripted_llm([...])` for a real model, e.g.
  `judge_llm = LLM("gpt-4o-mini", temperature=0)` — cheaper/faster models
  work well as judges since the task is classification, not generation.
* **Rubric:** `judge_rubric` should stay objective and checkable ("cites a
  source", "the returned SQL was validated", "every claimed number matches a
  tool result") — a vague rubric ("sounds good") degenerates into the judge
  rubber-stamping everything.
