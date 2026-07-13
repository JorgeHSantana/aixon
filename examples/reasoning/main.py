"""Reasoning — the `LLM(model, reasoning=...)` knob, live against Anthropic.

Turns on Claude's extended thinking with a token budget and shows the real
chain-of-thought arriving on the neutral `Chunk.reasoning` field, separately
from `Chunk.content` — first on a bare `LLMAgent`, then on a `ToolAgent`,
where the model's own thinking flows through the live reasoning channel
BEFORE the "Calling {tool}..." step label it led to.

Needs a real key and one cheap model (claude-haiku-4-5):

    export ANTHROPIC_API_KEY=sk-ant-...
    cd examples/reasoning
    python main.py

See README.md for the expected output and the per-provider honesty notes
(visible thinking is an Anthropic/Gemini thing; OpenAI improves silently).
"""

from __future__ import annotations

from aixon import LLM, LLMAgent, Message, ToolAgent

# ── the knob ─────────────────────────────────────────────────────────────────
# reasoning={"budget_tokens": 2048} → Anthropic `thinking={"type": "enabled",
# "budget_tokens": 2048}`. The provider FORCES temperature=1 (Anthropic's
# extended-thinking API requires it — a warning is logged if you asked for
# something else) and raises max_tokens above the budget when needed.
# `reasoning=True` would be the {"effort": "medium"} (4096-token) shorthand.
# See docs/agents.md, "Reasoning (extended thinking / reasoning effort)".

REASONING_LLM = LLM("claude-haiku-4-5", reasoning={"budget_tokens": 2048})


# ── 1. LLMAgent: thinking before the answer ──────────────────────────────────


class RiddlerAgent(LLMAgent):
    """Direct LLM call — reasoning shows up as `Chunk.reasoning` deltas."""

    name = "riddler"
    hidden = True
    description = "Answers riddles, thinking out loud first."
    llm = REASONING_LLM
    prompt = "Answer the riddle. Be brief."


# ── 2. ToolAgent: thinking flows in the live channel BEFORE the tool label ──


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


class CalculatorAgent(ToolAgent):
    """Tool loop — the model's own thinking is emitted into the live
    ReasoningChannel ahead of that turn's "Calling multiply..." label: the
    model literally reasoned before deciding to call the tool."""

    name = "calculator"
    hidden = True
    description = "Multiplies numbers with a local tool."
    llm = REASONING_LLM
    prompt = "Use the multiply tool for any multiplication. Answer briefly."
    tools = [multiply]


# ── driver ───────────────────────────────────────────────────────────────────


def stream_and_print(agent, question: str) -> None:
    """Stream one run, printing reasoning (dimmed, "[thinking]"-prefixed)
    separately from content — the neutral Chunk carries them apart, so a
    consumer never has to parse thinking out of the answer text."""
    print(f"> {question}\n")
    for chunk in agent.stream([Message(role="user", content=question)]):
        if chunk.reasoning:
            for line in chunk.reasoning.splitlines():
                print(f"\033[2m[thinking] {line}\033[0m")
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


def invoke_and_print_usage(agent, question: str) -> None:
    """Non-stream invoke: `message.reasoning` carries the thinking text and
    `message.usage` the provider's REAL token counts. Thinking tokens bill as
    OUTPUT tokens — they are already inside `completion_tokens`; there is no
    separate meter to watch."""
    message = agent.invoke([Message(role="user", content=question)])
    print(f"> {question}")
    print(f"  answer: {message.content}")
    print(f"  usage:  {message.usage}  (thinking bills as output tokens)\n")


def main() -> None:
    print("── 1. LLMAgent: extended thinking streams before the answer ──\n")
    stream_and_print(
        RiddlerAgent(),
        "I speak without a mouth and hear without ears. What am I?",
    )
    invoke_and_print_usage(RiddlerAgent(), "What has keys but can't open locks?")

    print("── 2. ToolAgent: thinking arrives BEFORE the tool-call label ──\n")
    stream_and_print(CalculatorAgent(), "What is 1337 times 42?")
    invoke_and_print_usage(CalculatorAgent(), "What is 256 times 64?")


# ── 3. Per-request override over the server (not executed here) ─────────────
# Behind `aixon serve`, the request body's `reasoning_effort` is allow-listed
# like temperature/max_tokens and OVERRIDES the class-level knob for that one
# request — translated as {"effort": "high"} (16384-token budget):
#
#   curl -s http://localhost:8000/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model": "riddler",
#          "messages": [{"role": "user", "content": "Why is the sky blue?"}],
#          "reasoning_effort": "high"}'
#
# Or with the OpenAI client (any OpenAI-compatible client works):
#
#   client.chat.completions.create(
#       model="riddler",
#       messages=[{"role": "user", "content": "Why is the sky blue?"}],
#       extra_body={"reasoning_effort": "high"},
#   )
#
# On a stream, `thought_stream_mode` (docs/server.md) picks how the reasoning
# reaches the wire: a separate `delta.reasoning` field, inline <think> tags,
# or hidden.


if __name__ == "__main__":
    main()
