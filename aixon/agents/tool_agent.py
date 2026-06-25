"""ToolAgent — the tool-calling agent subtype, langgraph-native.

Builds a LangGraph agent with ``langchain.agents.create_agent`` (LangChain 1.x;
the removed 0.x ``create_tool_calling_agent`` + ``AgentExecutor`` and the
deprecated ``langgraph.prebuilt.create_react_agent`` are NOT used) over
``self.llm`` and the coerced tools, speaking ONLY neutral Message/Chunk at its
boundary.

Reasoning is surfaced through the contextvars-based ReasoningChannel (contract
§2.1): parent tool-call labels are derived from the graph's AI messages, and a
nested agent's ``emit_reasoning`` bubbles up because it targets the active
channel. langchain is imported lazily inside methods so importing ``aixon``
never requires it."""

from __future__ import annotations

import time
from typing import Iterator

from aixon.agent import Agent
from aixon.exceptions import AixonError
from aixon.logging import Logger
from aixon.message import Chunk, Message
from aixon.reasoning import current_channel, emit_reasoning, reasoning_channel
from aixon._interop.tools import coerce_tools

_log = Logger("aixon.tool_agent")


class ToolAgent(Agent, abstract=True):
    """Tool-calling agent. Declarative attributes:

        class Diagnosis(ToolAgent):
            llm = LLM("gpt-4o-mini", temperature=0.1)
            prompt = "..."
            tools = [LibraryRetriever.as_tool(), check_battery]

    ``max_iterations`` maps to LangGraph's per-invocation ``recursion_limit``
    (a model+tool pair plus the final model turn per iteration);
    ``max_execution_time`` is a wall-clock backstop enforced here (LangGraph's
    compiled graph has no built-in time knob)."""

    _suffix = "Agent"

    llm = None  # REQUIRED LLM instance on concrete subclasses
    prompt: str = ""
    tools: list = []
    max_iterations: int = 15
    max_execution_time: int = 600

    @classmethod
    def _validate_subclass(cls) -> None:
        # Validate the required declarative LLM on concrete subclasses. This
        # overrides Agent._validate_subclass (a hook the base calls AFTER suffix/
        # abstract-method checks and BEFORE registration), so a missing `llm`
        # raises without leaving a ghost in the registry. Do NOT override
        # __init_subclass__ to validate after super() — that registers first,
        # then fails (the register-then-validate ghost bug). The hook fires only
        # for concrete subclasses, so no abstract=True guard is needed here.
        if getattr(cls, "llm", None) is None:
            raise AixonError(
                f"ToolAgent subclass '{cls.__name__}' must declare an `llm` "
                f"attribute (e.g. `llm = LLM(\"gpt-4o-mini\")`). It was missing or None."
            )

    # ---- internal: build the langgraph agent + neutral message prep -------

    def _build_agent(self, messages: list[Message]):
        """Return (compiled_agent, lc_messages, config). A leading neutral
        system message overrides self.prompt."""
        from langchain.agents import create_agent
        from aixon._interop.messages import to_langchain

        system_prompt = self.prompt or None
        if messages and messages[0].role == "system":
            system_prompt = messages[0].content or system_prompt
            messages = messages[1:]

        lc_tools = coerce_tools(list(self.tools))
        agent = create_agent(self.llm.chat_model, lc_tools, system_prompt=system_prompt)
        lc_messages = to_langchain(messages)
        config = {"recursion_limit": 2 * self.max_iterations + 1}
        return agent, lc_messages, config

    @staticmethod
    def _emit_tool_call_labels(message) -> None:
        """If an AI message carries tool calls, emit one reasoning label per
        call into the active ReasoningChannel (the langgraph-native equivalent
        of olympus' on_agent_action callback)."""
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if name:
                emit_reasoning(f"Calling {name}...")

    # ---- neutral boundary: invoke ---------------------------------------

    def invoke(self, messages: list[Message]) -> Message:
        """Run the tool-calling graph to completion; return a neutral Message.

        Reasoning collected during the run (tool-step labels, plus reasoning a
        nested agent emitted) is set on Message.reasoning. If a ReasoningChannel
        is already active (this agent is itself nested under an outer stream()
        or invoke()), reuse it so this agent's own emit_reasoning calls bubble
        to the outer channel instead of being shadowed by a fresh one; otherwise
        open a fresh channel so a nested agent's emit_reasoning is still
        captured when this agent is invoked directly (no outer stream)."""
        from aixon._interop.messages import from_langchain
        from contextlib import nullcontext

        agent, lc_messages, config = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        outer_channel = current_channel()
        cm = nullcontext(outer_channel) if outer_channel is not None else reasoning_channel()
        with cm as channel:
            result = agent.invoke({"messages": lc_messages}, config=config)
            # Derive parent tool-call labels from the AI messages in the result.
            for m in result["messages"]:
                if getattr(m, "type", "") == "ai":
                    self._emit_tool_call_labels(m)
            if time.monotonic() > deadline:
                _log.warning(
                    f"agent '{self.name}' exceeded max_execution_time "
                    f"({self.max_execution_time}s)"
                )
            # Only drain (and consume) the lines if we own this channel. When
            # nested, leave them in the outer channel for its owner to drain.
            reasoning_lines = [] if outer_channel is not None else channel.drain()
        final = from_langchain(result["messages"][-1])
        if reasoning_lines:
            final.reasoning = "\n".join(reasoning_lines)
        _log.info(f"agent '{self.name}' completed ({len(reasoning_lines)} step(s))")
        return final

    # ---- neutral boundary: stream ---------------------------------------

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Stream the run: Chunk(reasoning=...) for tool-step labels (parent +
        nested) and Chunk(content=...) for the final answer; final
        Chunk(done=True)."""
        agent, lc_messages, config = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        final_content = ""
        with reasoning_channel() as channel:
            for update in agent.stream(
                {"messages": lc_messages}, config=config, stream_mode="updates"
            ):
                # Each update is {node_name: {"messages": [...]}}.
                for node_state in update.values():
                    for m in node_state.get("messages", []) or []:
                        if getattr(m, "type", "") == "ai":
                            self._emit_tool_call_labels(m)
                            if getattr(m, "content", ""):
                                final_content = m.content
                # Surface reasoning accrued since the last update (parent labels
                # + any nested-agent emit_reasoning) before yielding content.
                for line in channel.drain():
                    yield Chunk(reasoning=line + "\n")
                if time.monotonic() > deadline:
                    emit_reasoning(
                        f"(stopped: exceeded max_execution_time "
                        f"{self.max_execution_time}s)"
                    )
                    for line in channel.drain():
                        yield Chunk(reasoning=line + "\n")
                    break
            # Any trailing reasoning emitted after the last update.
            for line in channel.drain():
                yield Chunk(reasoning=line + "\n")
        if final_content:
            yield Chunk(content=final_content)
        yield Chunk(done=True)
