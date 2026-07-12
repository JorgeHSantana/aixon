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
from typing import TYPE_CHECKING, AsyncIterator, Iterator

from aixon.agent import Agent
from aixon.exceptions import AixonError
from aixon.logging import Logger
from aixon.message import Chunk, Message
from aixon.reasoning import current_channel, emit_reasoning, reasoning_channel
from aixon._interop.tools import coerce_tools

if TYPE_CHECKING:
    from aixon.llm import LLM

_log = Logger("aixon.tool_agent")


class ToolAgent(Agent, abstract=True):
    """Tool-calling agent. Declarative attributes:

        class Diagnosis(ToolAgent):
            llm = LLM("gpt-4o-mini", temperature=0.1)
            prompt = "..."
            tools = [LibraryRetriever.as_tool(), check_battery]

    ``max_iterations`` maps to LangGraph's per-invocation ``recursion_limit``
    (a model+tool pair plus the final model turn per iteration).
    ``max_execution_time`` is a wall-clock **deadline**, not an interrupt:
    ``invoke`` checks it after the run completes and raises ``AixonError`` if
    exceeded; ``stream`` checks it between updates. Neither can abort a single
    in-flight tool call (LangGraph's compiled graph has no time knob)."""

    _suffix = "Agent"

    llm: "LLM | None" = None  # REQUIRED LLM instance on concrete subclasses
    prompt: str = ""
    tools: list = []
    max_iterations: int = 15
    max_execution_time: int = 600
    tool_call_label: str = "Calling {name}..."  # reasoning label per tool call; {name} = tool name

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
        system (or developer — OpenAI's system-role alias) message overrides
        self.prompt."""
        from langchain.agents import create_agent
        from aixon._interop.messages import to_langchain

        system_prompt = self.prompt or None
        if messages and messages[0].role in ("system", "developer"):
            system_prompt = messages[0].content or system_prompt
            messages = messages[1:]

        lc_tools = coerce_tools(list(self.tools))
        # _validate_subclass() (__init_subclass__ hook, above) already refuses
        # to register any concrete ToolAgent subclass with `llm=None`.
        assert self.llm is not None
        agent = create_agent(self.llm.request_chat_model(), lc_tools, system_prompt=system_prompt)
        lc_messages = to_langchain(messages)
        config = {"recursion_limit": 2 * self.max_iterations + 1}
        return agent, lc_messages, config

    def _emit_tool_call_labels(self, message) -> None:
        """If an AI message carries tool calls, emit one reasoning label per
        call into the active ReasoningChannel (the langgraph-native equivalent
        of olympus' on_agent_action callback). The label text comes from
        ``self.tool_call_label``, a ``{name}``-templated string subclasses may
        override (e.g. for a friendlier phrase or i18n).

        Consecutive duplicates are skipped: a run that calls the same tool N
        times in a row (or a label that doesn't interpolate ``{name}``) would
        otherwise spam N identical lines, which reads as noise — or a hang —
        in chat UIs that render reasoning. The comparison is against the
        channel's last line ever emitted (``ReasoningChannel.last``), so the
        dedupe also holds across streaming drain boundaries and across
        parent/nested agents sharing one channel."""
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if not name:
                continue
            label = self.tool_call_label.format(name=name)
            channel = current_channel()
            if channel is not None and channel.last == label:
                continue
            emit_reasoning(label)

    @staticmethod
    def _sum_usage(new_messages) -> dict | None:
        """Sum provider-real usage over the AI messages produced by THIS run
        (a multi-step run bills every model turn, not just the final answer).
        Returns a neutral OpenAI-shaped dict, or None when no AI message
        carried ``usage_metadata`` (the server then falls back to estimating)."""
        from aixon._interop.messages import usage_from_metadata

        totals: dict | None = None
        for m in new_messages:
            if getattr(m, "type", "") != "ai":
                continue
            usage = usage_from_metadata(getattr(m, "usage_metadata", None))
            if usage is None:
                continue
            if totals is None:
                totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for key in totals:
                totals[key] += usage[key]
        return totals

    def _iteration_limit_error(self, exc: Exception) -> AixonError:
        """AixonError for an exhausted iteration budget (LangGraph's recursion
        limit), matching the Orchestrator's wrapping style: the neutral
        boundary must not leak a raw GraphRecursionError naming an internal
        recursion limit the user never set."""
        return AixonError(
            f"agent '{self.name}' hit its iteration limit (max_iterations="
            f"{self.max_iterations}) without producing a final answer. Raise "
            f"`max_iterations` or simplify the task. (LangGraph: {exc})"
        )

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

        from langgraph.errors import GraphRecursionError

        agent, lc_messages, config = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        outer_channel = current_channel()
        cm = nullcontext(outer_channel) if outer_channel is not None else reasoning_channel()
        with cm as channel:
            try:
                result = agent.invoke({"messages": lc_messages}, config=config)
            except GraphRecursionError as exc:
                raise self._iteration_limit_error(exc) from exc
            # Derive parent tool-call labels from the AI messages produced by
            # THIS run only. LangGraph's add_messages reducer preserves the
            # input prefix in result["messages"], so iterating the full list
            # would re-emit a label for an AI(tool_calls) message that already
            # belongs to a prior turn in the caller's history.
            for m in result["messages"][len(lc_messages):]:
                if getattr(m, "type", "") == "ai":
                    self._emit_tool_call_labels(m)
            if time.monotonic() > deadline:
                # Post-hoc deadline: checked after the run completes (it does NOT
                # interrupt an in-flight tool call). Raise so an over-budget run
                # is rejected rather than silently returned. For between-step
                # enforcement, use stream(), which breaks between updates.
                raise AixonError(
                    f"agent '{self.name}' exceeded max_execution_time "
                    f"({self.max_execution_time}s)."
                )
            # Only drain (and consume) the lines if we own this channel. When
            # nested, leave them in the outer channel for its owner to drain.
            reasoning_lines = [] if outer_channel is not None else channel.drain()
        final = from_langchain(result["messages"][-1])
        # Usage must cover the WHOLE run (every model turn), not just the
        # final message that from_langchain converted above.
        final.usage = self._sum_usage(result["messages"][len(lc_messages):])
        if reasoning_lines:
            final.reasoning = "\n".join(reasoning_lines)
        _log.info(f"agent '{self.name}' completed ({len(reasoning_lines)} step(s))")
        return final

    # ---- neutral boundary: stream ---------------------------------------

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Stream the run: Chunk(reasoning=...) for tool-step labels (parent +
        nested) and Chunk(content=...) for the final answer; final
        Chunk(done=True)."""
        from aixon._interop.messages import _flatten_content

        from langgraph.errors import GraphRecursionError

        agent, lc_messages, config = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        final_content = ""
        with reasoning_channel() as channel:
            try:
                for update in agent.stream(
                    {"messages": lc_messages}, config=config, stream_mode="updates"
                ):
                    # Each update is {node_name: {"messages": [...]}}.
                    for node_state in update.values():
                        for m in node_state.get("messages", []) or []:
                            if getattr(m, "type", "") == "ai":
                                self._emit_tool_call_labels(m)
                                # Only an AI message WITHOUT tool calls is a
                                # final answer; content on a tool-calling
                                # message is preamble thought and must not be
                                # surfaced as the answer on a deadline break.
                                if not (getattr(m, "tool_calls", None) or []):
                                    content = _flatten_content(getattr(m, "content", ""))
                                    if content:
                                        final_content = content
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
            except GraphRecursionError as exc:
                raise self._iteration_limit_error(exc) from exc
            # Any trailing reasoning emitted after the last update.
            for line in channel.drain():
                yield Chunk(reasoning=line + "\n")
        if final_content:
            yield Chunk(content=final_content)
        yield Chunk(done=True)

    # ---- async neutral boundary -----------------------------------------

    async def ainvoke(self, messages: list[Message]) -> Message:
        """Async invoke over LangGraph's native ``ainvoke``. ``max_execution_time``
        becomes a REAL deadline here: the run is wrapped in ``asyncio.wait_for``
        and cancelled at the next await point if it overruns (unlike sync
        ``invoke``, whose deadline is only post-hoc)."""
        import asyncio
        from contextlib import nullcontext

        from langgraph.errors import GraphRecursionError

        from aixon._interop.messages import from_langchain

        agent, lc_messages, config = self._build_agent(messages)
        outer_channel = current_channel()
        cm = nullcontext(outer_channel) if outer_channel is not None else reasoning_channel()
        with cm as channel:
            try:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": lc_messages}, config=config),
                    timeout=self.max_execution_time,
                )
            except asyncio.TimeoutError:
                raise AixonError(
                    f"agent '{self.name}' exceeded max_execution_time "
                    f"({self.max_execution_time}s); the run was cancelled."
                )
            except GraphRecursionError as exc:
                raise self._iteration_limit_error(exc) from exc
            # See the sync invoke() comment: only THIS run's new messages.
            for m in result["messages"][len(lc_messages):]:
                if getattr(m, "type", "") == "ai":
                    self._emit_tool_call_labels(m)
            reasoning_lines = [] if outer_channel is not None else channel.drain()
        final = from_langchain(result["messages"][-1])
        # See the sync invoke() comment: usage covers the WHOLE run.
        final.usage = self._sum_usage(result["messages"][len(lc_messages):])
        if reasoning_lines:
            final.reasoning = "\n".join(reasoning_lines)
        _log.info(f"agent '{self.name}' completed async ({len(reasoning_lines)} step(s))")
        return final

    async def astream(self, messages: list[Message]) -> AsyncIterator[Chunk]:
        """Async stream mirroring ``stream`` over the graph's ``astream``.

        ``max_execution_time`` is a HARD wall here, not just a between-update
        check: each step is awaited under ``asyncio.wait_for`` with the time
        remaining until the deadline, so a step that stalls mid-flight (e.g. a
        provider stream that stops delivering bytes) is cancelled at the
        deadline instead of hanging the request forever. The underlying graph
        stream is closed on the way out.
        """
        import asyncio

        from langgraph.errors import GraphRecursionError

        from aixon._interop.messages import _flatten_content

        agent, lc_messages, config = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        final_content = ""
        timed_out = False
        with reasoning_channel() as channel:
            stream = agent.astream(
                {"messages": lc_messages}, config=config, stream_mode="updates"
            )
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    try:
                        update = await asyncio.wait_for(
                            stream.__anext__(), timeout=remaining
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        timed_out = True
                        break
                    except GraphRecursionError as exc:
                        raise self._iteration_limit_error(exc) from exc
                    for node_state in update.values():
                        for m in node_state.get("messages", []) or []:
                            if getattr(m, "type", "") == "ai":
                                self._emit_tool_call_labels(m)
                                # Only an AI message WITHOUT tool calls is a
                                # final answer (see stream): preamble thought
                                # on a tool-calling message must not become
                                # the answer when the deadline breaks the run.
                                if not (getattr(m, "tool_calls", None) or []):
                                    content = _flatten_content(getattr(m, "content", ""))
                                    if content:
                                        final_content = content
                    for line in channel.drain():
                        yield Chunk(reasoning=line + "\n")
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass
            if timed_out:
                emit_reasoning(
                    f"(stopped: exceeded max_execution_time "
                    f"{self.max_execution_time}s)"
                )
            for line in channel.drain():
                yield Chunk(reasoning=line + "\n")
        if final_content:
            yield Chunk(content=final_content)
        yield Chunk(done=True)
