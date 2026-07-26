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

_CLIENT_TOOL_STOP = "__aixon_client_tool_call__"


def _client_proxy_tools(defs: list[dict]) -> list:
    """One ``return_direct`` ``StructuredTool`` per client tool def (#18c).

    The proxy body is a stub returning a sentinel — the LangGraph loop never
    actually "does" the client's tool. ``return_direct`` (the OnlyOffice
    native-bridge precedent) ends the graph right after the call when ALL of
    the turn's calls are proxies; on a MIXED turn (internal + client call in
    the same AI message) LangChain keeps looping, so the post-run detection
    (``_surface_client_calls``) does not rely on the graph ending — it scans
    the run's AI messages for the first turn that called a client tool and
    surfaces those calls as the neutral ``Message.tool_calls`` the CLIENT
    must execute."""
    from langchain_core.tools import StructuredTool

    proxies = []
    for d in defs:
        fn = (d.get("function") or {})
        name = fn.get("name")
        if not name:
            continue

        def _stop(**kwargs):  # noqa: ANN003 — schema comes from the client def
            return _CLIENT_TOOL_STOP

        proxies.append(StructuredTool.from_function(
            func=_stop, name=name,
            description=fn.get("description") or name,
            args_schema=fn.get("parameters") or None,
            return_direct=True,
        ))
    return proxies


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
    # Error shield (#9): True (default) converts ANY exception raised by a tool
    # into a readable error result handed back to the model, so one failing
    # tool/service reports instead of killing the whole run/stream. Set False
    # for a strict policy where tool exceptions propagate (pre-#9 behavior).
    shield_tool_errors: bool = True
    tool_call_label: str = "Calling {name}..."  # reasoning label per tool call; {name} = tool name
    # Context pruning (#16): with an int N, tool results OLDER than the last N
    # assistant turns are replaced by a short stub when building the payload —
    # already-consumed query dumps stop re-billing every turn. None = off.
    prune_tool_results_after: int | None = None

    # Client tools (#18c): "ignore" (default) | "merge" (client defs join the
    # internal tools; a CLIENT call ends the turn with Message.tool_calls) |
    # "replace" (client defs only for this request).
    client_tools: str = "ignore"
    # Name collision between an internal tool and a client def:
    # "error" (explicit, default) | "internal" (client def dropped) |
    # "client" (internal tool dropped).
    client_tools_conflict: str = "error"

    def client_tools_filter(self, defs: list[dict]) -> list[dict]:
        """Curation hook (#18c): which client tool defs are exposed. Default:
        all of them. Override e.g. to allow only a prefix."""
        return defs

    # Tool-call hooks (#17): override for deterministic guardrails/telemetry.
    # on_tool_start may return a dict to REWRITE the call's kwargs; raising in
    # it is treated as the tool failing (shield applies). on_tool_end is
    # observation-only — its exceptions are logged and swallowed.
    def on_tool_start(self, name: str, args: dict):  # noqa: D401
        """Called before a tool executes. Return a dict to rewrite ``args``
        (the memo cache lookup runs on the rewritten kwargs); return None to
        keep them unchanged. Raising here is treated as the tool call itself
        failing — the error shield (#9) converts it to a TOOL ERROR result
        and the run continues. Default: no-op."""
        return None

    def on_tool_end(self, name: str, args: dict, result, error) -> None:
        """Called after a tool call, whether it succeeded, failed, or was
        served from the memo cache (``error`` is the exception in the failure
        case, else None). Observation-only: any exception raised here is
        logged as a warning and swallowed — it never affects the tool result
        seen by the model. Default: no-op."""
        return None

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
        # #16: values <= 0 are INVALID configuration, rejected at registration
        # (same precedent as ReflectiveAgent's max_rounds). keep_turns=0 would
        # hit `assistant_idx[-0]` == `assistant_idx[0]` (Python: -0 == 0) — an
        # IndexError on a history with no assistant message yet, and a
        # near-no-op otherwise; both the opposite of the attribute's name.
        prune = getattr(cls, "prune_tool_results_after", None)
        if prune is not None and (not isinstance(prune, int) or prune < 1):
            raise AixonError(
                f"'{cls.__name__}' has prune_tool_results_after={prune!r}; "
                f"use None (off) or an int >= 1 (keep the last N assistant turns)."
            )
        # #18c: client_tools/client_tools_conflict are declarative enums,
        # validated at registration so a typo fails loudly instead of
        # silently behaving as "ignore"/"error" at request time.
        mode = getattr(cls, "client_tools", "ignore")
        if mode not in ("ignore", "merge", "replace"):
            raise AixonError(
                f"'{cls.__name__}' has client_tools={mode!r}; use 'ignore', "
                f"'merge' or 'replace'."
            )
        conflict = getattr(cls, "client_tools_conflict", "error")
        if conflict not in ("error", "internal", "client"):
            raise AixonError(
                f"'{cls.__name__}' has client_tools_conflict={conflict!r}; "
                f"use 'error', 'internal' or 'client'."
            )

    # ---- internal: build the langgraph agent + neutral message prep -------

    @staticmethod
    def _prune_history(messages: list[Message], keep_turns: int) -> list[Message]:
        """Copy of *messages* with tool results before the last *keep_turns*
        assistant messages stubbed out (#16). Never mutates the caller's list."""
        import dataclasses

        assistant_idx = [i for i, m in enumerate(messages) if m.role == "assistant"]
        cutoff = assistant_idx[-keep_turns] if len(assistant_idx) >= keep_turns else 0
        out: list[Message] = []
        for i, m in enumerate(messages):
            if m.role == "tool" and i < cutoff and m.content:
                stub = (f"[resultado de ferramenta omitido "
                        f"({len(m.content)} caracteres) — já utilizado em "
                        f"resposta anterior]")
                out.append(dataclasses.replace(m, content=stub))
            else:
                out.append(m)
        return out

    def _build_agent(self, messages: list[Message]):
        """Return (compiled_agent, lc_messages, config, client_tool_names).
        A leading neutral system (or developer — OpenAI's system-role alias)
        message overrides self.prompt.

        ``client_tool_names`` (#18c) is the set of CLIENT tool names proxied
        into this build (empty when ``client_tools="ignore"`` or the request
        declared none). It is RETURNED — never stored on ``self`` — because
        the agent instance is a registry singleton shared by concurrent
        requests: request-scoped state on the instance would let request B
        (running ``_build_agent`` during one of A's awaits) overwrite request
        A's set — including resetting it to empty for a request without
        client tools — making A leak the proxy sentinel as content and lose
        its tool_calls."""
        from langchain.agents import create_agent
        from aixon._interop.messages import to_langchain

        if self.prune_tool_results_after is not None:
            messages = self._prune_history(messages, self.prune_tool_results_after)

        system_prompt = self.prompt or None
        if messages and messages[0].role in ("system", "developer"):
            system_prompt = messages[0].content or system_prompt
            messages = messages[1:]

        # Pass the hooks to coerce_tools only when a subclass overrides them
        # (#17) — avoids a dict-copy per tool call in the (default) no-hook
        # case, and keeps the guard's on_start/on_end branches unused/cheap.
        overridden = (type(self).on_tool_start is not ToolAgent.on_tool_start
                      or type(self).on_tool_end is not ToolAgent.on_tool_end)
        lc_tools = coerce_tools(
            list(self.tools), shield_errors=self.shield_tool_errors,
            on_tool_start=self.on_tool_start if overridden else None,
            on_tool_end=self.on_tool_end if overridden else None,
        )

        # Client tools (#18c): request-scoped by construction — computed on
        # every _build_agent call (the first step of all 4 entry points) and
        # handed back to the caller as a LOCAL value, never cached on the
        # (singleton) instance.
        client_tool_names: set[str] = set()
        if self.client_tools != "ignore":
            from aixon.runtime import current_client_tools

            defs = self.client_tools_filter(current_client_tools())
            if defs:
                internal_names = {t.name for t in lc_tools}
                client_names = {(d.get("function") or {}).get("name")
                                 for d in defs} - {None}
                clash = internal_names & client_names
                if clash and self.client_tools_conflict == "error":
                    raise AixonError(
                        f"client tool(s) {sorted(clash)} collide with internal "
                        f"tools of '{self.name}'. Set client_tools_conflict to "
                        f"'internal' or 'client', or filter the defs."
                    )
                if clash and self.client_tools_conflict == "internal":
                    defs = [d for d in defs
                            if (d.get("function") or {}).get("name") not in clash]
                if clash and self.client_tools_conflict == "client":
                    lc_tools = [t for t in lc_tools if t.name not in clash]
                proxies = _client_proxy_tools(defs)
                client_tool_names = {t.name for t in proxies}
                lc_tools = proxies if self.client_tools == "replace" else \
                    lc_tools + proxies

        # _validate_subclass() (__init_subclass__ hook, above) already refuses
        # to register any concrete ToolAgent subclass with `llm=None`.
        assert self.llm is not None
        agent = create_agent(self.llm.request_chat_model(), lc_tools, system_prompt=system_prompt)
        lc_messages = to_langchain(messages)
        config = {"recursion_limit": 2 * self.max_iterations + 1}
        return agent, lc_messages, config, client_tool_names

    def _emit_message_reasoning(self, message) -> None:
        """If a NEW AI message carries reasoning extracted by R2's
        ``reasoning_from_message`` (Anthropic-style thinking blocks, or the
        ``reasoning_content`` convention), emit it into the active
        ReasoningChannel. Called BEFORE ``_emit_tool_call_labels`` for the same
        message so the user sees the model's own thinking live, ahead of that
        turn's tool-call label(s) — the model literally reasoned before
        deciding to call the tool.

        Same consecutive-duplicate dedupe as ``_emit_tool_call_labels`` (against
        ``ReasoningChannel.last``), for the same reason: a provider that
        re-sends identical reasoning text must not spam the channel."""
        from aixon._interop.messages import reasoning_from_message

        reasoning = reasoning_from_message(message)
        if not reasoning:
            return
        channel = current_channel()
        if channel is not None and channel.last == reasoning:
            return
        emit_reasoning(reasoning)

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

    @staticmethod
    def _surface_client_calls(new_messages,
                              client_tool_names: set[str]) -> Message | None:
        """If THIS run's model called a CLIENT tool (#18c), rebuild the
        neutral assistant message carrying that turn's client tool_calls
        (original ``id``/``args``) so the wire adapter emits
        ``finish_reason="tool_calls"``. Returns ``None`` when no client tool
        was called (or none is configured).

        Pure static helper: ``new_messages`` must be the messages PRODUCED BY
        this run (invoke/ainvoke: ``result["messages"][len(lc_messages):]``;
        streams: the accumulated update messages) and ``client_tool_names``
        the set ``_build_agent`` returned — no request state ever lives on
        the (singleton, concurrently shared) agent instance. Scoping the scan
        to the run's NEW messages also keeps a resumed request correct: the
        history's own ``assistant(tool_calls)`` turn (the one the client
        already executed) is not in ``new_messages``, so it can't re-surface.

        The scan takes the FIRST AI message whose tool_calls include a client
        tool, not just a run that ENDED on a client-proxy ToolMessage. That
        covers the mixed turn (internal + client call in the SAME AI message):
        LangChain's ``return_direct`` only ends the graph when ALL of a turn's
        calls are return_direct, so a mixed turn keeps looping — the model
        sees the proxy's sentinel as a "result" and may answer in text. A
        proxy call never has a real result, so the client call must surface
        regardless; any turns the model generated after the sentinel are
        discarded (their cost is spent, but no fabricated "result" leaks)."""
        if not client_tool_names:
            return None
        for m in new_messages:
            if getattr(m, "type", "") != "ai":
                continue
            calls = [
                {"name": c.get("name"), "args": c.get("args") or {},
                 "id": c.get("id")}
                for c in (getattr(m, "tool_calls", None) or [])
                if c.get("name") in client_tool_names
            ]
            if calls:
                return Message(role="assistant", content="",
                               tool_calls=calls)
        return None

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

        agent, lc_messages, config, client_tool_names = self._build_agent(messages)
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
                    self._emit_message_reasoning(m)
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
        # Client tools (#18c): a run whose model called a CLIENT tool is NOT
        # a final answer — the client must execute the call. Scan only THIS
        # run's new messages (a resumed history's own tool_calls turn must
        # not re-surface) and discard anything the model produced after the
        # proxy sentinel.
        surfaced = self._surface_client_calls(
            result["messages"][len(lc_messages):], client_tool_names)
        if surfaced is not None:
            surfaced.usage = self._sum_usage(result["messages"][len(lc_messages):])
            return surfaced
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

        agent, lc_messages, config, client_tool_names = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        final_content = ""
        seen_messages: list = []
        with reasoning_channel() as channel:
            try:
                for update in agent.stream(
                    {"messages": lc_messages}, config=config, stream_mode="updates"
                ):
                    # Each update is {node_name: {"messages": [...]}}.
                    for node_state in update.values():
                        for m in node_state.get("messages", []) or []:
                            seen_messages.append(m)
                            if getattr(m, "type", "") == "ai":
                                self._emit_message_reasoning(m)
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
        # Client tools (#18c): a run whose model called a CLIENT tool is not
        # a final answer — surface those calls instead of any content the
        # model produced after the proxy sentinel, and skip the normal
        # content/done tail below.
        surfaced = self._surface_client_calls(seen_messages, client_tool_names)
        if surfaced is not None:
            yield Chunk(tool_calls=surfaced.tool_calls)
            yield Chunk(done=True)
            return
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

        agent, lc_messages, config, client_tool_names = self._build_agent(messages)
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
                    self._emit_message_reasoning(m)
                    self._emit_tool_call_labels(m)
            reasoning_lines = [] if outer_channel is not None else channel.drain()
        # See the sync invoke() comment: a client tool call surfaces as
        # Message.tool_calls; post-sentinel model output is discarded.
        surfaced = self._surface_client_calls(
            result["messages"][len(lc_messages):], client_tool_names)
        if surfaced is not None:
            surfaced.usage = self._sum_usage(result["messages"][len(lc_messages):])
            return surfaced
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

        agent, lc_messages, config, client_tool_names = self._build_agent(messages)
        deadline = time.monotonic() + self.max_execution_time
        final_content = ""
        timed_out = False
        seen_messages: list = []
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
                            seen_messages.append(m)
                            if getattr(m, "type", "") == "ai":
                                self._emit_message_reasoning(m)
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
        # Client tools (#18c): same short-circuit as stream() — a client
        # tool call surfaces as Chunk.tool_calls; post-sentinel content is
        # discarded.
        surfaced = self._surface_client_calls(seen_messages, client_tool_names)
        if surfaced is not None:
            yield Chunk(tool_calls=surfaced.tool_calls)
            yield Chunk(done=True)
            return
        if final_content:
            yield Chunk(content=final_content)
        yield Chunk(done=True)
