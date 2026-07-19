"""ReflectiveAgent — the evaluator-optimizer subtype (generate -> judge -> retry).

Wraps a worker agent in a review loop: a judge LLM scores each answer against a
declarative rubric; a rejected answer comes back to the worker together with the
judge's critique, up to ``max_rounds`` attempts. The verdict travels as a text
sentinel (first line ``APROVADO``), following the DELEGAR/END precedent.

The neutral boundary holds: the worker is called through ``Agent.invoke`` and
the judge through ``LLM.complete`` — Message/Chunk only, no provider types.
Exhausting the rounds is NOT an error: the last attempt is returned and the
``exhausted_label`` is emitted on the reasoning channel (neutral-error precept:
a quality shortfall must not crash a run that produced an answer)."""

from __future__ import annotations

import dataclasses
import re
from typing import Any, AsyncIterator, Iterator

from aixon.agent import Agent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.logging import Logger
from aixon.message import Chunk, Message
from aixon.reasoning import emit_reasoning
from aixon.runtime import prediction_scope
from aixon.toolcache import tool_call_cache
from aixon.usage import merge_usage

_log = Logger("aixon.reflective")

APPROVE_SENTINEL = "APROVADO"

_JUDGE_PROMPT = """You are a strict quality judge. Evaluate the assistant's \
answer against the rubric below — and ONLY the rubric; do not invent criteria.

RUBRIC:
{rubric}

If the answer satisfies EVERY rubric item, reply with exactly:
APROVADO

Otherwise reply with a numbered list of actionable critiques (what is wrong
and what to change), nothing else. Never include the word APROVADO in a
rejection."""

_RETRY_TEMPLATE = """Sua resposta anterior foi avaliada e REPROVADA por um \
revisor. Corrija-a atendendo às críticas abaixo e responda novamente à \
pergunta original, completa e corrigida:

{critique}"""

_PATCH_RETRY_TEMPLATE = """Sua resposta anterior foi avaliada e REPROVADA por \
um revisor. Corrija-a atendendo às críticas abaixo, mas NÃO reescreva a \
resposta inteira: responda SOMENTE com um ou mais blocos de edição, no formato \
EXATO abaixo, aplicados sobre a sua resposta anterior:

<<<<<<< SEARCH
(trecho EXATO da resposta anterior, copiado caractere a caractere)
=======
(o trecho corrigido)
>>>>>>> REPLACE

Regras: o trecho em SEARCH deve existir LITERALMENTE na resposta anterior; use \
um bloco por trecho a corrigir; nenhum texto fora dos blocos.

CRÍTICAS:
{critique}"""

# Bloco de edição do modo patch (#7). DOTALL: search/replace podem ter \n.
_PATCH_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL
)


class ReflectiveAgent(Agent, abstract=True):
    """Declarative evaluator-optimizer loop.

    Required class attributes on concrete subclasses:
        agent:        worker Agent (class or instance) that produces answers.
        judge_llm:    LLM used by the judge (often a cheaper model).
        judge_rubric: objective approval criteria (non-empty).
    Optional:
        max_rounds:     worker attempts before giving up (default 3, >= 1).
        revision_mode:  "full" (default: regenerate the whole answer on a
                        rejected round) or "patch" (#7: the retry emits
                        SEARCH/REPLACE edit blocks applied over the previous
                        answer; non-applying patches fall back to full).

    Cost per round (#4/#5/#6): every retry only APPENDS messages, so the
    prompt prefix is byte-identical across rounds — OpenAI's automatic prompt
    caching applies as-is; for Anthropic workers/judges, opt in with
    ``LLM(..., cache=True)`` (cache_control breakpoints on system + last
    message, giving incremental caching per round). Tool calls repeated with
    identical args across rounds are memoized for the whole run
    (``aixon.toolcache``; opt out per tool via ``as_tool(memoize=False)``).
    On OpenAI workers, the previous attempt is sent as a Predicted Output on
    retries (latency win on unchanged spans; other providers ignore it).
    """

    _suffix = "Agent"

    agent: Any = None
    judge_llm: Any = None
    judge_rubric: str = ""
    max_rounds: int = 3
    # Revision mode (#7): "full" (default) regenerates the whole answer on a
    # rejected round — the historical behavior, byte-identical. "patch" asks
    # the worker for SEARCH/REPLACE edit blocks applied programmatically over
    # the previous answer (output-cost saver for long answers); a patch that
    # doesn't apply falls back to full regeneration for that round.
    revision_mode: str = "full"

    # Reasoning-channel labels ({round}/{max} interpolated on retry).
    judge_label: str = "Avaliando a resposta…"
    retry_label: str = "Refinando a resposta (rodada {round}/{max})…"
    exhausted_label: str = "Rodadas esgotadas — entregando a melhor tentativa."
    patch_fallback_label: str = (
        "Correções pontuais não aplicáveis — regenerando a resposta completa…"
    )

    # ----- validation (runs BEFORE registration; no registry ghosts) -------

    @classmethod
    def _validate_subclass(cls) -> None:
        agent = cls.__dict__.get("agent", None) or getattr(cls, "agent", None)
        is_agent_cls = isinstance(agent, type) and issubclass(agent, Agent)
        if not (is_agent_cls or isinstance(agent, Agent)):
            raise AixonError(
                f"'{cls.__name__}' must declare a class-level 'agent' attribute "
                f"(an Agent class or instance to wrap). Got: {agent!r}."
            )
        judge = cls.__dict__.get("judge_llm", None) or getattr(cls, "judge_llm", None)
        if not isinstance(judge, LLM):
            raise AixonError(
                f"'{cls.__name__}' must declare a class-level 'judge_llm' of type "
                f"LLM (e.g. judge_llm = LLM('gpt-4o-mini')). Got: {judge!r}."
            )
        rubric = getattr(cls, "judge_rubric", "")
        if not isinstance(rubric, str) or not rubric.strip():
            raise AixonError(
                f"'{cls.__name__}' must declare a non-empty 'judge_rubric'. A judge "
                f"without objective criteria degenerates into 'looks fine to me' — "
                f"state what an approved answer must contain."
            )
        rounds = getattr(cls, "max_rounds", 3)
        if not isinstance(rounds, int) or rounds < 1:
            raise AixonError(
                f"'{cls.__name__}' has max_rounds={rounds!r}; it must be an int >= 1."
            )
        mode = getattr(cls, "revision_mode", "full")
        if mode not in ("full", "patch"):
            raise AixonError(
                f"'{cls.__name__}' has revision_mode={mode!r}; "
                f"use 'full' (regenerate) or 'patch' (SEARCH/REPLACE edits)."
            )
        cls._check_composition_cycle()

    # ----- composition-cycle guard (same walk shape as Orchestrator's) -----
    # NOTE: small duplication of Orchestrator._check_composition_cycle; a
    # future refactor may hoist the walk into aixon.agent.

    @classmethod
    def _referenced_agent_classes(cls) -> list[type]:
        agent = cls.__dict__.get("agent", None) or getattr(cls, "agent", None)
        if agent is None:
            return []
        klass = agent if isinstance(agent, type) else type(agent)
        return [klass] if issubclass(klass, Agent) else []

    @classmethod
    def _check_composition_cycle(cls) -> None:
        from aixon.exceptions import CompositionCycleError

        path: list[type] = []

        def walk(node_cls: type) -> None:
            if node_cls in path:
                chain = " -> ".join(c.__name__ for c in path + [node_cls])
                raise CompositionCycleError(
                    f"Composition cycle detected: {chain}. An agent cannot "
                    f"(transitively) include itself as a worker/node/tool."
                )
            path.append(node_cls)
            neighbors = getattr(node_cls, "_referenced_agent_classes", None)
            if callable(neighbors):
                # Call the already-fetched (and callable-checked) `neighbors`,
                # not `node_cls._referenced_agent_classes()` again — `node_cls`
                # is typed as plain `type` here (any class may be a node/tool),
                # so a direct attribute access on it doesn't type-check even
                # though we just proved the attribute exists and is callable.
                for nxt in neighbors():
                    walk(nxt)
            path.pop()

        walk(cls)

    # ----- worker resolution ------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        from aixon.agents.orchestrator import _instantiate

        self._worker: Agent = _instantiate(type(self).agent)

    # ----- judge ------------------------------------------------------------

    def _judge_messages(self, messages: list[Message], answer: Message) -> list[Message]:
        question = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        return [
            Message(role="system",
                    content=_JUDGE_PROMPT.format(rubric=self.judge_rubric)),
            Message(role="user",
                    content=f"PERGUNTA:\n{question}\n\nRESPOSTA:\n{answer.content}"),
        ]

    @staticmethod
    def _approved(verdict: str) -> bool:
        first_line = verdict.strip().splitlines()[0].strip() if verdict.strip() else ""
        return first_line == APPROVE_SENTINEL

    def _retry_messages(self, messages: list[Message], answer: Message,
                        critique: str) -> list[Message]:
        # New list — never mutate the caller's (same precept as _with_prompt).
        return [
            *messages,
            Message(role="assistant", content=answer.content),
            Message(role="user", content=_RETRY_TEMPLATE.format(critique=critique)),
        ]

    # ----- patch revision mode (#7) ----------------------------------------

    def _patch_retry_messages(self, messages: list[Message], answer: Message,
                              critique: str) -> list[Message]:
        return [
            *messages,
            Message(role="assistant", content=answer.content),
            Message(role="user",
                    content=_PATCH_RETRY_TEMPLATE.format(critique=critique)),
        ]

    @staticmethod
    def _parse_patches(text: str) -> list[tuple[str, str]]:
        """SEARCH/REPLACE blocks found in *text* (possibly none)."""
        return [(m.group(1), m.group(2))
                for m in _PATCH_BLOCK_RE.finditer(text or "")]

    @staticmethod
    def _apply_patches(content: str,
                       patches: list[tuple[str, str]]) -> str | None:
        """Apply the blocks over *content*; each SEARCH must occur literally
        (first occurrence replaced). No blocks, or any non-matching SEARCH →
        None, signalling the caller to fall back to full regeneration."""
        if not patches:
            return None
        for search, replace in patches:
            if search not in content:
                return None
            content = content.replace(search, replace, 1)
        return content

    def _apply_patch_reply(self, answer: Message,
                           raw: Message) -> Message | None:
        """Parse the worker's edit blocks and apply them over *answer*.
        None → patch didn't apply (caller falls back to full regeneration;
        the raw call's usage was still spent and must be summed by the caller)."""
        patched = self._apply_patches(answer.content,
                                      self._parse_patches(raw.content))
        if patched is None:
            _log.info(f"reflective '{self.name}': patch did not apply — "
                      f"falling back to full regeneration")
            return None
        return Message(role="assistant", content=patched)

    def _patch_round_sync(
        self, msgs: list[Message], answer: Message, verdict: str,
    ) -> tuple[list[Message], Message | None, Any]:
        """One patch-mode retry attempt (sync worker call). Returns
        (patch_retry_msgs, applied_answer_or_None, raw_usage)."""
        pmsgs = self._patch_retry_messages(msgs, answer, verdict)
        raw = self._worker.invoke(pmsgs)
        return pmsgs, self._apply_patch_reply(answer, raw), raw.usage

    async def _patch_round_async(
        self, msgs: list[Message], answer: Message, verdict: str,
    ) -> tuple[list[Message], Message | None, Any]:
        """Async twin of ``_patch_round_sync``."""
        pmsgs = self._patch_retry_messages(msgs, answer, verdict)
        raw = await self._worker.ainvoke(pmsgs)
        return pmsgs, self._apply_patch_reply(answer, raw), raw.usage

    # ----- sync neutral interface -------------------------------------------

    def invoke(self, messages: list[Message]) -> Message:
        # Tool-call memoization (#5): rounds of this run share one cache, so a
        # retry re-issuing an identical tool call reuses the first result
        # (cost, latency, AND answer/critique consistency). tool_call_cache()
        # reuses an outer (request-scoped) cache when one is already active.
        with tool_call_cache():
            return self._invoke(messages)

    def _invoke(self, messages: list[Message]) -> Message:
        msgs = list(messages)
        answer = self._worker.invoke(msgs)
        # Every worker AND judge turn attempted this run is summed here (a
        # turn that reports no usage contributes zero, never erasing what the
        # others reported) so the final Message.usage covers the WHOLE run,
        # not just the last worker/judge turn.
        total_usage = answer.usage
        for round_ in range(1, self.max_rounds + 1):
            emit_reasoning(self.judge_label)
            verdict_msg = self.judge_llm.complete(self._judge_messages(messages, answer))
            total_usage = merge_usage(total_usage, verdict_msg.usage)
            verdict = verdict_msg.content
            if self._approved(verdict):
                # A COPY carrying the run's total — the worker owns `answer`
                # (it may be cached/shared, or a nested agent's own Message),
                # so the neutral boundary must not mutate it in place.
                return dataclasses.replace(answer, usage=total_usage)
            if round_ == self.max_rounds:
                break
            emit_reasoning(self.retry_label.format(round=round_ + 1,
                                                   max=self.max_rounds))
            # Patch mode (#7): try a SEARCH/REPLACE edit round first; the
            # applied candidate goes straight to the next judgement. A patch
            # that doesn't apply falls through to full regeneration below.
            if self.revision_mode == "patch" and answer.content:
                pmsgs, applied, raw_usage = self._patch_round_sync(
                    msgs, answer, verdict)
                total_usage = merge_usage(total_usage, raw_usage)
                if applied is not None:
                    msgs, answer = pmsgs, applied
                    continue
                emit_reasoning(self.patch_fallback_label)
            msgs = self._retry_messages(msgs, answer, verdict)
            # Predicted Outputs (#6): the retry mostly repeats the previous
            # answer — publish it so OpenAI-backed workers regenerate the
            # unchanged spans by speculative decoding (latency win).
            with prediction_scope(answer.content or None):
                answer = self._worker.invoke(msgs)
            total_usage = merge_usage(total_usage, answer.usage)
        emit_reasoning(self.exhausted_label)
        _log.info(
            f"reflective '{self.name}': rounds exhausted "
            f"(max_rounds={self.max_rounds}); returning last attempt"
        )
        return dataclasses.replace(answer, usage=total_usage)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """LIVE evaluator-optimizer stream.

        The worker is streamed natively each round: its reasoning chunks pass
        through AS THEY ARRIVE (tool labels, nested-agent thoughts), so the
        client sees activity from the first step instead of a mute wait for
        the whole loop. Candidate CONTENT is buffered — an answer must never
        reach the user before the judge approves it — and only the approved
        (or last, on exhaustion) answer is emitted as content."""
        # See invoke(): one tool-call cache for all rounds of this run (#5).
        with tool_call_cache():
            yield from self._stream(messages)

    def _stream(self, messages: list[Message]) -> Iterator[Chunk]:
        msgs = list(messages)
        answer = Message(role="assistant", content="")
        # Patch mode (#7): a successfully applied candidate skips the worker
        # stream on its round — it goes straight to judgement.
        pending: Message | None = None
        for round_ in range(1, self.max_rounds + 1):
            if pending is not None:
                answer, pending = pending, None
            else:
                parts: list[str] = []
                # See _invoke(): round 1 has no previous answer (no-op); retry
                # rounds publish it as the predicted output (#6).
                with prediction_scope(answer.content or None):
                    for chunk in self._worker.stream(msgs):
                        if chunk.reasoning:
                            yield Chunk(reasoning=chunk.reasoning)
                        if chunk.content:
                            parts.append(chunk.content)
                answer = Message(role="assistant", content="".join(parts))
            yield Chunk(reasoning=self.judge_label + "\n")
            verdict = self.judge_llm.complete(
                self._judge_messages(messages, answer)
            ).content
            if self._approved(verdict):
                yield Chunk(content=answer.content)
                yield Chunk(done=True)
                return
            if round_ == self.max_rounds:
                break
            yield Chunk(reasoning=self.retry_label.format(round=round_ + 1,
                                                          max=self.max_rounds) + "\n")
            # See _invoke(): patch mode first (#7). The worker's patch reply is
            # produced via invoke (short output; its raw text must NEVER leak
            # as stream content) and the applied candidate is judged next round.
            if self.revision_mode == "patch" and answer.content:
                pmsgs, applied, _ = self._patch_round_sync(msgs, answer, verdict)
                if applied is not None:
                    msgs, pending = pmsgs, applied
                    continue
                yield Chunk(reasoning=self.patch_fallback_label + "\n")
            msgs = self._retry_messages(msgs, answer, verdict)
        yield Chunk(reasoning=self.exhausted_label + "\n")
        _log.info(
            f"reflective '{self.name}': rounds exhausted "
            f"(max_rounds={self.max_rounds}); returning last attempt"
        )
        yield Chunk(content=answer.content)
        yield Chunk(done=True)

    # ----- async neutral interface (native — no thread bridge) ----------------

    async def ainvoke(self, messages: list[Message]) -> Message:
        # See invoke(): one tool-call cache for all rounds of this run (#5).
        with tool_call_cache():
            return await self._ainvoke(messages)

    async def _ainvoke(self, messages: list[Message]) -> Message:
        msgs = list(messages)
        answer = await self._worker.ainvoke(msgs)
        # See invoke(): sum every worker + judge turn attempted this run.
        total_usage = answer.usage
        for round_ in range(1, self.max_rounds + 1):
            emit_reasoning(self.judge_label)
            verdict_msg = await self.judge_llm.acomplete(
                self._judge_messages(messages, answer)
            )
            total_usage = merge_usage(total_usage, verdict_msg.usage)
            verdict = verdict_msg.content
            if self._approved(verdict):
                # See invoke(): a COPY — never mutate the worker's Message.
                return dataclasses.replace(answer, usage=total_usage)
            if round_ == self.max_rounds:
                break
            emit_reasoning(self.retry_label.format(round=round_ + 1,
                                                   max=self.max_rounds))
            # See _invoke(): patch mode first (#7), full regeneration fallback.
            if self.revision_mode == "patch" and answer.content:
                pmsgs, applied, raw_usage = await self._patch_round_async(
                    msgs, answer, verdict)
                total_usage = merge_usage(total_usage, raw_usage)
                if applied is not None:
                    msgs, answer = pmsgs, applied
                    continue
                emit_reasoning(self.patch_fallback_label)
            msgs = self._retry_messages(msgs, answer, verdict)
            # See _invoke(): predicted output for the retry (#6).
            with prediction_scope(answer.content or None):
                answer = await self._worker.ainvoke(msgs)
            total_usage = merge_usage(total_usage, answer.usage)
        emit_reasoning(self.exhausted_label)
        _log.info(
            f"reflective '{self.name}': rounds exhausted "
            f"(max_rounds={self.max_rounds}); returning last attempt"
        )
        return dataclasses.replace(answer, usage=total_usage)

    async def astream(self, messages: list[Message]) -> "AsyncIterator[Chunk]":
        """Async mirror of ``stream`` — same live pass-through/buffering."""
        # See invoke(): one tool-call cache for all rounds of this run (#5).
        with tool_call_cache():
            async for chunk in self._astream(messages):
                yield chunk

    async def _astream(self, messages: list[Message]) -> "AsyncIterator[Chunk]":
        msgs = list(messages)
        answer = Message(role="assistant", content="")
        # See _stream(): applied patch candidates skip the worker stream (#7).
        pending: Message | None = None
        for round_ in range(1, self.max_rounds + 1):
            if pending is not None:
                answer, pending = pending, None
            else:
                parts: list[str] = []
                # See _invoke(): predicted output for retry rounds (#6).
                with prediction_scope(answer.content or None):
                    async for chunk in self._worker.astream(msgs):
                        if chunk.reasoning:
                            yield Chunk(reasoning=chunk.reasoning)
                        if chunk.content:
                            parts.append(chunk.content)
                answer = Message(role="assistant", content="".join(parts))
            yield Chunk(reasoning=self.judge_label + "\n")
            verdict = (
                await self.judge_llm.acomplete(self._judge_messages(messages, answer))
            ).content
            if self._approved(verdict):
                yield Chunk(content=answer.content)
                yield Chunk(done=True)
                return
            if round_ == self.max_rounds:
                break
            yield Chunk(reasoning=self.retry_label.format(round=round_ + 1,
                                                          max=self.max_rounds) + "\n")
            # See _stream(): patch mode first (#7), full regeneration fallback.
            if self.revision_mode == "patch" and answer.content:
                pmsgs, applied, _ = await self._patch_round_async(
                    msgs, answer, verdict)
                if applied is not None:
                    msgs, pending = pmsgs, applied
                    continue
                yield Chunk(reasoning=self.patch_fallback_label + "\n")
            msgs = self._retry_messages(msgs, answer, verdict)
        yield Chunk(reasoning=self.exhausted_label + "\n")
        _log.info(
            f"reflective '{self.name}': rounds exhausted "
            f"(max_rounds={self.max_rounds}); returning last attempt"
        )
        yield Chunk(content=answer.content)
        yield Chunk(done=True)
