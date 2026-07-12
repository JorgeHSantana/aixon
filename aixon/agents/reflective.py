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

from typing import Any, AsyncIterator, Iterator

from aixon.agent import Agent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.logging import Logger
from aixon.message import Chunk, Message
from aixon.reasoning import emit_reasoning

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


class ReflectiveAgent(Agent, abstract=True):
    """Declarative evaluator-optimizer loop.

    Required class attributes on concrete subclasses:
        agent:        worker Agent (class or instance) that produces answers.
        judge_llm:    LLM used by the judge (often a cheaper model).
        judge_rubric: objective approval criteria (non-empty).
    Optional:
        max_rounds:   worker attempts before giving up (default 3, >= 1).
    """

    _suffix = "Agent"

    agent: Any = None
    judge_llm: Any = None
    judge_rubric: str = ""
    max_rounds: int = 3

    # Reasoning-channel labels ({round}/{max} interpolated on retry).
    judge_label: str = "Avaliando a resposta…"
    retry_label: str = "Refinando a resposta (rodada {round}/{max})…"
    exhausted_label: str = "Rodadas esgotadas — entregando a melhor tentativa."

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

    # ----- sync neutral interface -------------------------------------------

    def invoke(self, messages: list[Message]) -> Message:
        msgs = list(messages)
        answer = self._worker.invoke(msgs)
        for round_ in range(1, self.max_rounds + 1):
            emit_reasoning(self.judge_label)
            verdict = self.judge_llm.complete(
                self._judge_messages(messages, answer)
            ).content
            if self._approved(verdict):
                return answer
            if round_ == self.max_rounds:
                break
            emit_reasoning(self.retry_label.format(round=round_ + 1,
                                                   max=self.max_rounds))
            msgs = self._retry_messages(msgs, answer, verdict)
            answer = self._worker.invoke(msgs)
        emit_reasoning(self.exhausted_label)
        _log.info(
            f"reflective '{self.name}': rounds exhausted "
            f"(max_rounds={self.max_rounds}); returning last attempt"
        )
        return answer

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """LIVE evaluator-optimizer stream.

        The worker is streamed natively each round: its reasoning chunks pass
        through AS THEY ARRIVE (tool labels, nested-agent thoughts), so the
        client sees activity from the first step instead of a mute wait for
        the whole loop. Candidate CONTENT is buffered — an answer must never
        reach the user before the judge approves it — and only the approved
        (or last, on exhaustion) answer is emitted as content."""
        msgs = list(messages)
        answer = Message(role="assistant", content="")
        for round_ in range(1, self.max_rounds + 1):
            parts: list[str] = []
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
        msgs = list(messages)
        answer = await self._worker.ainvoke(msgs)
        for round_ in range(1, self.max_rounds + 1):
            emit_reasoning(self.judge_label)
            verdict = (
                await self.judge_llm.acomplete(self._judge_messages(messages, answer))
            ).content
            if self._approved(verdict):
                return answer
            if round_ == self.max_rounds:
                break
            emit_reasoning(self.retry_label.format(round=round_ + 1,
                                                   max=self.max_rounds))
            msgs = self._retry_messages(msgs, answer, verdict)
            answer = await self._worker.ainvoke(msgs)
        emit_reasoning(self.exhausted_label)
        _log.info(
            f"reflective '{self.name}': rounds exhausted "
            f"(max_rounds={self.max_rounds}); returning last attempt"
        )
        return answer

    async def astream(self, messages: list[Message]) -> "AsyncIterator[Chunk]":
        """Async mirror of ``stream`` — same live pass-through/buffering."""
        msgs = list(messages)
        answer = Message(role="assistant", content="")
        for round_ in range(1, self.max_rounds + 1):
            parts: list[str] = []
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
            msgs = self._retry_messages(msgs, answer, verdict)
        yield Chunk(reasoning=self.exhausted_label + "\n")
        _log.info(
            f"reflective '{self.name}': rounds exhausted "
            f"(max_rounds={self.max_rounds}); returning last attempt"
        )
        yield Chunk(content=answer.content)
        yield Chunk(done=True)
