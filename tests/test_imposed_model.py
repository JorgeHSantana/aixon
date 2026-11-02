# tests/test_imposed_model.py
"""0.1.28 — the imposed model on `LLM`.

Some deployments send every LLM call through a gateway that, not the agent's
class, picks the model. `imposed_model_scope(model, provider)` publishes that
pick; while it is active an `LLM` builds the imposed model whatever it was
declared with, and never falls back to its configured one: a fallback would
run the call on the wrong model and nothing would say so. Everything here is
offline: the imposed model is `tests/_fakes.RecordingProvider`, registered
as "rec" by the `rec` fixture."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage

from aixon import llm as llm_module
from aixon.exceptions import AixonError, ImposedModelViolation
from aixon.llm import LLM
from aixon.message import Message
from aixon.runtime import (
    PORTABLE_PARAMS,
    current_imposed_by,
    current_imposed_model,
    generation_params,
    imposed_model_scope,
    prediction_scope,
)
from tests._fakes import (
    FakeChatModel,
    FakeProvider,
    RecordingChatModel,
    make_recording_provider,
)


# ── request_chat_model under the scope ───────────────────────────────────────

def test_imposed_model_even_without_params(rec):
    llm = LLM("fake-1", provider="fake")

    with imposed_model_scope("tier-x", "rec"):
        assert llm.request_chat_model() is rec.model

    assert rec.builds == [("tier-x", {})]


def test_only_portable_params_reach_imposed_build(rec):
    # The real gateway provider supports the `reasoning` knob: a build through
    # `LLM._build` would hand it `reasoning` and the request's `reasoning_effort`.
    rec.supports_reasoning = True
    llm = LLM(
        "fake-1", provider="fake", temperature=0.2, reasoning=True, timeout=5, seed=7,
        presence_penalty=None,
    )
    request = {"max_tokens": 64, "top_p": 0.9, "reasoning_effort": "high", "stop": ["FIM"]}

    with imposed_model_scope("tier-x", "rec"), generation_params(request):
        llm.request_chat_model()

    assert rec.builds[-1] == (
        "tier-x",
        {"temperature": 0.2, "max_tokens": 64, "top_p": 0.9, "stop": ["FIM"]},
    )

    # the request wins over the class, as it always did
    with imposed_model_scope("tier-x", "rec"), generation_params({"temperature": 0.7}):
        llm.request_chat_model()
    assert rec.builds[-1][1]["temperature"] == 0.7


def test_class_param_aliases_are_normalized_before_the_portable_filter(rec):
    # `generation_params()` turns a request's dialect spellings into the
    # canonical keys; the class's params get the same treatment under the scope,
    # so a portable knob spelled the other way does not vanish.
    llm = LLM("fake-1", provider="fake", max_completion_tokens=64, stop_sequences=["FIM"])

    with imposed_model_scope("tier-x", "rec"):
        llm.request_chat_model()
    assert rec.builds[-1] == ("tier-x", {"max_tokens": 64, "stop": ["FIM"]})

    # the canonical key wins over its alias, and the request over the class
    both = LLM("fake-1", provider="fake", max_tokens=32, max_completion_tokens=64)
    with imposed_model_scope("tier-x", "rec"):
        both.request_chat_model()
    assert rec.builds[-1] == ("tier-x", {"max_tokens": 32})
    with imposed_model_scope("tier-x", "rec"), generation_params({"max_tokens": 8}):
        both.request_chat_model()
    assert rec.builds[-1] == ("tier-x", {"max_tokens": 8})


def test_portable_params_are_exactly_the_contract():
    assert PORTABLE_PARAMS == frozenset(
        {"temperature", "top_p", "max_tokens", "stop", "presence_penalty", "frequency_penalty"}
    )


def test_same_llm_inside_and_outside_scope_never_crosses(rec):
    llm = LLM("fake-1", provider="fake")

    plain = llm.request_chat_model()
    with imposed_model_scope("tier-x", "rec"):
        assert llm.request_chat_model() is rec.model
    with generation_params({"temperature": 0.3}):
        plain_params = llm.request_chat_model()
    with imposed_model_scope("tier-x", "rec"), generation_params({"temperature": 0.3}):
        assert llm.request_chat_model() is rec.model

    assert plain is llm.chat_model
    assert llm.request_chat_model() is plain
    assert plain_params is not rec.model
    assert isinstance(plain_params, FakeChatModel)


def test_llm_entry_points_run_on_the_imposed_model(rec):
    rec.model.script = [AIMessage(content=f"r{n}") for n in range(1, 5)]
    llm = LLM("fake-1", provider="fake")
    ask = [Message(role="user", content="oi")]

    async def run_async() -> list[str]:
        answer = await llm.acomplete(ask)
        streamed = "".join([chunk.content async for chunk in llm.astream(ask)])
        return [answer.content, streamed]

    with imposed_model_scope("tier-x", "rec"):
        answers = [
            llm.complete(ask).content,
            "".join(chunk.content for chunk in llm.stream(ask)),
            *asyncio.run(run_async()),
        ]

    assert answers == ["r1", "r2", "r3", "r4"]
    assert [[m.content for m in call] for call in rec.model.calls] == [["oi"]] * 4
    assert llm._chat_model is None  # the configured model was not even built


# ── the scope ────────────────────────────────────────────────────────────────

def test_scope_sets_nests_and_restores():
    assert current_imposed_model() is None
    with imposed_model_scope("tier-x", "rec") as outer:
        assert outer == ("tier-x", "rec")
        assert current_imposed_model() == ("tier-x", "rec")
        with imposed_model_scope("tier-y", "rec"):
            assert current_imposed_model() == ("tier-y", "rec")
        assert current_imposed_model() == ("tier-x", "rec")
    assert current_imposed_model() is None


def test_scope_exit_in_another_context_does_not_raise():
    # Same hazard as reasoning_channel() (#21): an abandoned async generator is
    # finalized in a different Context, where ContextVar.reset() raises
    # ValueError. Run in a Context of its own: the failed reset leaves the
    # value stale there, which must not leak into the other tests.
    def body() -> None:
        scope = imposed_model_scope("tier-x", "rec")
        scope.__enter__()
        contextvars.copy_context().run(scope.__exit__, None, None, None)  # no ValueError

        with imposed_model_scope("tier-y", "rec"):  # the var is still usable
            assert current_imposed_model() == ("tier-y", "rec")

    contextvars.copy_context().run(body)


# ── fail closed ──────────────────────────────────────────────────────────────

def test_chat_model_fails_closed(rec):
    llm = LLM("fake-1", provider="fake")

    with imposed_model_scope("tier-x", "rec"):
        with pytest.raises(ImposedModelViolation, match="tier-x") as caught:
            llm.chat_model

    # it names the configured model and the imposed pair, and points the way out
    for part in ("fake-1", "'tier-x'", "provider 'rec'", "request_chat_model()", "pinned=True"):
        assert part in caught.value.message
    assert llm._chat_model is None  # nothing was built behind the error
    assert rec.builds == []
    assert isinstance(llm.chat_model, FakeChatModel)  # outside the scope: as ever


def test_chat_model_built_before_the_scope_still_fails_closed(rec):
    llm = LLM("fake-1", provider="fake")
    configured = llm.chat_model  # built and cached outside any scope
    assert isinstance(configured, FakeChatModel)

    with imposed_model_scope("tier-x", "rec"):
        with pytest.raises(ImposedModelViolation, match="tier-x"):
            llm.chat_model  # the cache must not slip past the check

    assert llm.chat_model is configured  # outside again: the same model as ever
    assert rec.builds == []


def test_unregistered_imposed_provider_fails_closed():
    llm = LLM("fake-1", provider="fake")

    with imposed_model_scope("tier-x", "no-such-provider"):
        with pytest.raises(AixonError, match="no-such-provider"):
            llm.request_chat_model()

    assert llm._chat_model is None
    assert llm._request_model_cache == {}


# ── pinned ───────────────────────────────────────────────────────────────────

def test_pinned_llm_keeps_configured_model(rec):
    llm = LLM("fake-1", provider="fake", pinned=True)

    with imposed_model_scope("tier-x", "rec"):
        model = llm.request_chat_model()
        assert model is llm.chat_model  # and no ImposedModelViolation
        with generation_params({"temperature": 0.3}):
            with_params = llm.request_chat_model()

    for kept in (model, with_params):  # the configured model, never the imposed one
        assert isinstance(kept, FakeChatModel) and not isinstance(kept, RecordingChatModel)
    assert rec.builds == []


def test_pinned_is_off_by_default_and_not_a_vendor_param():
    assert LLM("fake-1", provider="fake").pinned is False

    llm = LLM("fake-1", provider="fake", pinned=True, temperature=0.2)
    assert llm.pinned is True
    assert llm.params == {"temperature": 0.2}  # `pinned` never reaches the provider


# ── the request-model cache ──────────────────────────────────────────────────

def test_cache_key_includes_model_and_provider(rec):
    rec2 = make_recording_provider("rec2")
    llm = LLM("fake-1", provider="fake")

    for model, provider in (
        ("tier-a", "rec"), ("tier-b", "rec"), ("tier-a", "rec2"), ("tier-a", "rec"),
    ):
        with imposed_model_scope(model, provider):
            llm.request_chat_model()

    assert [model for model, _ in rec.builds] == ["tier-a", "tier-b"]
    assert [model for model, _ in rec2.builds] == ["tier-a"]


def test_imposed_pair_equal_to_the_configured_one_still_gets_its_own_build(monkeypatch):
    # The imposed (model, provider) can be the very pair the LLM is declared
    # with. The two models still differ: the class kwargs (timeout, ...) never
    # reach the imposed build, so the cache must tell them apart.
    built: list[dict] = []
    original = FakeProvider.build

    def build(self, model, **params):
        built.append(params)
        return original(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", build)
    llm = LLM("fake-1", provider="fake", timeout=5)

    with generation_params({"top_p": 0.9}):
        configured = llm.request_chat_model()
        with imposed_model_scope("fake-1", "fake"):
            imposed = llm.request_chat_model()

    assert imposed is not configured
    assert built == [{"timeout": 5, "top_p": 0.9}, {"top_p": 0.9}]


def test_request_cache_keeps_8_entries_and_evicts_the_oldest(rec):
    llm = LLM("fake-1", provider="fake")

    def ask(i: int) -> None:
        with imposed_model_scope("tier-x", "rec"), generation_params({"temperature": i / 10}):
            llm.request_chat_model()

    for i in range(9):  # 9 distinct param sets: the first one is pushed out
        ask(i)
    assert len(llm._request_model_cache) == 8
    assert len(rec.builds) == 9

    ask(8)  # the newest and ...
    ask(1)  # ... the second oldest are still there
    assert len(rec.builds) == 9
    ask(0)  # the oldest is gone: rebuilt
    assert len(rec.builds) == 10


def test_racing_builds_of_one_key_keep_the_first_stored_model(monkeypatch):
    llm = LLM("fake-1", provider="fake")
    original = FakeProvider.build
    racing = {"started": False}
    rival: list = []

    def build(self, model, **params):
        # the build runs outside the lock ...
        assert not llm_module._REQUEST_MODEL_CACHE_LOCK.locked()
        if not racing["started"]:
            # ... so a rival (another thread, in real life) can build and store
            # the same key while this build is still running
            racing["started"] = True
            rival.append(llm.request_chat_model())
        return original(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", build)

    with generation_params({"temperature": 0.3}):
        ours = llm.request_chat_model()

    assert ours is rival[0]  # the model stored first wins; the one just built is dropped
    assert list(llm._request_model_cache.values()) == [ours]


def _yielding(method):
    def wrapper(self, *args):
        time.sleep(0)  # let another thread run before the operation
        return method(self, *args)

    return wrapper


class _YieldingDict(dict):
    """A dict that gives other threads a turn before every operation, so any
    check-then-act sequence on it that is not atomic gets interleaved. Without
    it the unguarded cache survives the test below: its evict-then-insert
    window is too short to be hit by chance. With it, the code from before the
    lock fails on every run."""

    get = _yielding(dict.get)
    pop = _yielding(dict.pop)
    __contains__ = _yielding(dict.__contains__)
    __getitem__ = _yielding(dict.__getitem__)
    __setitem__ = _yielding(dict.__setitem__)
    __delitem__ = _yielding(dict.__delitem__)
    __len__ = _yielding(dict.__len__)
    __iter__ = _yielding(dict.__iter__)


def test_request_cache_is_thread_safe(monkeypatch):
    original = FakeProvider.build

    def slow_build(self, model, **params):
        time.sleep(0.001)  # keeps many builds in flight at once
        return original(self, model, **params)

    monkeypatch.setattr(FakeProvider, "build", slow_build)
    llm = LLM("fake-1", provider="fake")
    llm._request_model_cache = _YieldingDict()

    def call(i: int):
        with generation_params({"temperature": (i % 24) / 10}):
            return llm.request_chat_model()

    with ThreadPoolExecutor(16) as pool:
        futures = [pool.submit(call, i) for i in range(400)]
        for future in futures:
            future.result()  # re-raises whatever the worker raised

    assert len(llm._request_model_cache) <= 8


# ── prediction and prompt cache follow the provider that really answers ──────

def _cache_marked(lc_message) -> bool:
    content = lc_message.content
    return isinstance(content, list) and any(
        isinstance(block, dict) and "cache_control" in block for block in content
    )


def test_prediction_and_prompt_cache_follow_effective_provider(rec):
    # prediction: only the LLM's own provider may use it, never an imposed model
    llm = LLM("gpt-5.4", provider="openai")
    pinned = LLM("gpt-5.4", provider="openai", pinned=True)
    attached = {"prediction": {"type": "content", "content": "anterior"}}

    with prediction_scope("anterior"):
        assert llm._invoke_kwargs() == attached
        with imposed_model_scope("tier-x", "rec"):
            assert llm._invoke_kwargs() == {}
            assert pinned._invoke_kwargs() == attached
        with imposed_model_scope("gpt-5.4", "openai"):  # openai does support it ...
            assert llm._invoke_kwargs() == {}  # ... but the contract forbids it here

    # prompt cache: the breakpoints follow the provider that answers
    system = Message(role="system", content="Você é um assistente.")
    user = Message(role="user", content="oi")
    llm = LLM("claude-sonnet-5", provider="anthropic", cache=True)
    pinned = LLM("claude-sonnet-5", provider="anthropic", cache=True, pinned=True)

    wire = llm._to_wire([system, user])
    assert [_cache_marked(m) for m in wire] == [True, True]  # anthropic, no scope
    with imposed_model_scope("tier-x", "rec"):
        wire = llm._to_wire([system, user])
        assert all(isinstance(m.content, str) for m in wire)  # rec has no cache_control
        wire = pinned._to_wire([system, user])
        assert [_cache_marked(m) for m in wire] == [True, True]  # pinned keeps its own


# ── 0.1.29: aviso de kwargs descartados quando o SERVIDOR impõe ──────────────

@pytest.fixture
def sem_avisos_previos(monkeypatch):
    monkeypatch.setattr(LLM, "_warned_imposed", set())


def _avisos(caplog):
    return [r for r in caplog.records
            if r.name == "aixon.llm" and "sob modelo imposto" in r.getMessage()]


def test_imposed_by_padrao_agent_e_restaura():
    assert current_imposed_by() is None
    with imposed_model_scope("tier-x", "rec"):
        assert current_imposed_by() == "agent"
        with imposed_model_scope("tier-y", "rec", by="server"):
            assert current_imposed_by() == "server"
        assert current_imposed_by() == "agent"
    assert current_imposed_by() is None


def test_aviso_uma_vez_por_chave(rec, caplog, sem_avisos_previos):
    llm = LLM("fake-1", provider="fake", seed=7)
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        with imposed_model_scope("tier-x", "rec", by="server"):
            llm.request_chat_model()
        with imposed_model_scope("tier-x", "rec", by="server"):
            llm.request_chat_model()
    (record,) = _avisos(caplog)
    assert record.levelno == logging.WARNING


def test_aviso_por_conjunto_de_kwargs_perdidos(rec, caplog, sem_avisos_previos):
    a = LLM("x", provider="fake", seed=7)
    b = LLM("x", provider="fake", timeout=300)
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        with imposed_model_scope("tier-x", "rec", by="server"):
            a.request_chat_model()
            b.request_chat_model()
    first, second = _avisos(caplog)
    assert "seed" in first.getMessage() and "timeout" not in first.getMessage()
    assert "timeout" in second.getMessage() and "seed" not in second.getMessage()


def test_sem_aviso_quando_a_classe_impoe(rec, caplog, sem_avisos_previos):
    from aixon import LLMAgent

    class ImpostoraAgent(LLMAgent):
        llm = LLM("tier-x", provider="rec", seed=7, timeout=300)
        impose_model = True

    rec.model.script = [AIMessage(content="ok")]
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        assert ImpostoraAgent().invoke([Message(role="user", content="oi")]).content == "ok"
    assert rec.builds  # passou mesmo pelo modelo imposto
    assert _avisos(caplog) == []


def test_impositor_nao_avisa_de_novo_sob_servidor(rec, caplog, sem_avisos_previos):
    # A classe impositora já avisou na definição; sob o servidor, o mesmo LLM
    # não repete o aviso.
    from aixon import LLMAgent

    with caplog.at_level(logging.WARNING, logger="aixon.agent"):
        class ImpAgent(LLMAgent):
            llm = LLM("fake-1", provider="fake", seed=7)
            impose_model = True
    assert [r for r in caplog.records if "ImpAgent" in r.getMessage()]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        with imposed_model_scope("tier-x", "rec", by="server"):
            ImpAgent.llm.request_chat_model()
    assert _avisos(caplog) == []


def test_llm_nao_impositor_ainda_avisa(rec, caplog, sem_avisos_previos):
    llm = LLM("fake-1", provider="fake", seed=7)
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        with imposed_model_scope("tier-x", "rec", by="server"):
            llm.request_chat_model()
    assert len(_avisos(caplog)) == 1


def test_sem_aviso_sem_kwargs_nao_portaveis(rec, caplog, sem_avisos_previos):
    llm = LLM("fake-1", provider="fake", temperature=0.2, max_tokens=64)
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        with imposed_model_scope("tier-x", "rec", by="server"):
            llm.request_chat_model()
        pinned = LLM("fake-1", provider="fake", seed=7, pinned=True)
        with imposed_model_scope("tier-x", "rec", by="server"):
            pinned.request_chat_model()
    assert _avisos(caplog) == []


def test_aviso_nomeia_as_chaves(rec, caplog, sem_avisos_previos):
    llm = LLM("fake-1", provider="fake", timeout=5, reasoning=True, temperature=0.2)
    with caplog.at_level(logging.WARNING, logger="aixon.llm"):
        with imposed_model_scope("tier-x", "rec", by="server"):
            llm.request_chat_model()
    (record,) = _avisos(caplog)
    message = record.getMessage()
    assert message.startswith("LLM('fake-1') sob modelo imposto 'tier-x' (rec): "
                              "ignorando kwargs não portáveis ")
    assert message.index("'reasoning'") < message.index("'timeout'")
    assert "temperature" not in message
