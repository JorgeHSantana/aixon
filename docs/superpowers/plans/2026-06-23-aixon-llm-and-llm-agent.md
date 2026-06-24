# aixon LLM + Providers + LLMAgent Implementation Plan (langgraph-native, LangChain 1.x)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `Provider` ABC + registry, three concrete providers (OpenAI / Anthropic / Google) with lazy SDK imports, the `LLM` declarative handle (lazy `chat_model`, neutral `complete`/`stream`), LangChain message-conversion helpers (`_langchain.py`), the `LLMAgent` abstract subtype, and the **single shared test-fakes module** `tests/_fakes.py` — so consumer projects can declare a chat agent with one class attribute and get a working, registered agent, and so Plans 3/4/5/7 have a hermetic offline LLM to test against.

**Architecture:** Providers build LangChain `BaseChatModel` instances on demand; the `LLM` class wraps the resulting model behind a neutral `Message`/`Chunk` boundary. `LLMAgent` inherits from `Agent(abstract=True)`, is **pure-LLM** (no tool-calling loop), implements `invoke`/`stream` by delegating to `self.llm`, and validates at subclass-definition time that a concrete subclass declared the required `llm` attribute. All provider SDK imports are inside `build()` so importing `aixon` never pulls in every vendor SDK. The hermetic `FakeChatModel` lives in `tests/_fakes.py` (owned by this plan; imported, never redefined, by Plans 3/4/5/7).

**Tech Stack (LangChain 1.x — validated 1.3 / core 1.4 / langgraph 1.2):** Python 3.11+, `langchain>=1.0`, `langchain-core>=1.0`, `langgraph>=1.0` (extra `llm`), `langchain-openai>=0.2` (extra `openai`), `langchain-anthropic>=0.2` (extra `anthropic`), `langchain-google-genai>=2.0` (extra `google`), `pytest`. **No `<1` ceiling anywhere.**

## Global Constraints

These values are copied verbatim from the interface contract (§0, §1, §9.5) and are **binding**.

- `requires-python >= "3.11"` — from contract §0.
- Build backend: `hatchling`. Package name `aixon`; import name `aixon`. Core `dependencies = []` (each plan adds only to its extras) — from §0.
- **Neutral boundary (§0):** `Agent.invoke`/`stream` and all public API speak ONLY `Message`/`Chunk`. LangChain/LangGraph/provider objects may be used INTERNALLY but must be converted at the boundary. Conversion helpers live in `aixon/_langchain.py` (§1.4).
- **Lazy provider imports (§1.2):** vendor SDK imports (`langchain_openai`, `langchain_anthropic`, `langchain_google_genai`) live **inside** the `build()` method of each concrete provider. Importing `aixon` — or any provider module — must never raise `ImportError` because a vendor SDK is absent.
- **Extras (§1.7 + §9.2, authoritative):**
  - `llm = ["langchain>=1.0", "langchain-core>=1.0", "langgraph>=1.0"]` (langgraph lives in `llm`; ToolAgent and Orchestrator both need it — there is NO separate `orchestration` extra).
  - `openai = ["langchain-openai>=0.2"]`
  - `anthropic = ["langchain-anthropic>=0.2"]`
  - `google = ["langchain-google-genai>=2.0"]`
  - Add all of the above to `all` (union-merge with the existing `server`/`cli` entries — never drop them).
  - **No `<1` pin anywhere.** The old plan's `langchain>=0.3` / `langchain-core>=0.3` are removed.
- **langgraph-native (§1.7):** the old 0.x `create_tool_calling_agent` + `AgentExecutor` are gone. The ToolAgent (Plan 3) will use `from langchain.agents import create_agent`. `langgraph.prebuilt.create_react_agent` is DEPRECATED in langgraph 1.0 — do not use it. This plan (Plan 2) does NOT build a ToolAgent, but its `FakeChatModel` MUST be drivable by `create_agent` (it is — validated below) because Plan 3 imports that exact class.
- **Test fakes — single owner (§9.1):** `tests/_fakes.py` is created by THIS plan and is the ONE place hermetic doubles live. It MUST export `register_fake_provider()` (idempotent), `FAKE_MODEL="fake-1"`, `FAKE_PROVIDER="fake"`, the EXACT `FakeChatModel` class from §9.1 (copied verbatim — validated against langchain 1.3 / core 1.4 / langgraph 1.2), `make_llm(**params)`, and `make_echo_agent(name="echo", *, hidden=False)`. Plans 3/4/5/7 import from it; they do NOT redefine it.
- **Hermetic tests (§1.6):** no real API keys, no network. All LLM/provider tests use `LLM("fake-1", provider="fake")`. Vendor-specific `build()` tests skip via `pytest.importorskip`.
- **Dedicated virtualenv (§9.5 — REQUIRED):** all install/run steps use the project-local `.venv`, created ONCE in Task 0. Every test run uses `.venv/bin/python -m pytest …` — NEVER a bare `pytest` (the console script can carry a stale shebang), NEVER another project's interpreter. `.venv` is already git-ignored.
- **No `tests/__init__.py`** — rootdir import mode, matches Plan 1.
- Error tone: state what was got and how to fix it.
- Commits carry the `Co-Authored-By` trailer per repo convention.
- **This plan does NOT re-implement** anything from Plan 1 (`Agent`, `AgentTool`, `Message`, `Chunk`, `Role`, `Registry`, `Logger`, `autodiscover`, `exceptions`).

---

### Task 0: Dedicated venv + LangChain 1.x extras (environment setup)

**Files:**
- Modify: `pyproject.toml` — add `llm`, `openai`, `anthropic`, `google` extras (LangChain 1.x) and union them into `all`.

**Interfaces:**
- Consumes: `pyproject.toml` as it exists after Plan 1 (core `dependencies = []`; extras `dev`, `server`, `cli`, `all`).
- Produces: a project-local `.venv` with aixon + LangChain 1.x installed editable; updated `pyproject.toml`.

> **Why this is Task 0:** Every later step runs `.venv/bin/python -m pytest`. The venv must exist and have LangChain 1.x BEFORE any test that imports `langchain_core`. There is currently NO `tests/test_providers.py` in the repo, so nothing is broken to clean up — the old plan's "stale test cleanup" step is dropped.

- [ ] **Step 1: Update `pyproject.toml` extras**

Open `/Users/jorge/Documents/Git/aixon/pyproject.toml` and replace the entire `[project.optional-dependencies]` block with exactly:

```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]
server = ["fastapi>=0.100", "uvicorn[standard]>=0.20", "pydantic>=2.0"]
cli = ["click>=8.0"]
llm = ["langchain>=1.0", "langchain-core>=1.0", "langgraph>=1.0"]
openai = ["langchain-openai>=0.2"]
anthropic = ["langchain-anthropic>=0.2"]
google = ["langchain-google-genai>=2.0"]
all = [
    "fastapi>=0.100",
    "uvicorn[standard]>=0.20",
    "pydantic>=2.0",
    "click>=8.0",
    "langchain>=1.0",
    "langchain-core>=1.0",
    "langgraph>=1.0",
    "langchain-openai>=0.2",
    "langchain-anthropic>=0.2",
    "langchain-google-genai>=2.0",
]
```

> Do NOT touch `dependencies = []` (core stays dependency-free per §0). Do NOT add a `<1` ceiling. `retrieval` is Plan 6's extra and is NOT added here.

- [ ] **Step 2: Create the dedicated venv (once)**

```bash
cd /Users/jorge/Documents/Git/aixon && python3 -m venv .venv
```

Expected: creates `.venv/` (already git-ignored). If `.venv` already exists from a prior run, this is a no-op-safe re-create; you may skip it.

- [ ] **Step 3: Install aixon editable with the LLM + vendor extras into `.venv`**

```bash
.venv/bin/python -m pip install -e ".[dev,llm,openai,anthropic,google]"
```

Expected: `Successfully installed ...` including `langchain-1.x`, `langchain-core-1.x`, `langgraph-1.x`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`. (A pip-upgrade notice on stderr is harmless.)

- [ ] **Step 4: Confirm the installed versions are LangChain 1.x**

```bash
.venv/bin/python -c "import importlib.metadata as m; print('langchain', m.version('langchain')); print('langchain-core', m.version('langchain-core')); print('langgraph', m.version('langgraph'))"
```

Expected (or newer 1.x):
```
langchain 1.3.11
langchain-core 1.4.8
langgraph 1.2.6
```

- [ ] **Step 5: Confirm the Plan 1 suite still passes in the venv**

```bash
.venv/bin/python -m pytest -v
```

Expected: all existing Plan 1 tests PASS (test_agent, test_as_tool, test_discovery, test_exceptions, test_logging, test_message, test_registry).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
chore: add langgraph-native llm extra (langchain/core/langgraph >=1.0) + vendor extras

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1: `Provider` ABC + provider registry + model-name inference

**Files:**
- Create: `aixon/providers/__init__.py`
- Create: `aixon/providers/base.py`
- Create: `tests/test_providers.py`

**Interfaces:**
- Consumes: `aixon.exceptions.AixonError`.
- Produces (contract §1.1, verbatim):
  - `aixon.providers.base.Provider` (ABC): class attributes `name: str`, `env_key: str`; abstract method `build(self, model: str, **params) -> "BaseChatModel"`.
  - `register_provider(provider: Provider) -> None` — keyed by `provider.name` (overwrites).
  - `get_provider(name: str) -> Provider` — raises `AixonError` if absent.
  - `resolve_provider_for_model(model: str) -> Provider` — infers provider from model name: `gpt*`/`o[0-9]*`/`text-*` → openai; `claude*` → anthropic; `gemini*` → google. Raises `AixonError` if no match.

> The fake provider and `FakeChatModel` are created in **Task 2** (`tests/_fakes.py`). This task's tests only exercise the registry + inference logic, which need no fake model — so Task 1 is independent of Task 2's fixtures and tests the inference rules directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_providers.py
from __future__ import annotations

import importlib
import os

import pytest

from aixon.exceptions import AixonError
from aixon.providers.base import (
    Provider,
    get_provider,
    register_provider,
    resolve_provider_for_model,
)


# ── A throwaway provider just for registry tests ─────────────────────────────

class _DummyProvider(Provider):
    name = "dummy"
    env_key = "DUMMY_API_KEY"

    def build(self, model: str, **params):
        raise NotImplementedError  # never called in these tests


def test_register_and_get_provider():
    register_provider(_DummyProvider())
    p = get_provider("dummy")
    assert isinstance(p, _DummyProvider)


def test_get_unknown_provider_raises():
    with pytest.raises(AixonError, match="no-such-provider"):
        get_provider("no-such-provider")


# ── resolve_provider_for_model (concrete providers registered below) ─────────

@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5.4", "o3", "o1-mini", "text-davinci-003"])
def test_resolve_openai_models(model):
    importlib.import_module("aixon.providers.openai")  # self-registers
    assert resolve_provider_for_model(model).name == "openai"


@pytest.mark.parametrize("model", ["claude-3-5-sonnet-20241022", "claude-opus-4"])
def test_resolve_anthropic_models(model):
    importlib.import_module("aixon.providers.anthropic")
    assert resolve_provider_for_model(model).name == "anthropic"


@pytest.mark.parametrize("model", ["gemini-2.0-flash", "gemini-1.5-pro"])
def test_resolve_google_models(model):
    importlib.import_module("aixon.providers.google")
    assert resolve_provider_for_model(model).name == "google"


def test_resolve_unknown_model_raises():
    with pytest.raises(AixonError, match="Cannot infer"):
        resolve_provider_for_model("totally-unknown-model-xyz")


# ── Vendor build (skipped if SDK not installed) ──────────────────────────────

def test_openai_provider_build():
    pytest.importorskip("langchain_openai")
    importlib.import_module("aixon.providers.openai")
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    model = get_provider("openai").build("gpt-4o-mini")
    assert hasattr(model, "invoke")


def test_anthropic_provider_build():
    pytest.importorskip("langchain_anthropic")
    importlib.import_module("aixon.providers.anthropic")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    model = get_provider("anthropic").build("claude-3-5-haiku-20241022")
    assert hasattr(model, "invoke")


def test_google_provider_build():
    pytest.importorskip("langchain_google_genai")
    importlib.import_module("aixon.providers.google")
    os.environ.setdefault("GOOGLE_API_KEY", "test-key")
    model = get_provider("google").build("gemini-2.0-flash")
    assert hasattr(model, "invoke")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_providers.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.providers'`.

- [ ] **Step 3: Create `aixon/providers/base.py`**

```python
# aixon/providers/base.py
"""Provider ABC, registry, and model-name inference.

Each concrete provider (OpenAI / Anthropic / Google) lives in its own
module under aixon/providers/ and self-registers at import time via
register_provider(). Provider SDK imports are LAZY (inside build()) so
importing this module — or any provider module — never fails due to a
missing vendor SDK.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from aixon.exceptions import AixonError

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Builds a LangChain BaseChatModel for one vendor.

    Reads the API key from the environment inside build(). Concrete
    providers live in aixon/providers/<vendor>.py.
    """

    name: str       # "openai" | "anthropic" | "google"
    env_key: str    # e.g. "OPENAI_API_KEY"

    @abstractmethod
    def build(self, model: str, **params: Any) -> "BaseChatModel":
        """Return a configured LangChain chat model.

        **params are passed through (temperature, max_tokens, top_p, etc.).
        The API key is read from os.getenv(self.env_key) inside this method.
        """


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_registry: dict[str, Provider] = {}


def register_provider(provider: Provider) -> None:
    """Register a provider instance keyed by provider.name (overwrites)."""
    _registry[provider.name] = provider


def get_provider(name: str) -> Provider:
    """Return the registered provider for *name*.

    Raises:
        AixonError: if no provider is registered under that name.
    """
    try:
        return _registry[name]
    except KeyError:
        available = sorted(_registry)
        raise AixonError(
            f"No provider registered as '{name}'. "
            f"Available: {available}. "
            f"Install the relevant extra (e.g. pip install aixon[openai]) "
            f"or call register_provider() with a custom Provider."
        )


# ---------------------------------------------------------------------------
# Model-name → provider inference
# ---------------------------------------------------------------------------

# Rules: (compiled regex, provider name). Evaluated in order; first match wins.
_INFERENCE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^(gpt-|o\d|text-)"), "openai"),
    (re.compile(r"^claude"), "anthropic"),
    (re.compile(r"^gemini"), "google"),
]


def resolve_provider_for_model(model: str) -> Provider:
    """Infer the provider from the model name and return the registered Provider.

    Rules:
    - gpt-* | o<digit>* | text-* → openai
    - claude*                    → anthropic
    - gemini*                    → google

    Raises:
        AixonError: if no rule matches or the inferred provider is not registered.
    """
    for pattern, provider_name in _INFERENCE_RULES:
        if pattern.match(model):
            return get_provider(provider_name)
    raise AixonError(
        f"Cannot infer provider for model '{model}'. "
        f"Pass provider= explicitly: LLM('{model}', provider='openai')."
    )
```

- [ ] **Step 4: Create `aixon/providers/__init__.py`**

```python
# aixon/providers/__init__.py
"""Provider sub-package.

Importing this package triggers self-registration of all three concrete
providers (OpenAI, Anthropic, Google). Because their SDK imports are lazy
(inside build()), this never raises ImportError for a missing vendor SDK.
"""
from aixon.providers.base import (
    Provider,
    get_provider,
    register_provider,
    resolve_provider_for_model,
)

# Trigger self-registration. Each module's top-level register_provider() call
# fires when the module is imported. SDK imports stay inside build().
from aixon.providers import anthropic as _anthropic  # noqa: E402,F401
from aixon.providers import google as _google  # noqa: E402,F401
from aixon.providers import openai as _openai  # noqa: E402,F401

__all__ = [
    "Provider",
    "get_provider",
    "register_provider",
    "resolve_provider_for_model",
]
```

> `__init__.py` imports the three concrete modules so that `import aixon.providers` registers all three. This is created now and the concrete modules land in Task 3 (next). If you run the provider tests between this step and Task 3, the `resolve_*` and vendor-build tests fail/error because `aixon.providers.openai` etc. don't exist yet — that is expected; finish Task 3 first. Until then, do NOT import `aixon.providers` (the package) — import `aixon.providers.base` directly, as the test file does.

- [ ] **Step 5: Run the registry-only tests (concrete providers land in Task 3)**

```bash
.venv/bin/python -m pytest tests/test_providers.py -v -k "register or unknown"
```

Expected: PASS for `test_register_and_get_provider`, `test_get_unknown_provider_raises`, `test_resolve_unknown_model_raises`. The `resolve_openai/anthropic/google` and vendor-build tests will ERROR on `ModuleNotFoundError: aixon.providers.openai` until Task 3 — that is expected at this step.

- [ ] **Step 6: Commit**

```bash
git add aixon/providers/__init__.py aixon/providers/base.py tests/test_providers.py
git commit -m "$(cat <<'EOF'
feat: Provider ABC + registry + model-name inference

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `tests/_fakes.py` — the single shared hermetic fixture module (contract §9.1)

**Files:**
- Create: `tests/_fakes.py` — **owned by Plan 2; imported (never redefined) by Plans 3/4/5/7.**

**Interfaces (contract §9.1, all required):**
- `FakeChatModel` — the EXACT class from §9.1, copied verbatim (validated against langchain 1.3 / core 1.4 / langgraph 1.2; drives `langchain.agents.create_agent` through a tool call then a final answer with no key and no network).
- `register_fake_provider() -> None` — idempotent; registers a `Provider` named `"fake"` whose `build()` returns a `FakeChatModel`.
- `FAKE_MODEL = "fake-1"`, `FAKE_PROVIDER = "fake"`.
- `make_llm(**params) -> LLM` — returns `LLM(FAKE_MODEL, provider=FAKE_PROVIDER, **params)`.
- `make_echo_agent(name="echo", *, hidden=False)` — defines/registers a concrete `Agent` subclass whose `invoke` echoes the last message and whose `stream` yields one content `Chunk` then `Chunk(done=True)`. Returns the registered instance.

> **Why a Provider subclass for "fake":** `Provider.build` is abstract, so `FakeProvider` must subclass `Provider` and implement `build`. `register_fake_provider()` is idempotent: calling it repeatedly just re-registers the same name (the registry overwrites). It is called from `make_llm` and at module import so any importer gets a working `LLM("fake-1", provider="fake")`.
>
> **`make_echo_agent` and the registry:** Plan 1's `Agent.__init_subclass__` auto-registers concrete subclasses at definition time. To allow repeated calls within one test session (e.g. `make_echo_agent("a")` then `make_echo_agent("b")`), generate a class name from `name` (suffix `Agent`) and set the instance's `.name`/`.hidden`. The autouse `reset_registry` fixture (Plan 1 conftest) clears the registry between tests, so name reuse across tests is safe.

- [ ] **Step 1: Create `tests/_fakes.py`**

```python
# tests/_fakes.py
"""Single owner of hermetic test doubles for aixon (contract §9.1).

Imported by Plan 2 tests and by Plans 3, 4, 5, 7. DO NOT redefine these
elsewhere. Everything here is offline: no API key, no network.

FakeChatModel is copied VERBATIM from the interface contract §9.1 and is
validated against langchain 1.3 / langchain-core 1.4 / langgraph 1.2 — it
drives langchain.agents.create_agent through a tool call then a final answer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aixon.agent import Agent
from aixon.message import Chunk, Message
from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    from aixon.llm import LLM


FAKE_MODEL = "fake-1"
FAKE_PROVIDER = "fake"


# ── FakeChatModel — VERBATIM from contract §9.1 (do not edit) ────────────────

class FakeChatModel(BaseChatModel):
    """Scriptable offline chat model. `script` is a list of AIMessages returned
    one per LLM call (set tool_calls on an AIMessage to drive a tool step)."""

    script: list = []
    _idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        return self  # tools ignored; script drives calls

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        i = self._idx
        msg = self.script[i] if i < len(self.script) else AIMessage(content="(done)")
        object.__setattr__(self, "_idx", i + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])


# Example script for a tool-calling test (used by Plan 3):
#   FakeChatModel(script=[
#       AIMessage(content="", tool_calls=[{"name":"get_weather","args":{"city":"Recife"},"id":"call_1"}]),
#       AIMessage(content="The weather in Recife is sunny."),
#   ])


# ── Fake provider ────────────────────────────────────────────────────────────

class FakeProvider(Provider):
    """Provider named 'fake' whose build() returns a FakeChatModel.

    The returned model's `script` can be overridden per test, e.g.:
        from tests._fakes import make_llm
        llm = make_llm()
        llm.chat_model.script = [AIMessage(content="hi")]
    A bare FakeChatModel() with an empty script echoes "(done)" for each call,
    which is enough for the LLM.complete / LLM.stream smoke tests below.
    """

    name = FAKE_PROVIDER
    env_key = "FAKE_API_KEY"

    def build(self, model: str, **params: Any) -> FakeChatModel:
        return FakeChatModel()


def register_fake_provider() -> None:
    """Register the 'fake' provider. Idempotent — safe to call repeatedly."""
    register_provider(FakeProvider())


# Register at import time so `LLM("fake-1", provider="fake")` works for any
# importer without an explicit call.
register_fake_provider()


# ── Convenience factories (used by Plans 3/4/5/7) ────────────────────────────

def make_llm(**params: Any) -> "LLM":
    """Return an LLM bound to the fake provider/model."""
    register_fake_provider()
    from aixon.llm import LLM  # local import: aixon.llm depends on providers

    return LLM(FAKE_MODEL, provider=FAKE_PROVIDER, **params)


def make_echo_agent(name: str = "echo", *, hidden: bool = False):
    """Define + register a concrete Agent that echoes the last message.

    invoke([... , Message(content="x")]) -> Message(role="assistant", content="x")
    stream(...) yields one content Chunk then Chunk(done=True).
    Returns the registered agent instance. Used by server/CLI/orchestrator
    tests that need an Agent but not a real LLM.
    """
    from typing import Iterator

    def invoke(self, messages: list[Message]) -> Message:
        last = messages[-1].content if messages else ""
        return Message(role="assistant", content=last)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        last = messages[-1].content if messages else ""
        yield Chunk(content=last)
        yield Chunk(done=True)

    cls_name = f"{name.capitalize()}Agent"
    cls = type(
        cls_name,
        (Agent,),
        {"invoke": invoke, "stream": stream, "name": name, "hidden": hidden},
    )
    # Agent.__init_subclass__ already instantiated + registered it; fetch it.
    from aixon.registry import get_registry

    return get_registry().resolve(name)
```

- [ ] **Step 2: There is no separate test here yet** — `tests/_fakes.py` is a helper module (no `test_` prefix), exercised by Tasks 4–6. Confirm it at least imports cleanly:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, '.'); import tests._fakes as f; print('FAKE_MODEL', f.FAKE_MODEL); print('FAKE_PROVIDER', f.FAKE_PROVIDER); print('FakeChatModel', f.FakeChatModel.__name__); print('make_llm', callable(f.make_llm)); print('make_echo_agent', callable(f.make_echo_agent))"
```

Expected:
```
FAKE_MODEL fake-1
FAKE_PROVIDER fake
FakeChatModel FakeChatModel
make_llm True
make_echo_agent True
```

> If the import fails with `ModuleNotFoundError: aixon.llm`, that is fine ONLY if it happens at call time — the top-level import must succeed because `aixon.llm` is imported lazily inside `make_llm`. If the top-level import fails, you imported `aixon.llm` at module top level by mistake; move it into `make_llm`.

- [ ] **Step 3: Commit**

```bash
git add tests/_fakes.py
git commit -m "$(cat <<'EOF'
test: tests/_fakes.py — single shared hermetic fixtures (FakeChatModel, fake provider, make_llm, make_echo_agent)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Concrete providers — OpenAI, Anthropic, Google (lazy SDK imports)

**Files:**
- Create: `aixon/providers/openai.py`
- Create: `aixon/providers/anthropic.py`
- Create: `aixon/providers/google.py`

**Interfaces (contract §1.2):**
- `OpenAIProvider(name="openai", env_key="OPENAI_API_KEY")` → `langchain_openai.ChatOpenAI` from `build()`.
- `AnthropicProvider(name="anthropic", env_key="ANTHROPIC_API_KEY")` → `langchain_anthropic.ChatAnthropic`.
- `GoogleProvider(name="google", env_key="GOOGLE_API_KEY")` → `langchain_google_genai.ChatGoogleGenerativeAI`.
- Each self-registers at module import time. SDK imports are LAZY (inside `build()`).

> The tests for this task already exist (written in Task 1): `test_resolve_openai/anthropic/google_models` and the three `test_*_provider_build` (which `importorskip` the SDK). After this task, all of `tests/test_providers.py` passes.

- [ ] **Step 1: Create `aixon/providers/openai.py`**

```python
# aixon/providers/openai.py
"""OpenAI provider — builds langchain_openai.ChatOpenAI.

Self-registers as 'openai' at import time. The langchain_openai import is
LAZY (inside build()) so importing this module never raises ImportError if
langchain-openai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class OpenAIProvider(Provider):
    name = "openai"
    env_key = "OPENAI_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_openai import ChatOpenAI  # lazy import

        api_key = os.getenv(self.env_key)
        return ChatOpenAI(model=model, api_key=api_key, **params)


register_provider(OpenAIProvider())
```

- [ ] **Step 2: Create `aixon/providers/anthropic.py`**

```python
# aixon/providers/anthropic.py
"""Anthropic provider — builds langchain_anthropic.ChatAnthropic.

Self-registers as 'anthropic' at import time. The langchain_anthropic import
is LAZY (inside build()) so importing this module never raises ImportError if
langchain-anthropic is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class AnthropicProvider(Provider):
    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_anthropic import ChatAnthropic  # lazy import

        api_key = os.getenv(self.env_key)
        return ChatAnthropic(model=model, api_key=api_key, **params)


register_provider(AnthropicProvider())
```

> `ChatAnthropic` accepts `api_key` (alias of `anthropic_api_key`) in langchain-anthropic; pass `api_key=` for consistency with the other providers.

- [ ] **Step 3: Create `aixon/providers/google.py`**

```python
# aixon/providers/google.py
"""Google provider — builds langchain_google_genai.ChatGoogleGenerativeAI.

Self-registers as 'google' at import time. The langchain_google_genai import
is LAZY (inside build()) so importing this module never raises ImportError if
langchain-google-genai is not installed; only build() will fail in that case.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from aixon.providers.base import Provider, register_provider

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class GoogleProvider(Provider):
    name = "google"
    env_key = "GOOGLE_API_KEY"

    def build(self, model: str, **params: Any) -> "BaseChatModel":
        from langchain_google_genai import ChatGoogleGenerativeAI  # lazy import

        api_key = os.getenv(self.env_key)
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, **params)


register_provider(GoogleProvider())
```

- [ ] **Step 4: Run the full provider test suite**

```bash
.venv/bin/python -m pytest tests/test_providers.py -v
```

Expected: all tests PASS. The three `test_*_provider_build` tests PASS if the SDKs are installed (they are, via `Task 0`'s `[openai,anthropic,google]` extras) and SKIP otherwise. (`build()` with a dummy key succeeds because LangChain defers the network call until `.invoke()`.)

- [ ] **Step 5: Confirm `import aixon.providers` registers all three without an SDK error**

```bash
.venv/bin/python -c "import aixon.providers; from aixon.providers.base import get_provider; print([get_provider(n).name for n in ('openai','anthropic','google')])"
```

Expected: `['openai', 'anthropic', 'google']`.

- [ ] **Step 6: Run the full suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add aixon/providers/openai.py aixon/providers/anthropic.py aixon/providers/google.py
git commit -m "$(cat <<'EOF'
feat: OpenAI, Anthropic, Google providers with lazy SDK imports

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `aixon/_langchain.py` — neutral ↔ LangChain message conversion (LangChain 1.x)

**Files:**
- Create: `aixon/_langchain.py`
- Create: `tests/test_langchain.py`

**Interfaces (contract §1.4):**
- `to_langchain(messages: list[Message]) -> list[BaseMessage]` — converts neutral `Message[]` to LangChain message objects (`SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage`).
- `from_langchain(msg: BaseMessage) -> Message` — converts a LangChain message back to a neutral `Message`. Carries `.content`; populates `tool_calls` if present; populates `reasoning` from `additional_kwargs["reasoning_content"]` if present.

> **LangChain 1.x note (validated):** `AIMessage(content="", tool_calls=[{"name":..,"args":..,"id":..}])` normalizes each tool call to include `"type": "tool_call"` and reorders keys. So `from_langchain` tests must NOT assert exact dict equality on `tool_calls`; assert on `name`/`args`/`id`. `ToolMessage` requires `tool_call_id`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_langchain.py
from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from aixon._langchain import from_langchain, to_langchain
from aixon.message import Message


# ── to_langchain ─────────────────────────────────────────────────────────────

def test_to_langchain_system():
    lc = to_langchain([Message(role="system", content="You are helpful.")])
    assert len(lc) == 1
    assert isinstance(lc[0], SystemMessage)
    assert lc[0].content == "You are helpful."


def test_to_langchain_user():
    lc = to_langchain([Message(role="user", content="Hello")])
    assert isinstance(lc[0], HumanMessage)
    assert lc[0].content == "Hello"


def test_to_langchain_assistant():
    lc = to_langchain([Message(role="assistant", content="Hi there")])
    assert isinstance(lc[0], AIMessage)
    assert lc[0].content == "Hi there"


def test_to_langchain_tool():
    lc = to_langchain(
        [Message(role="tool", content="42", tool_call_id="call_1", name="calc")]
    )
    assert isinstance(lc[0], ToolMessage)
    assert lc[0].content == "42"
    assert lc[0].tool_call_id == "call_1"


def test_to_langchain_mixed():
    lc = to_langchain(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="user msg"),
            Message(role="assistant", content="reply"),
        ]
    )
    assert [type(m).__name__ for m in lc] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
    ]


def test_to_langchain_unknown_role_raises():
    msg = Message.__new__(Message)
    object.__setattr__(msg, "role", "badrole")
    object.__setattr__(msg, "content", "x")
    object.__setattr__(msg, "name", None)
    object.__setattr__(msg, "tool_calls", [])
    object.__setattr__(msg, "tool_call_id", None)
    object.__setattr__(msg, "reasoning", None)
    with pytest.raises(ValueError, match="badrole"):
        to_langchain([msg])


# ── from_langchain ────────────────────────────────────────────────────────────

def test_from_langchain_ai_message():
    m = from_langchain(AIMessage(content="Hello back"))
    assert m.role == "assistant"
    assert m.content == "Hello back"
    assert m.tool_calls == []
    assert m.reasoning is None


def test_from_langchain_carries_tool_calls():
    lc = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "calc", "args": {"x": 1}}],
    )
    m = from_langchain(lc)
    assert len(m.tool_calls) == 1
    tc = m.tool_calls[0]
    assert tc["name"] == "calc"
    assert tc["args"] == {"x": 1}
    assert tc["id"] == "call_1"


def test_from_langchain_carries_reasoning_from_additional_kwargs():
    lc = AIMessage(
        content="answer",
        additional_kwargs={"reasoning_content": "I thought about it."},
    )
    m = from_langchain(lc)
    assert m.reasoning == "I thought about it."


def test_from_langchain_human_message():
    m = from_langchain(HumanMessage(content="Hi"))
    assert m.role == "user"
    assert m.content == "Hi"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_langchain.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'aixon._langchain'`.

- [ ] **Step 3: Create `aixon/_langchain.py`**

```python
# aixon/_langchain.py
"""Conversion helpers between neutral Message/Chunk and LangChain types.

INTERNAL to aixon. Public code speaks only Message/Chunk. LLM, LLMAgent,
ToolAgent, and Orchestrator call these helpers at the boundary where they
must interact with LangChain internals. Validated for LangChain 1.x.
"""
from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from aixon.message import Message


def to_langchain(messages: list[Message]) -> list[BaseMessage]:
    """Convert neutral Message[] to LangChain message objects.

    Mapping:
        system    → SystemMessage
        user      → HumanMessage
        assistant → AIMessage (tool_calls forwarded if present)
        tool      → ToolMessage (requires tool_call_id)
    """
    result: list[BaseMessage] = []
    for msg in messages:
        role = msg.role
        if role == "system":
            result.append(SystemMessage(content=msg.content))
        elif role == "user":
            result.append(HumanMessage(content=msg.content))
        elif role == "assistant":
            kwargs: dict = {"content": msg.content}
            if msg.tool_calls:
                kwargs["tool_calls"] = msg.tool_calls
            result.append(AIMessage(**kwargs))
        elif role == "tool":
            result.append(
                ToolMessage(
                    content=msg.content,
                    tool_call_id=msg.tool_call_id or "",
                    name=msg.name,
                )
            )
        else:
            raise ValueError(
                f"Unknown message role '{role}'. "
                f"Expected one of: system, user, assistant, tool."
            )
    return result


def from_langchain(msg: BaseMessage) -> Message:
    """Convert a LangChain BaseMessage to a neutral Message.

    - Role inferred from the LangChain type.
    - tool_calls: forwarded from AIMessage.tool_calls (list of dicts).
    - reasoning: read from additional_kwargs['reasoning_content'] if present.
    """
    if isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    else:
        role = "assistant"  # safe fallback for unknown LangChain types

    content = msg.content if isinstance(msg.content, str) else str(msg.content)

    tool_calls: list[dict] = []
    if isinstance(msg, AIMessage) and msg.tool_calls:
        tool_calls = [dict(tc) for tc in msg.tool_calls]

    reasoning: str | None = None
    if getattr(msg, "additional_kwargs", None):
        reasoning = msg.additional_kwargs.get("reasoning_content")

    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning or None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_langchain.py -v
```

Expected: PASS (10 tests).

- [ ] **Step 5: Run the full suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add aixon/_langchain.py tests/test_langchain.py
git commit -m "$(cat <<'EOF'
feat: neutral <-> LangChain 1.x message conversion helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `LLM` class — lazy `chat_model`, neutral `complete`/`stream` (LangChain 1.x)

**Files:**
- Create: `aixon/llm.py`
- Create: `tests/test_llm.py`

**Interfaces (contract §1.3):**
- `LLM(model: str, *, provider: str | None = None, **params)` — declarative handle; stores `model`, `params`, `_provider_name`, `_chat_model=None`.
- `LLM.chat_model` (property) — lazily build + cache the LangChain `BaseChatModel` (explicit provider, else inferred).
- `LLM.complete(messages) -> Message` — `chat_model.invoke(to_langchain(messages))` → `from_langchain(...)`.
- `LLM.stream(messages) -> Iterator[Chunk]` — `chat_model.stream(to_langchain(messages))` yields chunks; emit `Chunk(content=delta)` for each non-empty content delta, then a final `Chunk(done=True)`.

> **LangChain 1.x streaming note (validated):** `BaseChatModel.stream` yields `AIMessageChunk` objects when the model defines `_stream`; when it only defines `_generate` (like our `FakeChatModel`), `stream` falls back and yields a single `AIMessage`. Both expose `.content` as a string. So read the delta via `getattr(chunk, "content", "")` and guard for non-str — this works for real providers (chunks) AND the fake (one message). Do NOT assume `AIMessageChunk`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
from __future__ import annotations

from langchain_core.messages import AIMessage

from tests._fakes import FakeChatModel, make_llm  # registers fake provider
from aixon.llm import LLM
from aixon.message import Chunk, Message


# ── Construction + lazy build ─────────────────────────────────────────────────

def test_llm_construction_stores_model_and_params():
    llm = LLM("fake-1", provider="fake", temperature=0.5)
    assert llm.model == "fake-1"
    assert llm.params == {"temperature": 0.5}
    assert llm._provider_name == "fake"
    assert llm._chat_model is None  # not built yet


def test_chat_model_is_fake_chat_model():
    llm = LLM("fake-1", provider="fake")
    assert llm._chat_model is None
    cm = llm.chat_model
    assert isinstance(cm, FakeChatModel)
    assert llm._chat_model is cm  # cached


def test_chat_model_returns_same_instance_on_second_access():
    llm = LLM("fake-1", provider="fake")
    assert llm.chat_model is llm.chat_model


def test_llm_infers_provider_from_model_name():
    import aixon.providers  # registers openai/anthropic/google
    from aixon.providers.base import resolve_provider_for_model

    assert resolve_provider_for_model("gpt-4o").name == "openai"


# ── complete (offline) ────────────────────────────────────────────────────────

def test_complete_returns_neutral_message():
    llm = LLM("fake-1", provider="fake")
    llm.chat_model.script = [AIMessage(content="pong")]
    result = llm.complete([Message(role="user", content="ping")])
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "pong"


def test_complete_default_script_echoes_done():
    # Empty script -> FakeChatModel returns AIMessage("(done)")
    llm = make_llm()
    result = llm.complete([Message(role="user", content="x")])
    assert result.content == "(done)"


# ── stream (offline) ──────────────────────────────────────────────────────────

def test_stream_yields_content_then_done():
    llm = LLM("fake-1", provider="fake")
    llm.chat_model.script = [AIMessage(content="streamed")]
    chunks = list(llm.stream([Message(role="user", content="hi")]))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any(c.content for c in chunks)
    assert chunks[-1].done is True


def test_stream_final_chunk_has_done_true():
    llm = make_llm()
    chunks = list(llm.stream([Message(role="user", content="x")]))
    assert chunks[-1].done is True
    for c in chunks[:-1]:
        assert c.done is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.llm'`.

- [ ] **Step 3: Create `aixon/llm.py`**

```python
# aixon/llm.py
"""Declarative LLM handle.

Usage on an agent class body:
    class MyAgent(LLMAgent):
        llm = LLM("gpt-5.4", temperature=0.2)

The LLM handle is lazy: it does not build the underlying LangChain model
until the first access to .chat_model (or .complete / .stream). Declaring an
LLM therefore needs neither an API key nor an installed vendor SDK.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

from aixon._langchain import from_langchain, to_langchain
from aixon.message import Chunk, Message

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class LLM:
    """Declarative handle for a LangChain chat model behind a neutral boundary."""

    def __init__(self, model: str, *, provider: str | None = None, **params: Any):
        self.model = model
        self.params = params
        self._provider_name = provider          # None → inferred from model name
        self._chat_model: "BaseChatModel | None" = None  # lazy

    @property
    def chat_model(self) -> "BaseChatModel":
        """Lazily build and cache the LangChain model.

        Used directly by ToolAgent and Orchestrator (Plan 3+). The provider
        must already be registered (via importing its module or a custom
        register_provider() call).
        """
        if self._chat_model is None:
            from aixon.providers.base import (
                get_provider,
                resolve_provider_for_model,
            )

            if self._provider_name is not None:
                provider = get_provider(self._provider_name)
            else:
                provider = resolve_provider_for_model(self.model)
            self._chat_model = provider.build(self.model, **self.params)
        return self._chat_model

    def complete(self, messages: list[Message]) -> Message:
        """Single-shot neutral completion. Used by LLMAgent.invoke."""
        lc_result = self.chat_model.invoke(to_langchain(messages))
        return from_langchain(lc_result)

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Neutral streaming. Used by LLMAgent.stream.

        Yields Chunk(content=delta) per non-empty delta, then Chunk(done=True).
        Works whether the model yields AIMessageChunk deltas (real providers)
        or a single AIMessage (the fake, which has no _stream).
        """
        for lc_chunk in self.chat_model.stream(to_langchain(messages)):
            content = getattr(lc_chunk, "content", "")
            if isinstance(content, str) and content:
                yield Chunk(content=content)
        yield Chunk(done=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_llm.py -v
```

Expected: PASS (8 tests).

- [ ] **Step 5: Run the full suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add aixon/llm.py tests/test_llm.py
git commit -m "$(cat <<'EOF'
feat: LLM declarative handle — lazy chat_model, neutral complete/stream (langchain 1.x)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `LLMAgent` abstract subtype (pure-LLM, hermetic offline run)

**Files:**
- Create: `aixon/agents/__init__.py`
- Create: `aixon/agents/llm_agent.py`
- Create: `tests/test_llm_agent.py`

**Interfaces (contract §1.5):**
- `LLMAgent(Agent, abstract=True)` — `_suffix = "Agent"`. Pure-LLM: no tools, no langgraph; delegates to `self.llm`.
- Class attributes: `llm: LLM` (REQUIRED — validated in `__init_subclass__`), `prompt: str = ""` (optional system prompt prepended to messages).
- `invoke(messages) -> Message` — prepend the system prompt (if non-empty) and delegate to `self.llm.complete`.
- `stream(messages) -> Iterator[Chunk]` — prepend the system prompt (if non-empty) and delegate to `self.llm.stream`.
- `__init_subclass__` calls `super().__init_subclass__(**kwargs)` first (so Agent's suffix/registration runs), then — for concrete subclasses only (`abstract` falsy) — validates that `llm` is an `LLM` instance; raises `AixonError` if missing.

> **Prompt prepending rule:** if `self.prompt` is non-empty, prepend `Message(role="system", content=self.prompt)`. Never mutate the caller's list — build a new one.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_agent.py
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tests._fakes import register_fake_provider  # registers fake provider

from aixon.agents.llm_agent import LLMAgent
from aixon.exceptions import AixonError, NamingError
from aixon.llm import LLM
from aixon.message import Chunk, Message
from aixon.registry import get_registry

register_fake_provider()


# ── Valid concrete subclass runs OFFLINE ─────────────────────────────────────

class EchoLLMAgent(LLMAgent):
    llm = LLM("fake-1", provider="fake")
    description = "Echoes via fake LLM"


def test_llm_agent_registers_itself():
    assert isinstance(get_registry().resolve("echollmagent"), EchoLLMAgent)


def test_llm_agent_invoke_runs_offline():
    agent = get_registry().resolve("echollmagent")
    agent.llm.chat_model.script = [AIMessage(content="pong")]
    result = agent.invoke([Message(role="user", content="ping")])
    assert isinstance(result, Message)
    assert result.role == "assistant"
    assert result.content == "pong"


def test_llm_agent_stream_runs_offline():
    agent = get_registry().resolve("echollmagent")
    agent.llm.chat_model.script = [AIMessage(content="streamed")]
    chunks = list(agent.stream([Message(role="user", content="hi")]))
    assert all(isinstance(c, Chunk) for c in chunks)
    assert any(c.content for c in chunks)
    assert chunks[-1].done is True


# ── System prompt prepending ──────────────────────────────────────────────────

class PromptedLLMAgent(LLMAgent):
    llm = LLM("fake-1", provider="fake")
    prompt = "You are a helpful assistant."


def test_prompt_prepended_as_system_message():
    agent = get_registry().resolve("promptedllmagent")
    seen: list[list[Message]] = []
    original = agent.llm.complete

    def capturing(messages):
        seen.append(list(messages))
        return original(messages)

    agent.llm.complete = capturing
    agent.invoke([Message(role="user", content="hello")])
    agent.llm.complete = original

    assert len(seen) == 1
    assert seen[0][0].role == "system"
    assert seen[0][0].content == "You are a helpful assistant."


def test_prompt_does_not_mutate_caller_list():
    agent = get_registry().resolve("promptedllmagent")
    msgs = [Message(role="user", content="hello")]
    agent.invoke(msgs)
    assert len(msgs) == 1


def test_no_prompt_does_not_prepend():
    agent = get_registry().resolve("echollmagent")
    assert agent.prompt == ""
    seen: list[list[Message]] = []
    original = agent.llm.complete

    def capturing(messages):
        seen.append(list(messages))
        return original(messages)

    agent.llm.complete = capturing
    agent.invoke([Message(role="user", content="x")])
    agent.llm.complete = original
    assert seen[0][0].role == "user"


# ── Validation ────────────────────────────────────────────────────────────────

def test_missing_llm_raises_aixon_error():
    with pytest.raises(AixonError, match="llm"):
        class NoLLMAgent(LLMAgent):
            pass


def test_llm_agent_itself_not_registered():
    names = [a.name for a in get_registry().all()]
    assert "llmagent" not in names


def test_bad_suffix_raises():
    with pytest.raises(NamingError, match="Agent"):
        class BadName(LLMAgent):
            llm = LLM("fake-1", provider="fake")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_llm_agent.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.agents'`.

- [ ] **Step 3: Create `aixon/agents/__init__.py`**

```python
# aixon/agents/__init__.py
"""Agent subtypes sub-package.

Import specific subtypes as needed:
    from aixon.agents.llm_agent import LLMAgent
"""
```

- [ ] **Step 4: Create `aixon/agents/llm_agent.py`**

```python
# aixon/agents/llm_agent.py
"""LLMAgent — abstract subtype for direct LLM access (no tool-calling loop).

Pure-LLM: it does NOT build a langgraph graph and has no tools. It prepends an
optional system prompt and delegates to its LLM. (Tool-calling lives in Plan 3's
ToolAgent, which uses langchain.agents.create_agent.)

Consumer usage:
    class Athena(LLMAgent):
        llm = LLM("gpt-5.4", temperature=0.2)
        prompt = "You are a strategic planner."
        description = "Strategic planning assistant"

Athena auto-registers, gets suffix-validated, and is ready to be routed by name.
"""
from __future__ import annotations

from typing import Iterator

from aixon.agent import Agent
from aixon.exceptions import AixonError
from aixon.llm import LLM
from aixon.message import Chunk, Message


class LLMAgent(Agent, abstract=True):
    """Abstract subtype for agents that delegate directly to an LLM.

    Required class attribute:
        llm: LLM   — e.g. LLM("gpt-5.4", temperature=0.2)
    Optional class attribute:
        prompt: str   — system prompt prepended to every invocation.
    """

    _suffix: str = "Agent"
    llm: LLM         # declared; absence on a concrete subclass is an error
    prompt: str = ""

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: object) -> None:
        # Let Agent handle suffix validation + auto-registration first.
        super().__init_subclass__(abstract=abstract, **kwargs)
        if abstract:
            return
        llm_val = cls.__dict__.get("llm") or getattr(cls, "llm", None)
        if not isinstance(llm_val, LLM):
            raise AixonError(
                f"'{cls.__name__}' must declare a class-level 'llm' attribute "
                f"of type LLM (e.g. llm = LLM('gpt-5.4')). Got: {llm_val!r}."
            )

    def invoke(self, messages: list[Message]) -> Message:
        """Prepend system prompt (if any) and delegate to self.llm.complete."""
        return self.llm.complete(self._with_prompt(messages))

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Prepend system prompt (if any) and delegate to self.llm.stream."""
        yield from self.llm.stream(self._with_prompt(messages))

    def _with_prompt(self, messages: list[Message]) -> list[Message]:
        """Return a new list with the system prompt prepended if set.

        Never mutates the caller's list.
        """
        if self.prompt:
            return [Message(role="system", content=self.prompt), *messages]
        return list(messages)
```

> **Validation timing:** `super().__init_subclass__` runs Agent's logic, which on a concrete subclass with a bad suffix raises `NamingError` BEFORE the `llm` check — so the suffix test sees `NamingError`, and a well-named class missing `llm` sees `AixonError`. Both are correct per §1.5.

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_llm_agent.py -v
```

Expected: PASS (10 tests). These prove `LLMAgent` runs fully OFFLINE via `tests/_fakes.py` (contract §1.6 / requirement E).

- [ ] **Step 6: Run the full suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add aixon/agents/__init__.py aixon/agents/llm_agent.py tests/test_llm_agent.py
git commit -m "$(cat <<'EOF'
feat: LLMAgent pure-LLM abstract subtype with llm validation and prompt prepending

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Public exports + final `aixon/__init__.py`

**Files:**
- Modify: `aixon/__init__.py` — add `LLM`, `Provider`, `register_provider`, `get_provider`, `LLMAgent`.
- Create: `tests/test_exports.py`

**Interfaces (contract §1.7 + §9.4):**
- `aixon.LLM`, `aixon.Provider`, `aixon.register_provider`, `aixon.get_provider`, `aixon.LLMAgent` all importable from `import aixon`.
- Per §9.4, top-level layers whose deps live in an extra may be guarded. `LLMAgent` imports langchain lazily (via `aixon.llm` → `aixon._langchain`, which imports `langchain_core`). Since this plan's `llm` extra is installed in the dedicated venv, a direct import is fine; but to keep `import aixon` working on a bare install (no `llm` extra), wrap the `LLM`/`LLMAgent` imports in a `try/except ImportError` and only add them to `__all__` when present. `Provider`/`register_provider`/`get_provider` import only `aixon.exceptions` (no langchain) and are always exported.

- [ ] **Step 1: Write the failing export smoke-test**

```python
# tests/test_exports.py
"""Smoke-test: Plan 2 symbols importable from the top-level aixon namespace."""
from tests._fakes import register_fake_provider

register_fake_provider()

import aixon


def test_llm_exported():
    from aixon import LLM
    assert LLM("fake-1", provider="fake").model == "fake-1"


def test_provider_exported():
    from aixon import Provider
    assert isinstance(aixon.Provider, type)


def test_register_get_provider_exported():
    from aixon import get_provider, register_provider
    assert callable(register_provider)
    assert callable(get_provider)


def test_llm_agent_exported():
    import inspect
    from aixon import LLMAgent
    assert inspect.isclass(LLMAgent)


def test_plan1_exports_still_present():
    from aixon import (  # noqa: F401
        Agent,
        AgentTool,
        AixonError,
        AgentNotFoundError,
        CompositionCycleError,
        NamingError,
        RegistrationError,
        Chunk,
        Message,
        Role,
        Logger,
        autodiscover,
        get_registry,
        reset_registry,
    )
    assert True
```

- [ ] **Step 2: Run to verify the test fails**

```bash
.venv/bin/python -m pytest tests/test_exports.py -v
```

Expected: FAIL with `ImportError: cannot import name 'LLM' from 'aixon'` (or `AttributeError`).

- [ ] **Step 3: Update `aixon/__init__.py`**

Overwrite the file with its complete final contents:

```python
# aixon/__init__.py
"""aixon — declarative AI-agent framework."""

from aixon.agent import Agent, AgentTool
from aixon.discovery import autodiscover
from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)
from aixon.logging import Logger
from aixon.message import Chunk, Message, Role
from aixon.providers.base import Provider, get_provider, register_provider
from aixon.registry import Registry, get_registry, reset_registry

__all__ = [
    # Plan 1 — foundation
    "Agent",
    "AgentTool",
    "autodiscover",
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    "Logger",
    "Message",
    "Chunk",
    "Role",
    "Registry",
    "get_registry",
    "reset_registry",
    # Plan 2 — providers (no langchain dep at import)
    "Provider",
    "get_provider",
    "register_provider",
]

# Plan 2 — LLM + LLMAgent. These pull in langchain_core (the `llm` extra).
# Guard so `import aixon` still works on a bare install without that extra
# (contract §9.4).
try:
    from aixon.llm import LLM
    from aixon.agents.llm_agent import LLMAgent

    __all__ += ["LLM", "LLMAgent"]
except ImportError:  # pragma: no cover - bare install without [llm]
    pass
```

> **Import order:** `aixon.providers.base` imports only `aixon.exceptions` (always safe). `aixon.llm` imports `aixon._langchain` → `langchain_core`; if absent it raises `ImportError`, caught by the guard. No circular imports: `aixon.agents.llm_agent` imports `aixon.agent` (already loaded) and `aixon.llm`.

- [ ] **Step 4: Run the export tests**

```bash
.venv/bin/python -m pytest tests/test_exports.py -v
```

Expected: PASS (5 tests). (In the dedicated venv the `llm` extra is installed, so `LLM`/`LLMAgent` are present and the guard's `try` branch succeeds.)

- [ ] **Step 5: Run the full suite**

```bash
.venv/bin/python -m pytest -v
```

Expected: all tests PASS across Plan 1 + Plan 2.

- [ ] **Step 6: Commit**

```bash
git add aixon/__init__.py tests/test_exports.py
git commit -m "$(cat <<'EOF'
feat: export LLM, LLMAgent, Provider, register_provider, get_provider from aixon

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

### Spec coverage (contract §1 + the required langgraph-native changes)

| Contract item | Covered by | Status |
|---|---|---|
| §9.5 Dedicated `.venv`; every run uses `.venv/bin/python -m pytest` | Task 0 + every run step | ✓ |
| §1.7/§9.2 `llm = ["langchain>=1.0","langchain-core>=1.0","langgraph>=1.0"]`, vendor extras, union into `all`, no `<1` pin | Task 0 `pyproject.toml` | ✓ |
| §1.1 `Provider` ABC (`name`, `env_key`, `build`) | Task 1 `providers/base.py` | ✓ |
| §1.1 `register_provider` / `get_provider` | Task 1 `providers/base.py` | ✓ |
| §1.1 `resolve_provider_for_model` (gpt\*/o[0-9]\*/text-\*→openai; claude\*→anthropic; gemini\*→google) | Task 1 `providers/base.py` | ✓ |
| §1.2 `OpenAIProvider`/`AnthropicProvider`/`GoogleProvider`, lazy SDK imports | Task 3 | ✓ |
| §1.3 `LLM.__init__(model, *, provider, **params)` | Task 5 `llm.py` | ✓ |
| §1.3 `LLM.chat_model` lazy property | Task 5 `llm.py` | ✓ |
| §1.3 `LLM.complete` = invoke(to_langchain) → from_langchain | Task 5 `llm.py` | ✓ |
| §1.3 `LLM.stream` = chat_model.stream → Chunk deltas + final `done=True` | Task 5 `llm.py` | ✓ |
| §1.4 `to_langchain` / `from_langchain` (content, tool_calls, reasoning) | Task 4 `_langchain.py` | ✓ |
| §1.5 `LLMAgent(Agent, abstract=True)`, pure-LLM, delegates to `self.llm` | Task 6 `agents/llm_agent.py` | ✓ |
| §1.5 `llm` required + validated in `__init_subclass__`; `prompt` prepended | Task 6 | ✓ |
| §9.1 `tests/_fakes.py` owns ALL: `register_fake_provider()` idempotent, `FAKE_MODEL`/`FAKE_PROVIDER`, verbatim `FakeChatModel`, `make_llm`, `make_echo_agent`; fake `build()` returns `FakeChatModel` | Task 2 | ✓ |
| §1.6 No real keys/network; vendor tests `pytest.importorskip` | Tasks 1–7 | ✓ |
| §1.7/§9.4 Export `LLM`,`Provider`,`register_provider`,`get_provider`,`LLMAgent`; guard langchain-backed exports | Task 7 | ✓ |
| Requirement C: test that `LLM("fake-1", provider="fake").chat_model` is a `FakeChatModel`; `complete`/`stream` work offline | Task 5 `test_llm.py` | ✓ |
| Requirement E: hermetic test proving `LLMAgent` runs offline | Task 6 `test_llm_agent.py` | ✓ |

### Placeholder scan

No "TBD", "TODO", "implement later", "add error handling", or "similar to Task N" left. Every step shows complete, runnable code or an exact command with expected output. No `AgentExecutor`, `create_tool_calling_agent`, or `create_react_agent` anywhere (langgraph-native). No `langchain>=0.3` / `<1` pin anywhere. ✓

### Type consistency (vs contract)

- `Message(role, content, name, tool_calls, tool_call_id, reasoning)` used identically across `_langchain.py`, `llm.py`, `llm_agent.py`, `_fakes.py`, and tests. ✓
- `Chunk(content, reasoning, done)` yielded correctly in `LLM.stream`, `LLMAgent.stream`, and `make_echo_agent`. ✓
- `LLM(model, *, provider, **params)` matches §1.3 verbatim; `make_llm`/tests match. ✓
- `Provider.build(self, model: str, **params) -> BaseChatModel` — concrete providers and `FakeProvider` match. ✓
- `LLMAgent.__init_subclass__` calls `super().__init_subclass__(abstract=abstract, **kwargs)` before its `llm` check — consistent with §1.5 and Plan 1's ABCMeta pattern. ✓
- `register_fake_provider`/`FAKE_MODEL`/`FAKE_PROVIDER`/`FakeChatModel`/`make_llm`/`make_echo_agent` names match §9.1 exactly; `FakeChatModel` body is the §9.1 source verbatim. ✓
- `from_langchain` reads `additional_kwargs["reasoning_content"]` and copies `tool_calls` as plain dicts (LangChain 1.x normalizes tool-call dicts, so tests assert on `name`/`args`/`id`, not dict equality). ✓

### Contract ambiguities resolved (one line each)

1. **`FakeChatModel` has no `_stream`, so `LLM.stream` reads `.content` off whatever the model yields** — validated: an `_generate`-only model makes `BaseChatModel.stream` yield one `AIMessage` (not `AIMessageChunk`), so `getattr(chunk, "content", "")` covers both real-provider chunks and the fake.
2. **No stale `tests/test_providers.py` exists in the repo** — the old plan's "Task 1: stale test cleanup" is dropped; Task 1 here creates `test_providers.py` fresh.
3. **`all` extra union** — Plan 1 left `all` holding only `server`+`cli` deps; this plan unions the LLM/vendor deps in without dropping the existing entries (per §9.2 merge rule).
4. **LangChain 1.x normalizes tool-call dicts** (adds `"type":"tool_call"`, reorders keys) — `from_langchain` tests assert on `name`/`args`/`id` rather than exact dict equality, since strict equality would fail under core 1.4.
5. **`anthropic` provider key kwarg** — `ChatAnthropic` accepts `api_key` (alias of `anthropic_api_key`) in langchain-anthropic 1.x, so `build()` passes `api_key=` uniformly with the other two providers.
