# aixon Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core scaffolding of the `aixon` framework — package layout, packaging, neutral message types, the `Agent` base class with suffix validation and auto-registration, the agent registry, and autodiscovery — so that consumer projects can declare agents that self-register and are discoverable, even before any LLM/server layer exists.

**Architecture:** A declarative base-class framework in the spirit of `restmcp`. The `Agent` base enforces a class-name suffix via `__init_subclass__`, auto-instantiates concrete subclasses at definition time, and registers each instance into a process-global `Registry`. Agents speak only **neutral types** (`Message[]` in, `Message`/`Chunk` out) — no protocol or provider type ever crosses into the runtime. `autodiscover()` imports every module in a consumer package to trigger registration.

**Tech Stack:** Python 3.11+, `pydantic` v2 (neutral types as dataclasses; pydantic reserved for later layers), `click` (CLI, later plans), `hatchling` build backend, `pytest`.

## Global Constraints

- `requires-python >= 3.11` — copied verbatim from spec packaging section.
- Build backend: `hatchling` (mirror restmcp).
- Package name `aixon`; import name `aixon`.
- `[project.scripts] aixon = "aixon.cli:app"` (the `cli` module is a stub in this plan; the real CLI lands in Plan 7).
- **Suffix rule:** every concrete `Agent` subclass name must end with its declared `_suffix` (`"Agent"` for `LLMAgent`/`ToolAgent` subtypes, `"Orchestrator"` for `Orchestrator`), validated in `__init_subclass__` and raising **before** the process is usable. Abstract intermediate classes are exempt via `abstract=True`.
- **No protocol/provider types in the runtime:** `agent.py` and `message.py` must not import from `aixon.server`, `aixon.providers`, OpenAI, or Anthropic. (This is why neutral types live in `aixon/message.py`, not in `server/protocol.py` as the spec's draft layout listed — documented deviation: it prevents a runtime→server import cycle. `server/protocol.py` will re-export them in Plan 5.)
- Error messages follow restmcp's tone: state what was got and how to fix it.
- **Logging:** terminal diagnostics go through `aixon.logging.Logger` (a thin `logging` wrapper, level via `LOG_LEVEL` env, default `INFO`) — mirrors restmcp. Lifecycle events (agent registration, autodiscover) emit log lines so you can see what comes up in the terminal. **Distinct from streaming the agent's own content/reasoning to the terminal** — that is the `reasoning` channel (Plan 3) + the `aixon chat` CLI (Plan 7), not logging.

---

### Task 1: Package skeleton, packaging, and exceptions

**Files:**
- Create: `pyproject.toml`
- Create: `README.md` (minimal — `readme = "README.md"` in pyproject requires it to exist or the build fails; Plan 8 expands it)
- Create: `aixon/__init__.py`
- Create: `aixon/exceptions.py`
- Create: `aixon/cli.py` (stub so the script entry point resolves)
- Create: `tests/conftest.py`
- Test: `tests/test_exceptions.py`

> Do NOT create `tests/__init__.py`. Keeping `tests/` a non-package lets pytest use rootdir import mode (mirrors restmcp) and avoids package-import-mode surprises.

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `aixon.exceptions.AixonError(Exception)` — base.
  - `AixonError(message: str)` subclasses: `NamingError`, `RegistrationError`, `AgentNotFoundError`, `CompositionCycleError`.
  - `aixon.cli.app` — a callable placeholder (`def app(): ...`).
  - `tests/conftest.py` exposes an autouse `reset_registry` fixture (defined in Task 3; here it is a no-op placeholder so imports work — replaced in Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exceptions.py
import pytest
from aixon.exceptions import (
    AixonError,
    NamingError,
    RegistrationError,
    AgentNotFoundError,
    CompositionCycleError,
)


@pytest.mark.parametrize(
    "exc",
    [NamingError, RegistrationError, AgentNotFoundError, CompositionCycleError],
)
def test_all_exceptions_subclass_aixon_error(exc):
    assert issubclass(exc, AixonError)


def test_exception_carries_message():
    err = NamingError("bad name")
    assert str(err) == "bad name"
    assert err.message == "bad name"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_exceptions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon'`.

- [ ] **Step 3: Write the packaging and source files**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "aixon"
version = "0.0.1"
description = "Declarative AI-agent framework — composable agents, multi-agent orchestration, and a protocol-decoupled server"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    { name = "Jorge Henrique Moreira Santana", email = "jorge.henrique.moreira.santana@gmail.com" },
]
license = { text = "MIT" }
keywords = ["ai", "llm", "agents", "langgraph", "framework", "orchestration", "openai"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries :: Application Frameworks",
    "Operating System :: OS Independent",
]
dependencies = [
    "pydantic>=2.0",
    "click>=8.0",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[project.urls]
Homepage = "https://github.com/JorgeHSantana/aixon"
Repository = "https://github.com/JorgeHSantana/aixon"
"Bug Tracker" = "https://github.com/JorgeHSantana/aixon/issues"

[project.scripts]
aixon = "aixon.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["aixon"]
```

> Note: server/LLM/orchestration deps (`fastapi`, `uvicorn`, `langgraph`, `langchain-core`, provider SDKs) are intentionally NOT added here. Each later plan adds the dependencies it needs, so the package stays installable at every step.

```markdown
<!-- README.md -->
# aixon

Declarative AI-agent framework — composable agents, multi-agent orchestration,
and a protocol-decoupled server. Documentation lands in a later milestone.
```

```python
# aixon/exceptions.py
"""Exception hierarchy for aixon. Every error subclasses ``AixonError`` and
carries a human-readable ``message``."""


class AixonError(Exception):
    """Base exception for aixon."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class NamingError(AixonError):
    """A subclass violated a required class-name suffix."""


class RegistrationError(AixonError):
    """An agent could not be registered (duplicate name or alias clash)."""


class AgentNotFoundError(AixonError):
    """No registered agent matches the requested name or alias."""


class CompositionCycleError(AixonError):
    """A cycle was detected in the agent composition graph (A uses B as a
    tool and B uses A, directly or transitively)."""
```

```python
# aixon/cli.py
"""CLI entry point. Real commands (chat/new/serve/list) land in a later plan;
this stub exists so the ``aixon`` console script resolves after install."""


def app() -> None:  # pragma: no cover - replaced by the real CLI later
    raise SystemExit("aixon CLI is not implemented yet.")
```

```python
# aixon/__init__.py
"""aixon — declarative AI-agent framework."""

from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)

__all__ = [
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
]
```

```python
# tests/conftest.py
import pytest


@pytest.fixture(autouse=True)
def reset_registry():
    # Replaced in Task 3 with a real registry reset. No-op for now.
    yield
```

- [ ] **Step 4: Install the package in editable mode**

Run: `cd /Users/jorge/Documents/Git/aixon && python -m pip install -e ".[dev]"`
Expected: `Successfully installed aixon-0.0.1`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_exceptions.py -v`
Expected: PASS (5 tests: 4 parametrized + 1).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md aixon/ tests/
git commit -m "feat: package skeleton, packaging, and exception hierarchy"
```

---

### Task 2: Neutral message types

**Files:**
- Create: `aixon/message.py`
- Modify: `aixon/__init__.py` (export `Message`, `Chunk`, `Role`)
- Test: `tests/test_message.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `aixon.message.Role = Literal["system", "user", "assistant", "tool"]`.
  - `aixon.message.Message` — frozen-ish dataclass with fields: `role: Role`, `content: str = ""`, `name: str | None = None`, `tool_calls: list[dict] = []`, `tool_call_id: str | None = None`, `reasoning: str | None = None`. Method `to_dict() -> dict` (omits empty optional fields).
  - `aixon.message.Chunk` — dataclass: `content: str = ""`, `reasoning: str = ""`, `done: bool = False`.
  - Both re-exported from `aixon`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_message.py
from aixon import Message, Chunk


def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.role == "user"
    assert m.content == "hi"
    assert m.tool_calls == []
    assert m.reasoning is None


def test_message_to_dict_omits_empty_optionals():
    m = Message(role="user", content="hi")
    assert m.to_dict() == {"role": "user", "content": "hi"}


def test_message_to_dict_includes_set_optionals():
    m = Message(role="tool", content="42", tool_call_id="call_1", name="calc")
    d = m.to_dict()
    assert d["tool_call_id"] == "call_1"
    assert d["name"] == "calc"


def test_tool_calls_are_per_instance():
    a = Message(role="assistant")
    b = Message(role="assistant")
    a.tool_calls.append({"id": "x"})
    assert b.tool_calls == []


def test_chunk_defaults():
    c = Chunk()
    assert c.content == ""
    assert c.reasoning == ""
    assert c.done is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_message.py -v`
Expected: FAIL with `ImportError: cannot import name 'Message' from 'aixon'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/message.py
"""Neutral message types. The agent runtime speaks ONLY these — protocol
adapters (Plan 5) translate wire formats to and from them. Nothing here may
import a provider or protocol module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single neutral message. ``tool_calls`` carries provider-agnostic
    tool-call dicts; ``reasoning`` carries model reasoning when present."""

    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    reasoning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict, omitting empty optional fields."""
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        return data


@dataclass
class Chunk:
    """A streamed delta from an Agent. ``content`` and ``reasoning`` are
    additive text deltas; ``done`` marks the final chunk of a stream."""

    content: str = ""
    reasoning: str = ""
    done: bool = False
```

```python
# aixon/__init__.py  (append Message/Chunk/Role to imports and __all__)
"""aixon — declarative AI-agent framework."""

from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)
from aixon.message import Chunk, Message, Role

__all__ = [
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    "Message",
    "Chunk",
    "Role",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_message.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/message.py aixon/__init__.py tests/test_message.py
git commit -m "feat: neutral Message and Chunk types"
```

---

### Task 3: Agent registry

**Files:**
- Create: `aixon/registry.py`
- Modify: `tests/conftest.py` (real reset fixture)
- Test: `tests/test_registry.py`

**Interfaces:**
- Consumes: `aixon.exceptions.RegistrationError`, `aixon.exceptions.AgentNotFoundError`.
- Produces:
  - `aixon.registry.Registry` with methods:
    - `register(self, agent) -> None` — agent must expose `.name: str`, `.aliases: list[str]`, `.hidden: bool`. Raises `RegistrationError` on duplicate name or alias collision.
    - `resolve(self, name: str) -> object` — returns the agent for a name or alias; raises `AgentNotFoundError` if absent and the registry does not hold exactly one agent. If exactly one agent is registered, any `name` resolves to it (single-agent default, mirroring olympus).
    - `public(self) -> list` — registered agents with `hidden is False`, in registration order.
    - `all(self) -> list` — every registered agent, in registration order.
    - `clear(self) -> None` — wipe state (tests).
  - `aixon.registry.get_registry() -> Registry` — process-global singleton accessor.
  - `aixon.registry.reset_registry() -> None` — clears the global singleton (tests).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import pytest
from aixon.registry import Registry, get_registry, reset_registry
from aixon.exceptions import RegistrationError, AgentNotFoundError


class _FakeAgent:
    def __init__(self, name, aliases=None, hidden=False):
        self.name = name
        self.aliases = aliases or []
        self.hidden = hidden


def test_register_and_resolve_by_name():
    reg = Registry()
    a = _FakeAgent("alpha")
    reg.register(a)
    assert reg.resolve("alpha") is a


def test_resolve_by_alias():
    reg = Registry()
    a = _FakeAgent("alpha", aliases=["a1", "a2"])
    reg.register(a)
    assert reg.resolve("a2") is a


def test_duplicate_name_raises():
    reg = Registry()
    reg.register(_FakeAgent("alpha"))
    with pytest.raises(RegistrationError, match="alpha"):
        reg.register(_FakeAgent("alpha"))


def test_alias_collision_with_name_raises():
    reg = Registry()
    reg.register(_FakeAgent("alpha"))
    with pytest.raises(RegistrationError):
        reg.register(_FakeAgent("beta", aliases=["alpha"]))


def test_single_agent_is_default():
    reg = Registry()
    only = _FakeAgent("alpha")
    reg.register(only)
    assert reg.resolve("anything-else") is only


def test_unknown_name_with_multiple_agents_raises():
    reg = Registry()
    reg.register(_FakeAgent("alpha"))
    reg.register(_FakeAgent("beta"))
    with pytest.raises(AgentNotFoundError):
        reg.resolve("gamma")


def test_public_excludes_hidden():
    reg = Registry()
    visible = _FakeAgent("v")
    reg.register(visible)
    reg.register(_FakeAgent("h", hidden=True))
    assert reg.public() == [visible]


def test_global_singleton_is_stable_and_resettable():
    reset_registry()
    get_registry().register(_FakeAgent("alpha"))
    assert get_registry().resolve("alpha").name == "alpha"
    reset_registry()
    assert get_registry().all() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.registry'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/registry.py
"""Process-global registry of agents. Agents self-register on definition
(see ``Agent.__init_subclass__``); the server and CLI read this registry to
route requests and build menus."""

from __future__ import annotations

from typing import Optional

from aixon.exceptions import AgentNotFoundError, RegistrationError


class Registry:
    def __init__(self) -> None:
        self._agents: dict[str, object] = {}   # name -> agent
        self._aliases: dict[str, str] = {}      # alias -> name
        self._order: list[str] = []             # registration order of names

    def register(self, agent: object) -> None:
        name = agent.name
        if name in self._agents or name in self._aliases:
            raise RegistrationError(
                f"Agent name '{name}' is already registered. Names and aliases "
                f"must be unique across the registry."
            )
        for alias in agent.aliases:
            if alias in self._agents or alias in self._aliases:
                raise RegistrationError(
                    f"Alias '{alias}' (on agent '{name}') collides with an "
                    f"existing name or alias."
                )
        self._agents[name] = agent
        self._order.append(name)
        for alias in agent.aliases:
            self._aliases[alias] = name

    def resolve(self, name: str) -> object:
        if name in self._agents:
            return self._agents[name]
        if name in self._aliases:
            return self._agents[self._aliases[name]]
        if len(self._agents) == 1:
            return next(iter(self._agents.values()))
        raise AgentNotFoundError(
            f"No agent registered as '{name}'. "
            f"Known agents: {sorted(self._agents)}."
        )

    def public(self) -> list:
        return [self._agents[n] for n in self._order if not self._agents[n].hidden]

    def all(self) -> list:
        return [self._agents[n] for n in self._order]

    def clear(self) -> None:
        self._agents.clear()
        self._aliases.clear()
        self._order.clear()


_registry: Optional[Registry] = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = Registry()
```

```python
# tests/conftest.py  (replace the placeholder)
import pytest

from aixon.registry import reset_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_registry()
    yield
    reset_registry()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/registry.py tests/conftest.py tests/test_registry.py
git commit -m "feat: agent registry with alias resolution and single-agent default"
```

---

### Task 4: Agent base class — suffix validation, abstract markers, auto-registration, neutral interface

**Files:**
- Create: `aixon/agent.py`
- Modify: `aixon/__init__.py` (export `Agent`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `aixon.message.Message`, `aixon.message.Chunk`, `aixon.exceptions.NamingError`, `aixon.registry.get_registry`.
- Produces:
  - `aixon.agent.Agent` — abstract base. Class attributes (declarative):
    - `name: str = ""` (defaults to class name lowercased if blank, resolved at instantiation)
    - `description: str = ""`
    - `aliases: list[str] = []`
    - `hidden: bool = False`
    - `owned_by: str = "aixon"`
    - `_suffix: str = "Agent"` (overridable by abstract subtypes)
  - `Agent.__init_subclass__(cls, *, abstract: bool = False, **kwargs)` — when `abstract=True`, marks the class abstract and skips validation/registration; otherwise enforces `cls.__name__.endswith(cls._suffix)` (raises `NamingError`) and auto-instantiates the class (`cls()`), which registers it.
  - `Agent.__init__(self)` — resolves `self.name` (class attr or lowercased class name), then `get_registry().register(self)`.
  - Abstract methods: `invoke(self, messages: list[Message]) -> Message` and `stream(self, messages: list[Message]) -> Iterator[Chunk]`.
  - A concrete subclass that does not implement both abstract methods raises `TypeError` at instantiation (standard ABC behavior), surfaced as a `NamingError`-wrapped startup failure is **not** required — plain `TypeError` is acceptable and tested as such.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
import pytest
from typing import Iterator

from aixon.agent import Agent
from aixon.message import Message, Chunk
from aixon.exceptions import NamingError
from aixon.registry import get_registry


def _make_concrete(name_cls="EchoAgent", **attrs):
    """Define a concrete Agent subclass at call time (so suffix errors raise here)."""
    body = {
        "invoke": lambda self, messages: Message(role="assistant", content="ok"),
        "stream": lambda self, messages: iter([Chunk(content="ok", done=True)]),
        **attrs,
    }
    return type(name_cls, (Agent,), body)


def test_concrete_subclass_registers_itself():
    _make_concrete("EchoAgent")
    agent = get_registry().resolve("echoagent")
    assert agent.invoke([]).content == "ok"


def test_explicit_name_attribute_wins():
    _make_concrete("EchoAgent", name="echo")
    assert get_registry().resolve("echo").name == "echo"


def test_bad_suffix_raises_naming_error():
    with pytest.raises(NamingError, match="Agent"):
        _make_concrete("Echo")  # missing 'Agent' suffix


def test_abstract_subtype_is_exempt_and_unregistered():
    # Simulate how LLMAgent/ToolAgent will be declared in later plans.
    class FakeSubtype(Agent, abstract=True):
        _suffix = "Agent"

    assert get_registry().all() == []
    # A concrete subclass of the abstract subtype validates against _suffix.
    type(
        "GreeterAgent",
        (FakeSubtype,),
        {
            "invoke": lambda self, m: Message(role="assistant"),
            "stream": lambda self, m: iter([Chunk(done=True)]),
        },
    )
    assert get_registry().resolve("greeteragent").name == "greeteragent"


def test_custom_suffix_on_abstract_subtype():
    class FakeOrchestrator(Agent, abstract=True):
        _suffix = "Orchestrator"

    with pytest.raises(NamingError, match="Orchestrator"):
        type(
            "BadName",
            (FakeOrchestrator,),
            {
                "invoke": lambda self, m: Message(role="assistant"),
                "stream": lambda self, m: iter([Chunk(done=True)]),
            },
        )


def test_missing_abstract_methods_raise_type_error():
    with pytest.raises(TypeError):
        type("IncompleteAgent", (Agent,), {})  # no invoke/stream -> ABC error on cls()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.agent'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/agent.py
"""The Agent base class. Every executable unit in aixon is an Agent and speaks
only neutral types (``Message[]`` in, ``Message``/``Chunk`` out). Concrete
subclasses self-register at definition time; abstract subtypes
(``LLMAgent``/``ToolAgent``/``Orchestrator``, defined in later plans) pass
``abstract=True`` to opt out of validation and registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from aixon.exceptions import NamingError
from aixon.message import Chunk, Message
from aixon.registry import get_registry


class Agent(ABC):
    # Declarative metadata (override in subclasses).
    name: str = ""
    description: str = ""
    aliases: list[str] = []
    hidden: bool = False
    owned_by: str = "aixon"

    # Required class-name suffix; abstract subtypes may override (e.g. "Orchestrator").
    _suffix: str = "Agent"
    # Set True on a class to mark it an abstract subtype (no validation/registration).
    _abstract: bool = True  # the base itself is abstract

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if abstract:
            cls._abstract = True
            return
        cls._abstract = False
        if not cls.__name__.endswith(cls._suffix):
            raise NamingError(
                f"Agent subclass '{cls.__name__}' must end with '{cls._suffix}' "
                f"(rename to '{cls.__name__}{cls._suffix}')."
            )
        # Auto-instantiate: running __init__ registers the agent.
        cls()

    def __init__(self) -> None:
        if not self.name:
            self.name = type(self).__name__.lower()
        get_registry().register(self)

    @abstractmethod
    def invoke(self, messages: list[Message]) -> Message:
        """Run the agent to completion and return one neutral Message."""

    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Run the agent, yielding neutral Chunks as they are produced."""
```

```python
# aixon/__init__.py  (add Agent)
from aixon.agent import Agent
from aixon.exceptions import (
    AixonError,
    AgentNotFoundError,
    CompositionCycleError,
    NamingError,
    RegistrationError,
)
from aixon.message import Chunk, Message, Role

__all__ = [
    "Agent",
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    "Message",
    "Chunk",
    "Role",
]
```

> Note on instantiation order: `__init_subclass__` calls `cls()` while the class object exists but its module-level name may not be bound yet — this matches restmcp's Endpoint pattern and is fine because registration only needs the class, not the module binding.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tests from Tasks 1–4).

- [ ] **Step 6: Commit**

```bash
git add aixon/agent.py aixon/__init__.py tests/test_agent.py
git commit -m "feat: Agent base with suffix validation and auto-registration"
```

---

### Task 5: Agent.as_tool() — neutral tool descriptor

**Files:**
- Modify: `aixon/agent.py` (add `as_tool` + `AgentTool`)
- Modify: `aixon/__init__.py` (export `AgentTool`)
- Test: `tests/test_as_tool.py`

**Interfaces:**
- Consumes: `aixon.message.Message`.
- Produces:
  - `aixon.agent.AgentTool` — dataclass: `name: str`, `description: str`, `func: Callable[[str], str]`. (Neutral; Plan 3 adapts it to a LangChain `StructuredTool`.)
  - `Agent.as_tool(self, name: str | None = None, description: str | None = None) -> AgentTool` — wraps `invoke`: the produced `func(text: str)` calls `self.invoke([Message(role="user", content=text)])` and returns the result `.content`. Each call runs with a fresh message list (state isolation). Defaults: `name = self.name`, `description = self.description`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_as_tool.py
from aixon.agent import Agent, AgentTool
from aixon.message import Message, Chunk


def _concrete(name_cls, reply, **attrs):
    return type(
        name_cls,
        (Agent,),
        {
            "invoke": lambda self, messages: Message(
                role="assistant", content=reply + ":" + messages[-1].content
            ),
            "stream": lambda self, m: iter([Chunk(done=True)]),
            **attrs,
        },
    )


def test_as_tool_returns_descriptor_with_defaults():
    _concrete("HelperAgent", "h", description="a helper")
    from aixon.registry import get_registry

    agent = get_registry().resolve("helperagent")
    tool = agent.as_tool()
    assert isinstance(tool, AgentTool)
    assert tool.name == "helperagent"
    assert tool.description == "a helper"


def test_as_tool_func_invokes_agent_with_user_message():
    _concrete("HelperAgent", "h")
    from aixon.registry import get_registry

    tool = get_registry().resolve("helperagent").as_tool()
    assert tool.func("ping") == "h:ping"


def test_as_tool_overrides():
    _concrete("HelperAgent", "h")
    from aixon.registry import get_registry

    tool = get_registry().resolve("helperagent").as_tool(
        name="custom", description="custom desc"
    )
    assert tool.name == "custom"
    assert tool.description == "custom desc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_as_tool.py -v`
Expected: FAIL with `ImportError: cannot import name 'AgentTool' from 'aixon.agent'`.

- [ ] **Step 3: Write the implementation**

In `aixon/agent.py`, adjust the imports. The file already has `from typing import Iterator` (from Task 4) and `from abc import ABC, abstractmethod`. Add `dataclass` and `Callable` WITHOUT duplicating the existing `Iterator` import — change the two import lines to exactly:

```python
# aixon/agent.py  — imports (replace the existing dataclass-less / Callable-less lines)
from dataclasses import dataclass
from typing import Callable, Iterator
```

(If a `from typing import Iterator` line already exists, replace it with the `Callable, Iterator` line above rather than adding a second import line.)

```python
# aixon/agent.py  — add above the Agent class
@dataclass
class AgentTool:
    """Neutral descriptor of an Agent exposed as a callable tool. Later plans
    adapt this to a LangChain StructuredTool for tool-calling agents."""

    name: str
    description: str
    func: Callable[[str], str]
```

```python
# aixon/agent.py  — add as a method on Agent
    def as_tool(
        self, name: str | None = None, description: str | None = None
    ) -> "AgentTool":
        """Expose this agent as a tool. Each call runs with a fresh message
        list, so the wrapped agent's state never leaks across invocations."""

        def _run(text: str) -> str:
            result = self.invoke([Message(role="user", content=text)])
            return result.content

        return AgentTool(
            name=name or self.name,
            description=description or self.description,
            func=_run,
        )
```

```python
# aixon/__init__.py  — add AgentTool
from aixon.agent import Agent, AgentTool
# ... add "AgentTool" to __all__
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_as_tool.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aixon/agent.py aixon/__init__.py tests/test_as_tool.py
git commit -m "feat: Agent.as_tool neutral tool descriptor"
```

---

### Task 6: autodiscover()

**Files:**
- Create: `aixon/discovery.py`
- Modify: `aixon/__init__.py` (export `autodiscover`)
- Test: `tests/test_discovery.py`

**Interfaces:**
- Consumes: nothing from earlier tasks at runtime (it triggers `Agent.__init_subclass__` indirectly via import).
- Produces:
  - `aixon.discovery.autodiscover(package: str) -> None` — imports every non-underscore module in `package`, triggering agent registration. Raises `ValueError` if `package` is not a package (no `__path__`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_discovery.py
import sys
import types
import pytest

from aixon.discovery import autodiscover
from aixon.registry import get_registry


def _make_pkg(monkeypatch, tmp_path, name="demo_agents"):
    """Create a temp package with one agent module and one underscore module
    that must be skipped. Evicts any cached copy of the package from
    sys.modules first, so a re-import picks up THIS tmp_path (not a stale one
    from an earlier test)."""
    for mod in list(sys.modules):
        if mod == name or mod.startswith(name + "."):
            del sys.modules[mod]
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "alpha.py").write_text(
        "from aixon.agent import Agent\n"
        "from aixon.message import Message, Chunk\n"
        "class AlphaAgent(Agent):\n"
        "    def invoke(self, messages): return Message(role='assistant')\n"
        "    def stream(self, messages): return iter([Chunk(done=True)])\n"
    )
    (pkg / "_skip.py").write_text("raise RuntimeError('should not be imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


def test_autodiscover_imports_modules_and_registers(monkeypatch, tmp_path):
    name = _make_pkg(monkeypatch, tmp_path)
    autodiscover(name)
    assert get_registry().resolve("alphaagent").name == "alphaagent"


def test_autodiscover_skips_underscore_modules(monkeypatch, tmp_path):
    name = _make_pkg(monkeypatch, tmp_path)
    autodiscover(name)  # must not raise from _skip.py


def test_autodiscover_rejects_non_package():
    mod = types.ModuleType("not_a_pkg")
    sys.modules["not_a_pkg"] = mod
    try:
        with pytest.raises(ValueError, match="not a package"):
            autodiscover("not_a_pkg")
    finally:
        del sys.modules["not_a_pkg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.discovery'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/discovery.py
"""Import every module in a consumer package so its agents register. Agents
self-register when their class body runs (``Agent.__init_subclass__``);
importing the module is what triggers that. Drop a new ``*.py`` into the
package and it goes live on the next start, with no list to maintain.
Modules whose name starts with ``_`` are skipped."""

from __future__ import annotations

import importlib
import pkgutil


def autodiscover(package: str) -> None:
    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        raise ValueError(f"{package!r} is not a package (has no __path__).")
    for module in pkgutil.iter_modules(pkg.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package}.{module.name}")
```

Now overwrite `aixon/__init__.py` with its complete contents (this supersedes the partial edits from Tasks 2, 4, and 5 — write the whole file exactly as below so the public API is unambiguous). Task 7 adds `Logger` to it afterward:

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
from aixon.message import Chunk, Message, Role

__all__ = [
    "Agent",
    "AgentTool",
    "autodiscover",
    "AixonError",
    "AgentNotFoundError",
    "CompositionCycleError",
    "NamingError",
    "RegistrationError",
    "Message",
    "Chunk",
    "Role",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_discovery.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all foundation tests).

- [ ] **Step 6: Commit**

```bash
git add aixon/discovery.py aixon/__init__.py tests/test_discovery.py
git commit -m "feat: autodiscover for self-registering agents"
```

---

### Task 7: Logging & startup visibility

**Files:**
- Create: `aixon/logging.py`
- Modify: `aixon/registry.py` (log on register)
- Modify: `aixon/discovery.py` (log on autodiscover)
- Modify: `aixon/__init__.py` (export `Logger` — final version of the file)
- Test: `tests/test_logging.py`

**Interfaces:**
- Consumes: `aixon.registry.Registry.register`, `aixon.discovery.autodiscover` (both already built).
- Produces:
  - `aixon.logging.Logger(name: str)` — thin wrapper over stdlib `logging`. Reads `LOG_LEVEL` env (default `INFO`), attaches a single `StreamHandler` (guards against duplicate handlers), formats as `[time] LEVEL name — message`. Methods: `info`, `warning`, `error`, `debug` (each `(msg, *args, **kwargs)`).
  - `Logger` re-exported from `aixon`.
  - Side effect: registering an agent logs `INFO` `"registered agent '<name>'..."`; `autodiscover(pkg)` logs `INFO` start/finish so the terminal shows what comes up.

> **Note on scope:** this task logs *framework lifecycle* events. Logging the agent's actual generated content/reasoning at runtime belongs to `LLMAgent`/`ToolAgent` (Plans 2–3), where `invoke`/`stream` are implemented; here those methods are abstract. Streaming reasoning/content to the terminal for a human is the `aixon chat` CLI's job (Plan 7).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging.py
import logging
import pytest

from aixon.logging import Logger


def test_logger_respects_log_level_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    log = Logger("aixon.test.level")
    assert log._logger.level == logging.DEBUG


def test_logger_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    log = Logger("aixon.test.default")
    assert log._logger.level == logging.INFO


def test_logger_does_not_duplicate_handlers():
    Logger("aixon.test.dup")
    Logger("aixon.test.dup")
    assert len(logging.getLogger("aixon.test.dup").handlers) == 1


def test_registering_agent_logs_info(caplog):
    from aixon.agent import Agent
    from aixon.message import Message, Chunk

    with caplog.at_level(logging.INFO, logger="aixon.registry"):
        type(
            "LoggedAgent",
            (Agent,),
            {
                "invoke": lambda self, m: Message(role="assistant"),
                "stream": lambda self, m: iter([Chunk(done=True)]),
            },
        )
    assert any("loggedagent" in r.message for r in caplog.records)


def test_autodiscover_logs(monkeypatch, tmp_path, caplog):
    import sys

    name = "logpkg_agents"
    for mod in list(sys.modules):
        if mod == name or mod.startswith(name + "."):
            del sys.modules[mod]
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "alpha.py").write_text(
        "from aixon.agent import Agent\n"
        "from aixon.message import Message, Chunk\n"
        "class AlphaAgent(Agent):\n"
        "    def invoke(self, messages): return Message(role='assistant')\n"
        "    def stream(self, messages): return iter([Chunk(done=True)])\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    from aixon.discovery import autodiscover

    with caplog.at_level(logging.INFO, logger="aixon.discovery"):
        autodiscover(name)
    assert any(name in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aixon.logging'`.

- [ ] **Step 3: Write the implementation**

```python
# aixon/logging.py
"""Thin wrapper over Python's stdlib logging. Level is configurable via the
LOG_LEVEL env var (default INFO). Use it for framework/consumer diagnostics —
NOT for streaming an agent's generated content (that is the reasoning channel
and the CLI)."""

import logging
import os


class Logger:
    def __init__(self, name: str):
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)
```

In `aixon/registry.py`, add the import at the top and a log line at the end of `register`:

```python
# aixon/registry.py — add near the other imports
from aixon.logging import Logger

_log = Logger("aixon.registry")
```

```python
# aixon/registry.py — at the END of Registry.register(), after the aliases loop
        hidden = " (hidden)" if agent.hidden else ""
        _log.info(f"registered agent '{name}'{hidden} aliases={agent.aliases}")
```

In `aixon/discovery.py`, add logging around the import loop:

```python
# aixon/discovery.py — add near the other imports
from aixon.logging import Logger

_log = Logger("aixon.discovery")
```

```python
# aixon/discovery.py — replace the body of autodiscover with the logged version
def autodiscover(package: str) -> None:
    pkg = importlib.import_module(package)
    if not hasattr(pkg, "__path__"):
        raise ValueError(f"{package!r} is not a package (has no __path__).")
    _log.info(f"autodiscover: scanning package '{package}'")
    count = 0
    for module in pkgutil.iter_modules(pkg.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{package}.{module.name}")
            count += 1
    _log.info(f"autodiscover: imported {count} module(s) from '{package}'")
```

Now overwrite `aixon/__init__.py` with its final contents (adds `Logger`):

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

__all__ = [
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
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_logging.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all foundation tests across Tasks 1–7).

- [ ] **Step 6: Commit**

```bash
git add aixon/logging.py aixon/registry.py aixon/discovery.py aixon/__init__.py tests/test_logging.py
git commit -m "feat: Logger and lifecycle logging (registration, autodiscover)"
```

---

## Self-Review

**Spec coverage (foundation slice only):**
- Package layout / `pyproject.toml` (hatch, `aixon` name, `[project.scripts]`) → Task 1. ✓
- Suffix validation in `__init_subclass__`, fails before server is usable → Task 4, with abstract-subtype exemption for `LLMAgent`/`ToolAgent` (`*Agent`) and `Orchestrator` (`*Orchestrator`). ✓
- Neutral types (`Message[]` in, `Message`/`Chunk` out), no protocol/provider leakage into the runtime → Task 2 + Global Constraints. ✓
- Registry with name + alias resolution and single-agent default (olympus `_resolve_chat_agent` behavior) → Task 3. ✓
- `Agent` as the single executable/composable unit, uniform `invoke`/`stream`/`as_tool` interface → Tasks 4–5. ✓
- `autodiscover()` → Task 6. ✓
- Terminal logging (`Logger`, `LOG_LEVEL`, lifecycle log lines) → Task 7. ✓ Streaming the agent's own content/reasoning to a human is intentionally NOT here — it is the `reasoning` channel (Plan 3) + `aixon chat` CLI (Plan 7).
- **Deferred to later plans (correctly out of this slice):** `LLM`/providers (Plan 2), `LLMAgent`/`ToolAgent` (Plans 2–3), reasoning channel (Plan 3), `Orchestrator` 3 tiers + `GraphState` + recursion guards (Plan 4), `Server`/`ProtocolAdapter`/`OpenAIAdapter`/`AnthropicAdapter`/auth (Plan 5), `Retriever`/`Embedding`/`Connector` (Plan 6), real CLI (Plan 7), docs (Plan 8). The composition-cycle guard (`CompositionCycleError` is defined here but enforced in Plan 4, where agents first reference each other as tools).

**Placeholder scan:** No TBD/TODO/"add error handling"/"similar to Task N" left; every code step is complete. ✓

**Type consistency:** `Message`/`Chunk` field names and `to_dict()` are used identically across Tasks 2, 4, 5. `Agent` attributes (`name`, `aliases`, `hidden`, `_suffix`, `_abstract`) match between Task 4 and the `Registry` contract in Task 3 (`.name`/`.aliases`/`.hidden`). `AgentTool(name, description, func)` is consistent between Task 5's interface block and implementation. `get_registry`/`reset_registry`/`Registry` names match across Tasks 3–6. `Logger(name)` and its `info/warning/error/debug` methods match between Task 7's interface and the registry/discovery call sites. No import cycle: `logging` → stdlib only; `registry`/`discovery` → `logging`; `agent` → `registry`. ✓
