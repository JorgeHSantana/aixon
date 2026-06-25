# aixon — Cross-Plan Interface Contract

**Purpose:** This document pins the exact public APIs, file layout, and design
decisions that span Plans 2–8, so each plan's implementation aligns with its
neighbors. **Plan writers and implementers MUST treat the signatures here as
binding.** Where a plan needs an internal helper not listed here, it may add
it, but anything another plan consumes must match this contract verbatim.

The design spec is `docs/superpowers/specs/2026-06-23-aixon-framework-design.md`.
Plan 1 (foundation) is already built and merged on branch `feat/foundation`.

---

## 0. Already built (Plan 1 — do not re-implement)

Public API from `aixon` (`aixon/__init__.py`):

```python
# Neutral types (aixon/message.py)
Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str = ""
    name: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    reasoning: Optional[str] = None
    def to_dict(self) -> dict[str, Any]: ...   # omits empty optional fields

@dataclass
class Chunk:
    content: str = ""
    reasoning: str = ""
    done: bool = False

# Agent base (aixon/agent.py)
class Agent(ABC):
    name: str = ""            # blank -> lowercased class name at __init__
    description: str = ""
    aliases: list[str] = []
    hidden: bool = False
    owned_by: str = "aixon"
    _suffix: str = "Agent"    # abstract subtypes may override (e.g. "Orchestrator")
    _abstract: bool = True
    _registered: bool = False
    def __init_subclass__(cls, *, abstract: bool = False, **kwargs): ...
    @classmethod
    def _validate_subclass(cls) -> None: ...  # hook: subtypes validate a concrete
                                              # subclass BEFORE registration (no-op by default)
    def __init__(self) -> None: ...          # resolves name, registers once
    @abstractmethod
    def invoke(self, messages: list[Message]) -> Message: ...
    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[Chunk]: ...
    def as_tool(self, name=None, description=None) -> AgentTool: ...

@dataclass
class AgentTool:
    name: str
    description: str
    func: Callable[[str], str]

# Registry (aixon/registry.py)
class Registry:
    def register(self, agent) -> None        # raises RegistrationError on dup
    def resolve(self, name: str) -> object   # name|alias; single-agent default; else AgentNotFoundError
    def public(self) -> list                 # hidden is False, registration order
    def all(self) -> list
    def clear(self) -> None
def get_registry() -> Registry: ...
def reset_registry() -> None: ...

# Discovery (aixon/discovery.py)
def autodiscover(package: str) -> None: ...   # imports non-underscore modules

# Logging (aixon/logging.py)
class Logger:
    def __init__(self, name: str): ...        # LOG_LEVEL env, default INFO
    def info/warning/error/debug(self, msg, *args, **kwargs): ...

# Exceptions (aixon/exceptions.py)
AixonError(Exception)            # .message attribute
NamingError, RegistrationError, AgentNotFoundError, CompositionCycleError  # all subclass AixonError
```

### Established conventions (every plan follows these)

- **Python 3.11+**, build backend `hatchling`, package name `aixon`.
- **Abstract subtype pattern:** intermediate base classes (`LLMAgent`,
  `ToolAgent`, `Orchestrator`) are declared `class X(Agent, abstract=True)`.
  Concrete user subclasses inherit and get suffix-validated + auto-registered.
  `LLMAgent`/`ToolAgent` keep `_suffix = "Agent"`; `Orchestrator` sets
  `_suffix = "Orchestrator"`.
- **ABCMeta timing workaround** (already in `agent.py`): abstract-method
  detection happens manually in `__init_subclass__` before `cls()`. New
  abstract subtypes that don't implement `invoke`/`stream` (because the
  subtype itself provides them) just implement them — they are concrete at
  the subtype level, so no issue.
- **Subtype validation hook** (already in `agent.py`): abstract subtypes that
  require a declared attribute on concrete subclasses (e.g. `LLMAgent`/`ToolAgent`
  need `llm`) MUST override the `Agent._validate_subclass()` classmethod, NOT
  `__init_subclass__`. `Agent.__init_subclass__` calls `cls._validate_subclass()`
  after suffix/abstract-method checks and **before** `cls()` (registration), so a
  failed validation raises `AixonError` without leaving a ghost in the registry.
  `NamingError` (bad suffix) still takes precedence because the suffix is checked
  first. Do NOT re-add an `__init_subclass__` override that validates after
  `super().__init_subclass__()` — that re-introduces the register-then-validate
  ghost bug.
- **Neutral boundary:** `Agent.invoke`/`stream` and the public API speak ONLY
  `Message`/`Chunk`. LangChain/LangGraph/provider objects may be used
  INTERNALLY but must be converted at the boundary. Conversion helpers live in
  the private `aixon/_interop/` package (see §1.4).
- **Dependencies:** core `dependencies = ["langchain>=1.0", "langchain-core>=1.0",
  "langgraph>=1.0"]` — the langchain stack is MANDATORY (every agent subtype and
  the orchestrator need it). Optional layers live in extras: `dev`, `server`,
  `cli`, `openai`/`anthropic`/`google` (provider bindings), `retrieval`,
  `openai-embedding`, `all`. There is NO `llm` extra (see §9.2/§9.4).
- **Tests:** pytest, `tests/conftest.py` has an autouse `reset_registry`
  fixture. NO `tests/__init__.py`. Tests must NOT require real API keys or
  network — use the fakes defined in §1.5 and §3.4.
- **Logging:** lifecycle/diagnostic events use `Logger("aixon.<area>")`.
  Streaming the agent's own content/reasoning to a human is the CLI's job, not
  logging.
- **Error tone:** state what was got and how to fix it.
- **Commits:** Co-Authored-By trailer per the repo convention.
- **The server is FastAPI/ASGI (NOT Flask).** olympus used Flask; aixon
  deliberately uses FastAPI + uvicorn, mirroring restmcp.

---

## 1. Plan 2 — LLM + providers + LLMAgent

### 1.1 `aixon/providers/base.py`

```python
class Provider(ABC):
    """Builds a LangChain BaseChatModel for one vendor. Reads the API key from
    the environment. Concrete providers live in aixon/providers/<vendor>.py."""
    name: str        # "openai" | "anthropic" | "google"
    env_key: str     # e.g. "OPENAI_API_KEY"

    @abstractmethod
    def build(self, model: str, **params) -> "BaseChatModel":
        """Return a configured LangChain chat model. **params are passed
        through (temperature, max_tokens, top_p, etc.)."""

# Provider registry
def register_provider(provider: Provider) -> None: ...   # keyed by provider.name
def get_provider(name: str) -> Provider: ...             # raises AixonError if absent
def resolve_provider_for_model(model: str) -> Provider:
    """Infer provider from model name: gpt*/o[0-9]*/text-* -> openai;
    claude* -> anthropic; gemini* -> google. Raises AixonError if no match."""
```

### 1.2 `aixon/providers/openai.py`, `anthropic.py`, `google.py`

Each defines a concrete `Provider` subclass and registers it at import time:
- `OpenAIProvider(name="openai", env_key="OPENAI_API_KEY")` → `langchain_openai.ChatOpenAI`
- `AnthropicProvider(name="anthropic", env_key="ANTHROPIC_API_KEY")` → `langchain_anthropic.ChatAnthropic`
- `GoogleProvider(name="google", env_key="GOOGLE_API_KEY")` → `langchain_google_genai.ChatGoogleGenerativeAI`

Provider SDK imports are **lazy** (inside `build`), so importing `aixon` never
requires every vendor SDK to be installed.

### 1.3 `aixon/llm.py`

```python
class LLM:
    """Declarative handle for a chat model. Stored as a class attribute on
    agents: `llm = LLM("gpt-4o-mini", temperature=0.2)`. Resolves to a LangChain
    chat model lazily (so construction needs no API key / network)."""
    def __init__(self, model: str, *, provider: str | None = None, **params):
        self.model = model
        self.params = params
        self._provider_name = provider     # None -> inferred from model
        self._chat_model = None            # lazy

    @property
    def chat_model(self) -> "BaseChatModel":
        """Lazily build & cache the LangChain model. Used by ToolAgent and
        Orchestrator. Resolves provider via explicit name or inference."""

    def complete(self, messages: list[Message]) -> Message:
        """Neutral single-shot completion (used by LLMAgent.invoke)."""

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Neutral streaming (used by LLMAgent.stream). Yields Chunk(content=...)
        deltas and a final Chunk(done=True)."""
```

### 1.4 `aixon/_interop/messages.py` (internal conversion helpers)

Import from the exact submodule: `from aixon._interop.messages import
to_langchain, from_langchain`. The `aixon/_interop/__init__.py` deliberately
does NOT re-export these (an eager re-export would import `langchain_core` at
`import aixon` time and break the neutral boundary).

```python
def to_langchain(messages: list[Message]) -> list["BaseMessage"]:
    """Neutral Message[] -> LangChain message objects (SystemMessage/
    HumanMessage/AIMessage/ToolMessage)."""

def from_langchain(msg: "BaseMessage") -> Message:
    """LangChain AIMessage -> neutral Message (carries .content; tool_calls if
    present; reasoning if the provider returned it)."""
```

### 1.5 `aixon/agents/llm_agent.py`

```python
class LLMAgent(Agent, abstract=True):
    _suffix = "Agent"
    llm: LLM            # REQUIRED declarative attribute (validated in _validate_subclass)
    prompt: str = ""    # optional system prompt prepended to messages

    @classmethod
    def _validate_subclass(cls) -> None:
        """Require a concrete subclass to declare an `llm: LLM`; raise AixonError if absent."""

    def invoke(self, messages: list[Message]) -> Message:
        """Prepend system prompt (if any) and delegate to self.llm.complete."""

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Delegate to self.llm.stream."""
```

`LLMAgent` validates the required `llm` by overriding `Agent._validate_subclass()`
(the hook the base calls before registration) — NOT by overriding
`__init_subclass__`. A concrete subclass missing a valid `llm: LLM` raises
`AixonError`; because the hook runs before `cls()`, no ghost is registered.
`abstract=True` subtypes are skipped automatically (the hook only fires for
concrete subclasses). See "Subtype validation hook" under Established conventions.

### 1.6 Testing without API keys (Plan 2 MUST establish this)

Define a `FakeProvider` in the test suite (or a tiny test helper module) that
returns a fake LangChain-compatible chat model echoing a canned response, and
register it under a test name (e.g. provider `"fake"`, model `"fake-1"`).
Tests for `LLM.complete`/`stream` and `LLMAgent` use `LLM("fake-1",
provider="fake")`. NO test may require a real provider SDK or network. If a
provider SDK isn't installed, `resolve_provider_for_model` / `build` tests for
that vendor should be skipped via `pytest.importorskip`.

### 1.7 `pyproject.toml` + exports

- Core deps: `langchain>=1.0`, `langchain-core>=1.0`, `langgraph>=1.0` (mandatory,
  not an extra — see §9.2).
  **langgraph-native (LangChain 1.x):** the old 0.x `create_tool_calling_agent`
  + `AgentExecutor` are removed; the ToolAgent (Plan 3) uses
  `from langchain.agents import create_agent` (validated against
  langchain 1.3 / langchain-core 1.4 / langgraph 1.2). `create_react_agent`
  from `langgraph.prebuilt` is DEPRECATED in langgraph 1.0 — use
  `langchain.agents.create_agent` instead. Provider SDKs
  go in vendor extras: `openai = ["langchain-openai>=0.2"]`,
  `anthropic = ["langchain-anthropic>=0.2"]`,
  `google = ["langchain-google-genai>=2.0"]`. Add these to `all`.
- Export from `aixon`: `LLM`, `Provider`, `register_provider`, `get_provider`,
  `LLMAgent`. (Keep `__all__` sorted into logical groups.)

---

## 2. Plan 3 — ToolAgent + reasoning channel

### 2.1 `aixon/reasoning.py`

```python
class ReasoningChannel:
    """Collects reasoning text emitted during an agent run and makes it
    available to the streaming layer. Propagates across nested agents."""

# A contextvars-based current channel (works with threads AND async):
def current_channel() -> ReasoningChannel | None: ...
def emit_reasoning(text: str) -> None:
    """Push a reasoning line to the current channel if one is active; no-op
    otherwise. Nested agents call this so their reasoning bubbles up to the
    parent's stream."""

@contextmanager
def reasoning_channel() -> Iterator[ReasoningChannel]:
    """Activate a channel for the duration of a stream(). Drained by the
    streaming loop into Chunk(reasoning=...)."""
```

Use `contextvars.ContextVar` (NOT thread-local) so it composes with both
sync and async execution and with LangGraph.

### 2.2 `aixon/agents/tool_agent.py`

```python
class ToolAgent(Agent, abstract=True):
    _suffix = "Agent"
    llm: LLM                     # REQUIRED (validated in _validate_subclass)
    prompt: str = ""             # system prompt
    tools: list = []             # @tool funcs | AgentTool | Retriever.as_tool() | other Agent.as_tool()
    max_iterations: int = 15
    max_execution_time: int = 600

    @classmethod
    def _validate_subclass(cls) -> None:
        """Require a concrete subclass to declare an `llm: LLM`; raise AixonError
        if absent. Override the hook — do NOT override __init_subclass__ (see
        "Subtype validation hook" in Established conventions; validating after
        super().__init_subclass__() would register a ghost before failing)."""

    def invoke(self, messages: list[Message]) -> Message:
        """Build a langgraph agent with `langchain.agents.create_agent(
        self.llm.chat_model, coerce_tools(self.tools), system_prompt=self.prompt)`
        and invoke it with the neutral messages converted via to_langchain.
        Convert the final AIMessage back with from_langchain. Any reasoning
        collected via the ReasoningChannel is set on Message.reasoning."""

    def stream(self, messages: list[Message]) -> Iterator[Chunk]:
        """Stream the compiled agent (graph.stream(..., stream_mode="messages")
        or "updates"): yield Chunk(reasoning=...) for tool-call/step labels (via
        the reasoning channel) and Chunk(content=...) for output deltas; final
        Chunk(done=True)."""
```

**Validated agent construction (langchain 1.x / langgraph 1.x — use exactly
this API, NOT the removed `AgentExecutor`):**

```python
from langchain.agents import create_agent          # NOT langgraph.prebuilt.create_react_agent (deprecated in lg 1.0)
agent = create_agent(self.llm.chat_model, coerce_tools(self.tools),
                     system_prompt=self.prompt or None)
result = agent.invoke({"messages": to_langchain(messages)})
final = from_langchain(result["messages"][-1])     # result["messages"] is [Human, AI(tool_calls), Tool, AI(final)]
```
`max_iterations`/`max_execution_time` map to langgraph's recursion/time
config where supported; if `create_agent` exposes no direct knob, document the
mapping and pass what it accepts (do not invent a parameter).

### 2.3 Tool coercion

A helper `aixon/_interop/tools.py::coerce_tools(tools: list) -> list[BaseTool]`
(import via `from aixon._interop.tools import coerce_tools`) that converts each
entry to a LangChain `BaseTool`:
- `AgentTool` (from `Agent.as_tool()` / `Retriever.as_tool()`) →
  `StructuredTool.from_function(func=tool.func, name=tool.name, description=tool.description)`
- a LangChain `BaseTool` / `@tool`-decorated function → passed through
- a plain callable → wrapped via `StructuredTool.from_function`

When a tool wraps a nested Agent, the nested agent's reasoning reaches the
parent stream automatically because `emit_reasoning` targets the active
`ReasoningChannel` (set by the outermost `stream()`).

### 2.4 `aixon/agent.py` change (one edit, owned by Plan 3)

`Agent.as_tool()` currently builds a neutral `AgentTool`. Keep it neutral — do
NOT make `as_tool` return a LangChain tool. Tool→LangChain conversion happens
only in `coerce_tools` (§2.3). This preserves the neutral boundary.

### 2.5 Tests

Use `LLM("fake-1", provider="fake")` with the shared `FakeChatModel` from
`tests/_fakes.py` (contract §9.1), scripting it to emit an `AIMessage` with
`tool_calls` then a final `AIMessage` — this drives `langchain.agents.create_agent`
through a tool call then a final answer with no network (validated). Do NOT
reference `AgentExecutor` (removed in langchain 1.x). Test that nested-agent
reasoning propagates via the reasoning channel.

### 2.6 Exports

Export `ToolAgent`, `emit_reasoning`, `reasoning_channel` from `aixon`.

---

## 3. Plan 4 — Orchestrator (3 tiers) + GraphState + recursion guards

### 3.1 `aixon/state.py`

```python
class GraphState(TypedDict, total=False):
    """Default LangGraph state. Carries the neutral conversation + reasoning.
    Users subclass to add fields (declared as `class State(GraphState): ...`
    inside their Orchestrator)."""
    messages: Annotated[list[Message], add_messages_neutral]
    reasoning: list[str]
```

(Provide `add_messages_neutral` reducer that appends neutral Messages.)

### 3.2 `aixon/agents/orchestrator.py`

```python
class Orchestrator(Agent, abstract=True):
    _suffix = "Orchestrator"

    # Tier 1 (supervisor):
    supervisor: LLM | None = None
    agents: list = []            # any Agent subclasses/instances (workers)

    # Tier 2 (explicit graph):
    nodes: dict = {}             # name -> Agent
    entry: str = ""
    edges: list = []             # list of (src, dst) fixed edges; dst may be END
    # conditional edges: methods named route_<node>(self, state) -> str | list[str]

    # Tier 3 (escape hatch): override build_graph(self) -> compiled graph

    # Runtime guards:
    recursion_limit: int | None = 25    # None = no cap (still bounded by timeout)
    timeout: int | None = None          # wall-clock seconds

    # Optional declarative state:
    # class State(GraphState): ...

    def invoke(self, messages: list[Message]) -> Message: ...
    def stream(self, messages: list[Message]) -> Iterator[Chunk]: ...
```

**Tier detection order** (in `__init_subclass__` or at first build):
`build_graph` overridden → Tier 3; else `nodes` non-empty → Tier 2; else
`supervisor` set → Tier 1. If none apply on a concrete subclass → `AixonError`.

**Tier 2 rules** (validate in `__init_subclass__`, raise `AixonError`):
- each node has exactly ONE exit form: a fixed edge in `edges` OR a
  `route_<node>` method — declaring both for the same node is an error; neither
  makes it terminal.
- `route_<node>` returning a `str` = conditional (one next node);
  returning a `list[str]` = parallel fan-out.
- `entry` must name a node in `nodes`.
- `END` sentinel is `aixon.state.END` (re-export langgraph's END).

**Recursion guards (two distinct kinds):**
- **(A) Composition cycle (structural), always on:** in `__init_subclass__`,
  walk the composition graph of nested agents (agents referenced via `agents`,
  `nodes`, or as `as_tool` tools). Revisiting a class already on the current
  path → raise `CompositionCycleError`. A cycle *within* a LangGraph graph (a
  node that loops back) is legitimate and allowed — guard B bounds it.
- **(B) Runtime depth/loop:** pass `recursion_limit` to LangGraph's compiled
  graph config; enforce `timeout` as a wall-clock backstop.

Tier 1 supervisor: use LangGraph's prebuilt supervisor pattern (or a minimal
hand-rolled supervisor loop) — the plan picks the concrete approach but the
declarative surface above is fixed.

### 3.3 reasoning propagation

Orchestrator nodes that are Agents run inside the active `ReasoningChannel`
(§2.1); their reasoning bubbles to the orchestrator's stream. Subgraph
isolation of state is automatic (each `invoke` gets its own State).

### 3.4 Tests

Hermetic: use fake-LLM agents (`LLM("fake-1", provider="fake")`) as nodes.
Test all three tiers, the exit-form validation errors, parallel fan-out,
composition-cycle detection, and that `recursion_limit`/`timeout` are wired.
`langgraph>=1.0` is already a core dependency (§9.2) — no extra to add.

### 3.5 Exports

Export `Orchestrator`, `GraphState`, `END` from `aixon`.

---

## 4. Plan 5 — Server + ProtocolAdapter + adapters + auth (FastAPI/ASGI)

### 4.1 `aixon/server/protocol.py`

```python
# Re-export neutral types so adapters import from here (spec layout)
from aixon.message import Message, Chunk, Role

@dataclass
class ParsedRequest:
    model: str                 # the requested agent name/alias
    messages: list[Message]
    params: dict               # temperature, max_tokens, stream, etc.
    stream: bool

class ProtocolAdapter(ABC):
    """Translates a wire format <-> neutral types. New wire styles = new
    subclass. NO neutral type leaks a vendor/wire detail."""
    name: str                  # e.g. "openai", "anthropic"

    @abstractmethod
    def parse_request(self, body: dict, *, path: str) -> ParsedRequest: ...
    @abstractmethod
    def format_response(self, *, model: str, message: Message, usage: dict) -> dict: ...
    @abstractmethod
    def format_stream_chunk(self, *, model: str, chunk: Chunk) -> str:
        """Return one SSE 'data: {...}\\n\\n' line (or '' to skip)."""
    @abstractmethod
    def format_stream_done(self, *, model: str) -> str: ...
    @abstractmethod
    def format_models(self, agents: list) -> dict: ...
    @abstractmethod
    def routes(self) -> list[tuple[str, str]]:
        """[(http_method, path)] this adapter serves, e.g.
        [("POST","/v1/chat/completions"), ("GET","/v1/models")]."""
```

### 4.2 `aixon/server/adapters/openai.py` + `anthropic.py`

- `OpenAIAdapter` — full OpenAI-compatible: `/v1/chat/completions` (stream +
  non-stream), `/v1/models`. Chunk → `chat.completion.chunk` deltas; reasoning
  via a configurable mode (hidden | reasoning-field | inline `<think>`).
- `AnthropicAdapter` — thin proof: `/v1/messages`, system outside the array,
  typed content blocks, `content[]`/`stop_reason` envelope, named streaming
  events. Proves the neutral types aren't OpenAI-in-disguise.

### 4.3 `aixon/server/server.py`

```python
class Server:
    """ASGI server (FastAPI). Singleton. Mounts one or more ProtocolAdapters,
    backed by the agent Registry. Bearer auth via AUTH_API_KEY env (disabled
    if unset; /health and model-list stay public)."""
    @classmethod
    def get_instance(cls) -> "Server": ...
    def __init__(self, adapters: list[ProtocolAdapter] | None = None): ...  # default [OpenAIAdapter()]
    @property
    def app(self):  # the FastAPI ASGI app
        ...
    def serve(self, host="0.0.0.0", port=8000): ...   # uvicorn.run wrapper
```

Request flow: ASGI → adapter.parse_request → `get_registry().resolve(model)` →
`agent.invoke|stream` (neutral) → adapter.format_* → HTTP/SSE. Log resolved
agent name and route at INFO.

### 4.4 Tests

Use FastAPI `TestClient` and fake-LLM agents. Test both adapters'
request/response/stream shapes, model resolution, auth on/off, and that no
vendor type crosses into `Agent.invoke` (assert the agent sees neutral
`Message[]`). Add `fastapi`/`uvicorn`/`httpx` to the `server` extra (httpx for
TestClient).

### 4.5 Exports

Export `Server`, `ProtocolAdapter`, `OpenAIAdapter`, `AnthropicAdapter`,
`ParsedRequest` from `aixon` (or a documented `aixon.server` namespace).

---

## 5. Plan 6 — Retriever + Embedding + Connector (independent of 2–5)

### 5.1 `aixon/embedding.py`

```python
class Embedding(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

class OpenAIEmbedding(Embedding):
    def __init__(self, model: str, *, api_key_env: str = "OPENAI_API_KEY"): ...
    # lazy client; embed_* delegate to langchain_openai.OpenAIEmbeddings
```

### 5.2 `aixon/retriever.py`

```python
class TypeAccess(Enum):
    READ = "read"; WRITE = "write"; ALL = "all"

class Retriever(ABC):
    """Context search. Declarative subclasses end with 'Retriever'.
    Suffix-validated in __init_subclass__ (raise NamingError)."""
    description: str = ""
    type_access: TypeAccess = TypeAccess.READ

    @abstractmethod
    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        """Return [{'text': str, 'metadata': dict}, ...]."""
    def write(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        """Optional; default raises if type_access is READ-only."""
    def as_tool(self, name=None, description=None, k=None) -> AgentTool:
        """Return a neutral AgentTool (matches Agent.as_tool's shape) so a
        ToolAgent can consume it via coerce_tools."""
```

`Retriever.as_tool()` returns the SAME `aixon.AgentTool` dataclass that
`Agent.as_tool()` returns, so `coerce_tools` (§2.3) handles both uniformly.

### 5.3 `aixon/connector.py`

```python
class Connector:
    """Base HTTP client for an external microservice. Subclasses end with
    'Connector' (suffix-validated). Reads base_url/auth from env by default."""
    base_url_env: str = ""
    auth_token_env: str = ""
    def __init__(self, *, base_url=None, auth_token=None, timeout=None): ...
    def get(self, path, **kw) -> dict: ...
    def post(self, path, json=None, **kw) -> dict: ...
    # methods return parsed JSON dicts; use httpx
```

### 5.4 Tests / deps

Hermetic: fake embeddings (no network), `Retriever` tested with an in-memory
concrete subclass, `Connector` tested with a mocked httpx transport. New extra:
`retrieval = ["httpx>=0.27"]` (vector-store backends like weaviate stay as
further-optional extras, OUT of scope here — YAGNI). Export `Retriever`,
`TypeAccess`, `Embedding`, `OpenAIEmbedding`, `Connector`.

---

## 6. Plan 7 — CLI (`aixon chat|new|serve|list`)

`aixon/cli.py` (replaces the stub) using `click`. `[project.scripts] aixon =
"aixon.cli:app"` already wired.

```
aixon chat [--url URL] [--package PKG]   # interactive menu -> chat
aixon new <name>                          # scaffold a consumer project
aixon serve [--host] [--port] [--package] # uvicorn -> Server (Plan 5)
aixon list [--package PKG]                # list registered agents
```

- `chat` flow: `autodiscover(package)` (default `"agents"`), show menu of
  non-hidden agents (`get_registry().public()`), pick one, stream
  `agent.stream(messages)` to the terminal — `Chunk.reasoning` dimmed, then
  `Chunk.content`. Commands: `/menu`, `/exit`, `Ctrl+C` (interrupt gen; again
  at empty prompt → back to menu).
- Two modes: **in-process** (default; invoke agents directly) and **remote**
  (`--url`; OpenAI client against a running `aixon` server, reusing the
  OpenAI wire format).
- `serve` imports `Server` from Plan 5; `list` prints name/type/description.

Tests: use click's `CliRunner`; fake-LLM agents for `chat`/`list`; `serve`
tested by asserting it builds the app (don't actually bind a port). CLI deps in
the `cli` extra (`click` already there; add `openai>=1.0` for remote mode under
`cli`).

---

## 7. Plan 8 — Documentation (README + docs/)

Depends on all prior plans' public APIs (this contract is the source of truth
for signatures). Deliver README + `docs/` at restmcp level: philosophy,
layered architecture, the `Agent` model + three subtypes, declarative API,
suffix rules, the three Orchestrator tiers (incl. `entry`/topology vs textual
`edges` order, and the two branching kinds), `ProtocolAdapter`/decoupling,
`Retriever`/`Connector`/`LLM`/`Embedding`, CLI, and a consumer-project
quickstart. This is a docs plan: tasks are "write section X covering API Y",
verified by doctest-able snippets where practical and a link/consistency check
against this contract. No production code beyond doc examples.

---

## 8. Final package layout (target, after all plans)

```
aixon/
├── __init__.py
├── agent.py            # (P1) Agent + AgentTool
├── agents/
│   ├── llm_agent.py    # (P2)
│   ├── tool_agent.py   # (P3)
│   └── orchestrator.py # (P4)
├── llm.py              # (P2)
├── providers/
│   ├── base.py         # (P2)
│   ├── openai.py       # (P2)
│   ├── anthropic.py    # (P2)
│   └── google.py       # (P2)
├── _interop/           # private LangChain-boundary package
│   ├── __init__.py     # namespace only — no eager re-export (keeps boundary)
│   ├── messages.py     # (P2) neutral<->langchain message conversion
│   └── tools.py        # (P3) coerce_tools
├── reasoning.py        # (P3)
├── state.py            # (P4) GraphState, END
├── retriever.py        # (P6)
├── embedding.py        # (P6)
├── connector.py        # (P6)
├── server/
│   ├── server.py       # (P5)
│   ├── protocol.py     # (P5)
│   └── adapters/
│       ├── openai.py   # (P5)
│       └── anthropic.py# (P5)
├── discovery.py        # (P1)
├── registry.py         # (P1)
├── logging.py          # (P1)
├── message.py          # (P1)
├── exceptions.py       # (P1)
└── cli.py              # (P7)
```

---

## 9. Cross-plan reconciliation (authoritative — added after parallel plan-writing)

The seven plans were drafted in parallel; these rules resolve the seams where
they would otherwise collide. **The execution controller MUST apply these when
running the plans; they override any conflicting instruction inside an
individual plan.**

### 9.1 Test fakes — single owner: `tests/_fakes.py` (Plan 2)

`tests/_fakes.py` is created by **Plan 2** and is the ONE place hermetic test
doubles live. Plans 3, 4, 5, 7 **import from it — they do NOT redefine it.**
Plan 2's `tests/_fakes.py` MUST provide all of:

```python
# A fake provider + model so LLM("fake-1", provider="fake") works offline.
def register_fake_provider() -> None: ...      # idempotent; safe to call repeatedly
FAKE_MODEL = "fake-1"; FAKE_PROVIDER = "fake"

# A fake LangChain BaseChatModel that create_agent / LangGraph can drive
# offline. THIS EXACT CLASS IS VALIDATED against langchain 1.3 / core 1.4 /
# langgraph 1.2 — it drives langchain.agents.create_agent through a tool call
# then a final answer with NO API key and NO network. Use it verbatim:
from typing import Any, Optional, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

class FakeChatModel(BaseChatModel):
    """Scriptable offline chat model. `script` is a list of AIMessages returned
    one per LLM call (set tool_calls on an AIMessage to drive a tool step)."""
    script: list = []
    _idx: int = 0
    @property
    def _llm_type(self) -> str: return "fake"
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        return self                                  # tools ignored; script drives calls
    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        i = self._idx
        msg = self.script[i] if i < len(self.script) else AIMessage(content="(done)")
        object.__setattr__(self, "_idx", i + 1)
        return ChatResult(generations=[ChatGeneration(message=msg)])
# Example script for a tool-calling test:
#   FakeChatModel(script=[
#       AIMessage(content="", tool_calls=[{"name":"get_weather","args":{"city":"Recife"},"id":"call_1"}]),
#       AIMessage(content="The weather in Recife is sunny."),
#   ])
# The "fake" Provider's build() returns a FakeChatModel (its script settable per test).

# Convenience factories used by later plans' tests:
def make_llm(**params) -> "LLM": ...            # LLM(FAKE_MODEL, provider=FAKE_PROVIDER, **params)
def make_echo_agent(name: str = "echo", *, hidden: bool = False): ...
    # returns/registers a concrete Agent subclass whose invoke echoes the last
    # message and whose stream yields one content Chunk then done — for
    # server/CLI/orchestrator tests that need an Agent but not a real LLM.
```

- **Plan 5** drops its proposed `tests/_server_fakes.py`; it uses
  `make_echo_agent` from `tests/_fakes.py` (server tests need a plain Agent, not
  a real LLM — `make_echo_agent` covers it).
- **Plan 7** drops its own `make_echo_agent` redefinition; imports it from
  `tests/_fakes.py`.
- **Plan 4** removes its try/except fallback shims; it imports `make_llm` /
  `make_echo_agent` / `emit_reasoning` / `reasoning_channel` directly, because
  by execution time Plans 2 and 3 are merged (see 9.3 ordering).

### 9.2 Dependencies and extras (final, authoritative set in `pyproject.toml`)

**Core (mandatory) dependencies:** `langchain>=1.0`, `langchain-core>=1.0`,
`langgraph>=1.0`. The framework does not function without them — every agent
subtype (LLMAgent/ToolAgent) and the Orchestrator require langchain/langgraph,
so they are core `project.dependencies`, NOT an optional extra. `import aixon`
always pulls them in; there is NO `llm` extra and NO `orchestration` extra.
(Superseded the earlier design where langgraph lived in an `llm` extra behind a
bare-install guard — see §9.4.)

**Extras** (genuinely optional layers):
`dev`, `openai` / `anthropic` / `google` (provider bindings —
`langchain-openai` / `langchain-anthropic` / `langchain-google-genai`; pick the
one you use), `server` (`fastapi`, `uvicorn[standard]`, `pydantic`, `httpx`),
`retrieval` (`httpx>=0.27`), `openai-embedding` (`langchain-openai`), `cli`
(`click`, `openai>=1.0`). `all` aggregates every extra (it does NOT re-list the
core langchain stack, which is always installed). When two plans both touch
`all`, the controller merges (union).

### 9.3 Execution order (dependency-driven)

`2 → 3 → 4 → 5 → 7 → 8`, with `6` insertable anywhere after `1` (it is
independent). Rationale: 3 consumes 2's `LLM`; 4 consumes 2+3; 7's `serve`
consumes 5; 8 documents all. Each plan is still its own branch/review cycle.

### 9.4 Top-level export guard (for extra-only layers)

Exports whose deps live in an OPTIONAL extra stay behind a `try/except
ImportError` in `aixon/__init__.py` so `import aixon` works without that extra.
This applies to the `Server`/adapter surface (needs the `server` extra:
FastAPI/uvicorn). It does NOT apply to `LLM`/`LLMAgent`/`ToolAgent`/
`Orchestrator`: langchain/langgraph are core dependencies (§9.2), always
present, so those are plain unconditional top-level imports — no guard. Provider
SDKs (openai/anthropic/google bindings) are still loaded lazily inside provider
methods, so `import aixon` does not require any specific provider binding.

### 9.5 Dedicated virtualenv (REQUIRED — do not reuse another project's venv)

All plans from 2 on install real deps (langchain/langgraph/fastapi). They MUST
run in a **dedicated aixon venv**, never a shared one. The controller creates
it ONCE before Plan 2 and every run/install step uses it:

```bash
cd /Users/jorge/Documents/Git/aixon
python3 -m venv .venv                      # .venv is git-ignored
.venv/bin/python -m pip install -e ".[dev,openai,server,retrieval,cli]"
```

Every plan's run step uses `.venv/bin/python -m pytest ...` (NOT a bare
`pytest` — the console script can carry a stale shebang; and NOT another
project's interpreter). The controller injects this venv path into each
implementer dispatch. langgraph/langchain are validated at 1.x (langchain 1.3,
langchain-core 1.4, langgraph 1.2); do not pin a `<1` ceiling anywhere.
