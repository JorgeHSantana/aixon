# Agents

An **`Agent`** is the single executable unit in `aixon`. Every agent — regardless
of subtype — exposes the same interface:

```python
agent.invoke(messages: list[Message]) -> Message
agent.stream(messages: list[Message]) -> Iterator[Chunk]
agent.as_tool(name=None, description=None, memoize=True, audience="human",
              args_schema=None, render_input=None) -> AgentTool
```

This uniformity means a `ToolAgent` can be a node in an `Orchestrator`, an
`Orchestrator` can be a tool inside a `ToolAgent`, and the `Server` never needs
to know which subtype it is calling.

---

## Declaring an agent

Subclass one of the concrete types and set class attributes. The agent
self-registers when Python processes the class body — no call to a registration
function required.

### Common attributes (all subtypes)

| Attribute | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | class name lowercased | Registry key and API `model` field. |
| `description` | `str` | `""` | Human-readable purpose; shown in `aixon list` and the chat menu. |
| `aliases` | `list[str]` | `[]` | Alternate registry names. |
| `hidden` | `bool` | `False` | Exclude from `get_registry().public()` and the `aixon chat` menu. |
| `owned_by` | `str` | `"aixon"` | Shown in `/v1/models` response. |
| `impose_model` | `bool` | `False` | `True` imposes the model of this agent's own `llm` on every LLM beneath it (its tools, sub-agents, judges, nodes) while it runs. Needs a class-level `llm`. See [Modelo imposto](#modelo-imposto-impose_model). |

---

## LLMAgent — direct LLM call

Use `LLMAgent` when you want a single LLM call with no tool loop — the simplest
path from question to answer.

```python
from aixon import LLMAgent, LLM

class PlannerAgent(LLMAgent):
    llm         = LLM("gpt-4o-mini", temperature=0.2)
    description = "Breaks complex goals into step-by-step plans"
    prompt      = "You are a concise strategic planner. Use numbered lists."
```

**Additional `LLMAgent` attributes:**

| Attribute | Type | Required | Description |
|---|---|---|---|
| `llm` | `LLM` | **Yes** | The language model. Missing `llm` on a concrete subclass raises `AixonError` at import time. |
| `prompt` | `str` | No | System prompt prepended to every `invoke`/`stream` call. |
| `client_tools` | `str` | No (default `"ignore"`) | Raw passthrough (#18a, unified with `ToolAgent`'s vocabulary/default by #25): `"passthrough"` forwards the request's `aixon.runtime.current_client_tools()`/`current_tool_choice()` straight to `LLM.complete`/`acomplete`/`stream`/`astream` for all 4 methods — the agent itself decides nothing, it just gives the model the client's tools. `"ignore"` (default) → no-op, as does `"passthrough"` when no tools are declared on the request. A `bool` raises `AixonError` at registration with a migration message (`False`→`"ignore"`, `True`→`"passthrough"`). See [server.md](server.md#openaiadapter) ("Client tools") for the request/response shape and the first-class `ToolAgent(client_tools="merge")` alternative below. |

**How it works:** `invoke` prepends the system prompt (if any) as a
`Message(role="system", content=self.prompt)` and delegates to
`self.llm.complete(messages)`. `stream` delegates to `self.llm.stream(messages)`,
yielding `Chunk` deltas and a final `Chunk(done=True)`. A leading `system` (or
`developer` — OpenAI's system-role alias, treated identically) message in the
caller's `messages` **wins** over the class-level `prompt` instead of both
reaching the provider as two separate system messages; if that leading
message's content is empty, it falls back to `self.prompt`. (`LLMAgent` has no
`client_system`: the leading client message always replaces the prompt here.
`ToolAgent` alone can append it instead — see below.)

### LLM — declaring a language model

```python
from aixon import LLM

# Explicit provider
llm = LLM("claude-3-5-haiku-20241022", provider="anthropic", temperature=0.3)

# Inferred provider (model prefix → provider):
#   gpt-* / o[0-9]* / text-*  →  openai
#   claude-*                   →  anthropic
#   gemini-*                   →  google
llm = LLM("gpt-4o-mini", temperature=0.2, max_tokens=4096)
```

The `LLM` object is lazy — it builds the underlying LangChain `BaseChatModel`
only on first use, so constructing an agent never requires a network call or an
API key to be present at import time.

**Provider inference table** (the model name's prefix selects the provider):

| Model prefix | Provider name |
|---|---|
| `gpt-*`, `o[0-9]*`, `text-*` | `"openai"` |
| `claude-*` | `"anthropic"` |
| `gemini-*` | `"google"` |
| `glm*` | `"zai"` |
| `grok*` | `"xai"` |
| _(no prefix)_ | `"gateway"` — never inferred: reached only with `provider="gateway"` |

Provider names are lowercase strings, not an enum. To override inference, pass
`provider=` explicitly: `LLM("some-model", provider="openai")`.

**z.AI (GLM models).** `LLM("glm-4.6", provider="zai")` (or a bare `glm-*` model
name, inferred) reuses `langchain_openai.ChatOpenAI` pointed at the z.AI
OpenAI-compatible endpoint. `ZAI_API_KEY` is **required** — unlike the other
providers, it does not fall back to `OPENAI_API_KEY` if unset; building the
model raises `AixonError` instead of silently sending your OpenAI credential
to the z.AI endpoint. `ZAI_BASE_URL` overrides the default
(`https://api.z.ai/api/paas/v4`).

**xAI (Grok models).** `LLM("grok-4", provider="xai")` (or a bare `grok-*`
model name, inferred) reuses `langchain_openai.ChatOpenAI` pointed at the xAI
OpenAI-compatible endpoint. `XAI_API_KEY` is **required** — unlike the other
providers, it does not fall back to `OPENAI_API_KEY` if unset; building the
model raises `AixonError` instead of silently sending your OpenAI credential
to the xAI endpoint. `XAI_BASE_URL` overrides the default
(`https://api.x.ai/v1`).

**Gateway (one OpenAI-compatible gateway in front of every call).**
`LLM("tier-x", provider="gateway")` reuses `langchain_openai.ChatOpenAI` pointed
at a gateway that speaks the OpenAI Chat Completions contract — a gateway like
LiteLLM, where the caller names a *tier*, the gateway picks the real model and
meters the spend per end user. Unlike z.AI and xAI there is no model-name prefix
to infer it from: it is reached **only** through an explicit
`provider="gateway"`. Three environment variables configure it:

| Variable | Default | Description |
|---|---|---|
| `AIXON_GATEWAY_BASE_URL` | _(required)_ | The gateway's OpenAI-compatible URL, e.g. `http://gateway.example/v1`. |
| `AIXON_GATEWAY_API_KEY` | _(required)_ | The gateway's key. |
| `AIXON_GATEWAY_USER_HEADER` | `X-OpenWebUI-User-Email` | Name of the header that carries the end user's e-mail, stripped. Empty (or blank) turns the forwarding off; a value that is not a valid HTTP header name (a space, a line break, a `:`) makes the build raise `AixonError`. |

A missing or blank required variable makes building the model raise
`AixonError`: the provider never reads `OPENAI_API_KEY`, `OPENAI_BASE_URL` or
`OPENAI_API_BASE`, so an OpenAI credential is never sent to the gateway by
accident. Nor do `OPENAI_PROXY`, `OPENAI_ORG_ID`/`OPENAI_ORGANIZATION` or
`OPENAI_PROJECT_ID`, which are settings of the OpenAI agents, reach the gateway's
client. Install with `pip install "aixon[gateway]"` (pulls in
`langchain-openai`).

*Who is asking.* Spend and limits at the gateway are per person, not per shared
key, so every call made while `aixon.runtime.user_email_scope(email)` is active
carries that e-mail in the header above. The `Server` opens the scope for each
request from its `x-openwebui-user-email` header (see
[server.md](server.md#request-flow)). The e-mail is read when the HTTP request
is **sent**, from the caller's context, not when the model is built: the `LLM`
caches one model, and that model serves many people. The hook is installed on
**both** the sync HTTP client (`invoke`/`stream`) and the async one
(`ainvoke`/`astream`, the path the `Server` runs agents on) — a hook on the sync
client alone would be skipped on the async path with no error, and every user
would be billed to the shared key. Without an e-mail no header goes out at all
(refusing a request that arrives without identity is the application's job),
and no other provider ever receives the e-mail. The e-mail must be ASCII (it is
an HTTP header): an internationalized address is not supported in this version.

*When the gateway says no.* A spent budget (HTTP 422 mentioning "budget") or a
rate limit (HTTP 429) reaches the caller as the SDK's own error,
`openai.UnprocessableEntityError` or `openai.RateLimitError`: aixon does **not**
raise `GatewayBudgetExceeded` or `GatewayRateLimited`. `classify_gateway_error(exc)`
(in `aixon.exceptions`) says which of the two an error stands for, walking its
`__cause__`/`__context__` chain (a `raise ... from None` cuts the context), or
returns `None`; it never raises. Only an error from a request to
`AIXON_GATEWAY_BASE_URL` counts. A tool may raise either class itself to signal a
quota, and it is treated as the gateway's verdict. The `ToolAgent` shield, the
`ReflectiveAgent` and the `Server` use this classification (see
[server.md](server.md#errors)).

**`pinned`.** `LLM(..., pinned=True)` keeps this `LLM`'s own model and provider
even while a model is imposed on the call (see [Modelo imposto](#modelo-imposto-impose_model)
below). Default `False`. `pinned` is a keyword of `LLM`, not a vendor param: it
never reaches the provider's constructor.

### Reasoning (extended thinking / reasoning effort)

`LLM(model, reasoning=...)` turns on the provider's native reasoning/thinking
mode:

```python
llm = LLM("claude-sonnet-4-5", reasoning=True)                    # {"effort": "medium"}
llm = LLM("claude-sonnet-4-5", reasoning={"effort": "high"})
llm = LLM("claude-sonnet-4-5", reasoning={"budget_tokens": 8000})
llm = LLM("gpt-5.4", reasoning={"effort": "low"})
```

- `None`/`False` (the default) — off; behavior is byte-for-byte unchanged from
  before the knob existed.
- `True` — shorthand for `{"effort": "medium"}`.
- A `dict` may give `budget_tokens`, `effort`, or both; whichever half is
  missing is derived from the fixed table below (an already-complete dict is
  kept exactly as given, no re-derivation):

| Effort | Budget tokens |
|---|---|
| `low` | 1024 |
| `medium` | 4096 |
| `high` | 16384 |

A bare `budget_tokens` is bucketed into the nearest effort tier the other way
(`<= 1024` → `low`, `<= 8192` → `medium`, else `high`) for providers with only
a coarse effort dial.

**Per-provider translation:**

| Provider | Translation |
|---|---|
| `anthropic` | `thinking={"type": "enabled", "budget_tokens": ...}`. Anthropic's extended-thinking API requires `temperature == 1`; the knob **forces** it (logging a warning if the caller/request asked for a different value). `max_tokens` is raised to `budget_tokens + 4096` when absent or not already comfortably above the budget. |
| `openai` | `reasoning_effort=<effort>` constructor kwarg on `ChatOpenAI`. No budget dial — only the effort string reaches the API. |
| `xai` (Grok) | `reasoning_effort=<effort>` constructor kwarg on `ChatOpenAI`, forwarded verbatim — same translation as `openai`. |
| `gateway` | `reasoning_effort=<effort>` constructor kwarg on `ChatOpenAI`, forwarded as is: the gateway hands it to whichever model it picked. Not sent under an imposed model (see [Modelo imposto](#modelo-imposto-impose_model)). |
| `zai` (GLM) | `extra_body={"thinking": {"type": "enabled", ...}}` (merged with any caller-supplied `extra_body`). GLM has no budget/effort dial of its own — any non-off spec just turns thinking on. |
| `google` (Gemini) | `thinking_budget=<budget_tokens>` and `include_thoughts=True` on `ChatGoogleGenerativeAI` — applied only if the installed `langchain-google-genai` declares those fields; an older install degrades gracefully (knob ignored, warning logged) instead of raising on an unknown kwarg. |
| custom (no `supports_reasoning = True`) | the knob is **ignored** (with a warning) rather than forwarded — a pydantic-strict vendor constructor never sees the stray `reasoning` kwarg, so the build never breaks. |

**Per-request override.** `reasoning_effort` in the request body (see
[server.md](server.md)) is allow-listed the same way as `temperature`/
`max_tokens`/etc., and, when present, overrides the class-level `reasoning=`
knob for that one build — translated as `{"effort": reasoning_effort}` through
the same table above.

**What actually comes back — read before relying on visible reasoning text:**
- **Anthropic** extracts `thinking` blocks into `Message.reasoning` /
  `Chunk.reasoning` (see [architecture.md](architecture.md#the-neutral-boundary))
  — real, provider-generated chain-of-thought text.
- **Gemini** does the same when `include_thoughts=True` is applied (always the
  case when the knob is on and the installed package supports it).
- **OpenAI's API does not return raw chain-of-thought at all.**
  `reasoning_effort` makes the model think harder and improves the answer,
  but there is no reasoning text to extract — `Message.reasoning` stays
  `None` for OpenAI models regardless of the knob.
- **z.AI (GLM)**: `thinking` IS enabled on the wire, but the installed
  `langchain-openai` does not populate `additional_kwargs["reasoning_content"]`
  from the Chat Completions response today — a provider/package gap, not
  something aixon papers over. Extraction already supports the
  `reasoning_content` convention the moment the installed SDK starts filling
  it in; until then, GLM reasoning text does not surface even though thinking
  is enabled server-side.
- **Cost.** Thinking/reasoning tokens bill as output tokens and already show
  up in `Message.usage["completion_tokens"]` — no separate accounting needed.

**Limitation — Anthropic extended thinking + client-executed tools across
requests.** A CLIENT-executed tool loop (see
[server.md](server.md#openaiadapter) — an agentic client like an editor/IDE
that calls tools itself and sends the results back on a *later* HTTP
request) does not round-trip Anthropic's `thinking` blocks: the neutral
boundary's `to_langchain` conversion drops reasoning content by design (the
neutral `Message`/`Chunk` types carry `reasoning` as plain text, not
Anthropic's signed thinking-block wire format), and Anthropic's API rejects a
request that mixes extended thinking with tool results whose matching
thinking block isn't present in that same request. Concretely: a reasoning-
enabled Claude model that returns a tool call to a client-executed tool, then
receives that tool's result back on the NEXT request, will get a 400 from
Anthropic. A normal in-process `ToolAgent` loop (tools executed by aixon
itself, all within one request/one call to the model) is unaffected — this
only bites client-executed tools spanning multiple HTTP requests.

### Per-request generation params

When an agent runs behind the `Server`, per-request generation params
(`temperature`, `top_p`, `max_tokens`, `presence_penalty`, `frequency_penalty`,
`stop`, `reasoning_effort`) are published on a `ContextVar` for the duration of
the call (see `aixon.runtime.generation_params`) and apply **on top of** the
`LLM(...)` class-level defaults, without mutating them:

- `LLMAgent` applies them via `LLM._bound_model()`, which delegates straight
  to `LLM.request_chat_model()` — the same path `ToolAgent` uses (see below).
- `ToolAgent` applies them via `LLM.request_chat_model()`: it builds a
  provider model with the params merged in as constructor kwargs *before*
  `Provider.build()` runs, so they go through the same reasoning translation
  and validation every other constructor kwarg gets. No active params (and no
  imposed model, below) → the cached `chat_model` (no rebuild). Models built
  for repeated identical param combinations are cached (bounded to 8 entries,
  oldest-evicted-first) so a hot request path reuses one provider client (and
  its HTTP connection pool) instead of rebuilding an SDK client per call. The
  cache key is the model, the provider, whether the model was imposed, and the
  params; a lock guards the cache, because agents run on worker threads and
  one `LLM` serves several requests at once.
- **Under an imposed model** (`impose_model`, see
  [Modelo imposto](#modelo-imposto-impose_model)) `request_chat_model()` builds
  THAT model through the imposed provider, even with no params at all, and
  hands it only the portable params. The same `LLM` used inside and outside
  the scope never gets the other's model.

Both paths read the exact same `ContextVar` and go through the exact same
`request_chat_model()` builder, so a request's `temperature` override behaves
identically whether the resolved agent is an `LLMAgent` or a `ToolAgent` —
`LLMAgent` used to bind params at invoke time on top of an already-built
model instead, which bypassed provider translation entirely (a client
`reasoning_effort` reached the vendor SDK raw, and a client `temperature`
could override the constructor-forced `temperature=1` Anthropic's
extended-thinking API requires); that gap is closed now that both agent
kinds build through the same path.

For a custom backend, subclass the `Provider` ABC (`aixon.providers.base`) and
register a single instance before first use:

```python
from aixon.providers.base import Provider, register_provider

class MyProvider(Provider):
    name = "myvendor"
    env_key = "MYVENDOR_API_KEY"
    def build(self, model: str, **params):
        from my_sdk import ChatModel        # lazy import
        return ChatModel(model=model, **params)

register_provider(MyProvider())             # one instance, keyed by .name
# then: LLM("my-model", provider="myvendor")
```

### Modelo imposto (`impose_model`)

Há deploys em que toda chamada de LLM passa por um gateway (o provedor
`gateway`, acima) e quem escolhe o modelo é o gateway, por tier, não a classe de
cada agente. `impose_model` leva essa escolha à árvore inteira: o agente declara
o tier uma vez, e todo `LLM` abaixo dele roda nele.

```python
from aixon import LLM, ToolAgent

class EspecialistaAgent(ToolAgent):
    llm = LLM("gpt-5.4")                   # declarado na OpenAI...

class CoordenadorAgent(ToolAgent):
    llm          = LLM("tier-x", provider="gateway", temperature=0.2)
    prompt       = "Você coordena os especialistas."
    impose_model = True                    # ...mas roda no tier-x, pelo gateway
    tools        = [EspecialistaAgent().as_tool(name="especialista")]
```

Enquanto um entry point do `CoordenadorAgent` roda (`invoke`, `ainvoke`,
`stream` ou `astream`), o par (modelo, provedor) do `llm` dele fica publicado
num `ContextVar` (`aixon.runtime.imposed_model_scope` /
`current_imposed_model`). Todo `LLM` que não seja `pinned` e rode ali — as tools,
os subagentes via `as_tool()`, o worker e o juiz de um `ReflectiveAgent`, os nós
de um `Orchestrator` — constrói esse modelo pelo provedor imposto, no lugar do
que declarou. Os padrões preservam o comportamento de antes: `impose_model =
False` e um `LLM` fora de qualquer escopo não mudam em nada.

- **Validação.** `impose_model` tem de ser `bool`, e `True` exige um `llm` de
  classe do tipo `LLM` (aquele de onde sai o modelo imposto). As duas coisas são
  checadas na definição da classe e levantam `AixonError` antes do registro.
- **Só quem impõe é embrulhado.** Na definição da classe, uma classe com
  `impose_model = True` (declarado nela ou herdado, também numa subclasse
  abstrata) troca os `invoke`/`ainvoke`/`stream`/`astream` que tem — escritos
  no corpo, herdados de um tipo do aixon ou de um mixin — por wrappers que abrem
  o escopo; as subclasses dela herdam os wrappers e embrulham os entry points
  que reescrevem. Uma classe que não impõe fica exatamente como na 0.1.27: os
  entry points dela continuam sendo as `async def` e os geradores do corpo, e
  quem decide por introspecção (`unittest.mock`, `RunnableLambda`, um nó do
  LangGraph, `coerce_tools`) não vê diferença. Por isso `impose_model` vale só
  se estiver na classe quando ela é criada: `Cls.impose_model = True` atribuído
  depois não embrulha nada e não impõe nada, em silêncio.
- **O escopo de fora vale.** Se já há um modelo imposto quando o agente começa a
  rodar — por um agente mais externo, ou pela aplicação, com
  `imposed_model_scope(modelo, provedor)` — o agente não abre o escopo dele: o
  escopo mais externo decide. Um subagente com `impose_model = True` dentro de
  um agente que não impõe, ao contrário, impõe o próprio modelo à própria
  subárvore.
- **Parâmetros portáveis.** Sob o escopo, só chegam ao build do modelo imposto
  `temperature`, `top_p`, `max_tokens`, `stop`, `presence_penalty` e
  `frequency_penalty` (`aixon.runtime.PORTABLE_PARAMS`), vindos da declaração do
  `LLM` e da request (a request vence; `None` é descartado). Ficam de fora o
  `reasoning` da classe, o `reasoning_effort` da request, o `prediction` e todo
  outro kwarg da classe (`timeout`, `max_retries`, `seed`, `model_kwargs`...):
  são ajustes para o modelo para o qual o agente foi escrito, e o ajuste do tier
  é do gateway. Os nomes alternativos `max_completion_tokens` e `stop_sequences`
  contam como `max_tokens` e `stop`, na classe e na request.
  `LLM(cache=True)` também segue o provedor que de fato responde.
- **O próprio impositor também.** Sob o próprio escopo, o `llm` de quem impõe é
  mais um `LLM` da árvore: os kwargs não portáveis dele (`timeout`,
  `max_retries`, `seed`, `reasoning`...) também não chegam ao modelo imposto. Na
  definição da classe, o `aixon.agent` loga um WARNING com as chaves que ficam
  de fora. Para manter a configuração do impositor nas chamadas dele, declare o
  `llm` dele com `LLM(..., pinned=True)` (o modelo dele continua sendo o
  imposto à árvore).
- **Timeout e retries sob imposição.** Como nenhum `timeout`/`max_retries` de
  classe passa, todo `LLM` da árvore roda com os padrões do provedor (120 s e 2
  retries). Um 429 cujo `Retry-After` passa do `max_execution_time` de um
  especialista vira `TOOL ERROR … tempo limite` para o coordenador, que segue
  sem ele, em vez da mensagem de limite: um limite conhecido, anterior a esta
  versão (na 0.1.27 todo 429 de especialista virava `TOOL ERROR`). O 422 de
  orçamento nunca é repetido pelo SDK.
- **Falha fechada.** Sob o escopo, `llm.chat_model` — o modelo com que o `LLM` foi
  CONFIGURADO — levanta `ImposedModelViolation` (um `AixonError`) em vez de
  entregá-lo: rodar a chamada no modelo errado sem avisar seria pior que parar.
  Respeitam o escopo `request_chat_model()`, `complete`/`acomplete`/`stream`/
  `astream` e o `ToolAgent`. Para o `LLM` que precisa do modelo próprio, declare
  `LLM(..., pinned=True)`. Um provedor imposto que não está registrado também
  falha alto: `request_chat_model()` levanta `AixonError` citando o nome.
- **Escopo por passo nos streams.** Em `invoke`/`ainvoke` o escopo envolve a
  chamada inteira. Em `stream`/`astream` ele abre por passo — só enquanto o
  iterador interno roda (criá-lo, cada `next`/`__anext__`, `close`/`aclose`) — e
  fecha antes de cada chunk ser entregue. Quem consome o stream nunca vê o modelo
  imposto, nem entre os chunks nem depois de abandonar o stream (`break`), e
  consumidores que avançam o gerador de contextos diferentes a cada passo
  funcionam. As tasks do asyncio e as threads que copiam o contexto (veja os
  limites) iniciadas durante um passo herdam o escopo: é assim que o grafo e as
  tools o recebem.
- **Limites.**
  - O `Embedding` não é coberto: ele não passa pelo `LLM`, então um
    `OpenAIEmbedding` continua usando o provedor e a chave dele sob um modelo
    imposto.
  - `MCPToolset._discover`, quando chamado de dentro de um loop em execução (o
    caminho do `Server`), roda numa thread de worker sem `copy_context` (#32).
    A descoberta só faz `tools/list`, sem LLM, então não é afetada, mas nada que
    dependa do escopo funciona ali.
  - Um entry point atribuído à classe depois de criada (`Cls.invoke = f`), como
    um `impose_model` atribuído depois, não é embrulhado: roda fora do escopo, e
    `impose_model = True` não faz nada por ele, em silêncio.
  - Os entry points de uma classe que impõe são funções comuns que embrulham as
    originais: nela, `inspect.iscoroutinefunction(agent.ainvoke)`,
    `inspect.isasyncgenfunction(agent.astream)` e
    `inspect.isgeneratorfunction(agent.stream)` dão `False`. Para usar um
    impositor como tool ou nó, passe `agent.as_tool()` ou uma `async def` que faz
    `await agent.ainvoke(...)`, e nos dublês use `new_callable=AsyncMock`. Não use
    o `__wrapped__`: chamá-lo pula o escopo.
  - Um salto de thread sem cópia de contexto perde o modelo imposto E o e-mail.
    Threads criadas com `threading.Thread`, `loop.run_in_executor` ou
    `ThreadPoolExecutor.submit` não copiam o contexto: um `LLM` chamado de lá
    não vê o modelo imposto e roda no modelo configurado dele, sem erro, e
    também não vê o e-mail de `user_email_scope`, e a chamada ao gateway sai sem
    o header. `asyncio.to_thread`, as tasks do asyncio e os executors do
    LangChain e do LangGraph copiam, e as pontes sync/async do próprio aixon
    também.

---

## ToolAgent — LLM + tool-calling loop

Use `ToolAgent` when your agent needs to call external functions, query a
`Retriever`, or invoke another agent as a tool, then loop until it has a final
answer.

```python
from aixon import ToolAgent, LLM
from langchain_community.tools import DuckDuckGoSearchRun  # pip install langchain-community

from retrievers.library import LibraryRetriever

class ResearchAgent(ToolAgent):
    llm                = LLM("gpt-4o-mini", temperature=0.1)
    description        = "Researches topics using web search and the knowledge base"
    prompt             = "Always cite your sources. Think step by step."
    tools              = [LibraryRetriever, DuckDuckGoSearchRun()]
    max_iterations     = 15
    max_execution_time = 600
```

**Additional `ToolAgent` attributes:**

| Attribute | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLM` | **Required** | The language model driving the loop. |
| `prompt` | `str` | `""` | System prompt. |
| `tools` | `list` | `[]` | Mix of `AgentTool`, `Retriever`, LangChain `@tool` functions, or any callable. All are coerced to `BaseTool` internally via `coerce_tools`. |
| `max_iterations` | `int` | `15` | Maximum tool-call rounds before the loop stops. |
| `max_execution_time` | `int` | `600` | Wall-clock timeout in seconds. |
| `timeout_message` | `str` | see below | Content emitted (#19) when `stream`/`astream` hits `max_execution_time` with NO answer accumulated yet — `{seconds}` is interpolated. See "Timeout sem morrer mudo (#19)" below. |
| `tool_call_label` | `str` | `"Calling {name}..."` | `{name}`-templated reasoning label emitted before each tool call. Override for a friendlier phrase or i18n, e.g. `"Chamando {name}..."`. Consecutive duplicate labels are emitted once (a run calling the same tool N times in a row shows a single line). |
| `shield_tool_errors` | `bool` | `True` | Error shield: any exception a tool raises (`httpx.ReadTimeout`, DB down, ...) becomes a readable `TOOL ERROR (...)` result handed back to the model — the agent reports the outage and/or proceeds, instead of the whole run/stream dying with an opaque server error. The one exception is a verdict of the gateway (budget spent, rate limited — `classify_gateway_error`): it is the end of the request, not a tool failing, so the shield re-raises the same object (after `on_tool_end` has seen it) for the `Server` to answer. Only an error from a request to `AIXON_GATEWAY_BASE_URL` counts; a 429 from any other provider is shielded as before. `False` restores the strict pre-shield behavior (exceptions propagate). Raw `BaseTool` entries are NOT shielded (see `coerce_tools`). |
| `prune_tool_results_after` | `int \| None` | `None` | Opt-in pruning of old tool results (#16). `None` disables it (zero behavior change). See "Poda de tool results antigos (#16)" below. |
| `on_tool_start` | `method` | no-op | Pre-call hook (#17). See "Hooks de tool call (#17)" below. |
| `on_tool_end` | `method` | no-op | Post-call hook (#17). See "Hooks de tool call (#17)" below. |
| `client_tools` | `str` | `"ignore"` | First-class client tools (#18c): `"ignore"` \| `"merge"` \| `"replace"`. See "Client tools mesclados no loop (#18c)" below. |
| `client_tools_conflict` | `str` | `"error"` | Name-collision policy between an internal tool and a client def: `"error"` \| `"internal"` \| `"client"`. |
| `client_tools_filter` | `method` | identity | Curation hook (#18c): `client_tools_filter(self, defs) -> defs` — override to keep only a subset of the client's declared tools. |
| `client_system` | `str` | `"replace"` | What a leading `system` (or `developer`) message sent by the CLIENT does to `prompt`: `"replace"` (default, the only behavior before 0.1.28) takes the prompt's place; `"append"` keeps the prompt first and puts the client's text after it, as context. Anything else raises `AixonError` at registration. `ToolAgent` only — see below. |
| `reasoning_flush_interval` | `float` | `0.25` | `astream` only (#20): how often (seconds) the ReasoningChannel is polled and drained while a tool node is still running. See "Streaming ao vivo (#20)" below. |

**Streaming ao vivo (#20).** `astream` polls for the next LangGraph update
instead of blocking on it, draining and yielding the ReasoningChannel every
`reasoning_flush_interval` regardless of whether the update has arrived — so
reasoning a NESTED agent emits from inside a still-running tool call (e.g. a
specialist agent invoked as a tool, itself streaming labels/retries) surfaces
as soon as it's emitted instead of being buffered until that tool call
completes. The sync `stream()` keeps its old between-update drain: a blocking
iterator can't be polled concurrently with a timer, so live progress during a
slow nested tool is an `astream`-only capability — which is what `Server` uses.

**Timeout sem morrer mudo (#19).** If `max_execution_time` breaks a `stream`/
`astream` run (the `timed_out`/deadline-break path) with **no** answer
accumulated yet, the stream still emits a `Chunk(content=...)` carrying
`timeout_message` (default: a short PT-BR sentence naming the elapsed
seconds) right before the closing `Chunk(done=True)` — the "(stopped:
exceeded max_execution_time ...)" reasoning line alone is not enough, since
chat UIs collapse/hide reasoning and would otherwise show an empty reply. A
run that already produced content by the time the deadline breaks (e.g. the
model's final answer text, as opposed to a tool-call preamble) keeps
delivering that content unchanged — `timeout_message` never overrides real
output. `invoke`/`ainvoke` are untouched: they already raise a visible
`AixonError` on timeout.

**Tool-call memoization (request scope).** Within one served request (and
within one `ReflectiveAgent` run), a tool called again with the SAME arguments
returns the first result instead of re-executing — retry loops stop re-running
identical DB queries/web searches, and the corrected answer stays consistent
with the data the judge criticized. The cache dies with the request; errors
are never cached. Opt out per tool for intentionally non-deterministic or
write tools: `retriever.as_tool(memoize=False)`, `agent.as_tool(memoize=False)`,
or `my_function.aixon_memoize = False` on a plain callable.

Like `LLMAgent`, a leading `system` (or `developer`) message in `messages`
overrides `self.prompt` as the graph's `system_prompt` rather than both being
sent to the provider.

**`client_system = "append"`** is for a chat UI that sends a system message of
its own (a model's, or a user's) to an agent whose `prompt` must keep governing
the answer: the system prompt becomes `prompt`, a blank line, then the client's
text. A side that is empty drops out, and two empty sides send no system prompt
at all. Only the first message counts (it is consumed here, never sent as a
message of its own); the default `"replace"` is exactly the behavior above.
`client_system` exists **only on `ToolAgent`**: an `LLMAgent`, which always lets
the client's message replace the prompt, never reads it, and declaring it there
— or on a `ReflectiveAgent`, where the worker's own `client_system` is the one
that counts — is accepted and ignored, without an error.

**Tool coercion:** anything in `tools` is normalized at runtime:
- An `AgentTool` (from `Agent.as_tool()` or `Retriever.as_tool()`) → `StructuredTool`
- A LangChain `BaseTool` or `@tool`-decorated function → passed through
- A plain callable → wrapped via `StructuredTool.from_function`

This means you can mix library tools, custom functions, and other agents freely.

**Paralelismo de tool calls (#13).** When the model emits several tool calls
in the same turn, both entry points run them concurrently already —
`langchain.agents.create_agent`'s internal `ToolNode` (langgraph 1.2) fans
out a turn's calls itself, no aixon-side change needed:
- **Async** (`ainvoke`/`astream` — what `Server` uses): `asyncio.gather` over
  each call's coroutine.
- **Sync** (`invoke`/`stream`): a thread-pool `executor.map` (LangChain's
  `get_executor_for_config`), so blocking calls (e.g. `time.sleep`,
  synchronous HTTP) also overlap.

Guaranteed by a regression test (`tests/test_tool_parallel.py`, async path):
two tools that each `await asyncio.sleep(0.4)` complete a turn in ~0.4s, not
~0.8s. Prefer `async def` tools with `ainvoke`/`astream` regardless — the
thread pool backing the sync path is an implementation detail of the
installed langgraph version, not a contract aixon pins with its own test.

### Poda de tool results antigos (#16)

Agentes que ficam muito tempo em conversas com múltiplos turnos de consulta a
banco (ex.: a família Analista/Gerente sobre `CropnetDB`) acumulam, no
histórico, os resultados brutos de tool calls de turnos já respondidos — cada
turno novo reenvia esses dumps ao provider mesmo que o modelo já os tenha
consumido e resumido na resposta anterior. `prune_tool_results_after` é opt-in
para esse caso: ligue-o em agentes cujas queries tendem a devolver payloads
grandes (`sql_static`, `cropnet_db`, etc.).

```python
class GerenteAgent(ToolAgent):
    llm = LLM("gpt-5.4", temperature=0.2)
    tools = [...]
    prune_tool_results_after = 1  # mantém as tool results do round mais recente
```

Com um `int N`, a âncora é por **rounds completos** — um round termina numa
mensagem `role="assistant"` SEM `tool_calls` (sua resposta final); a mensagem
`assistant` que EMITIU a tool call pertence ao round que ela abriu, não é um
limite de round. Toda mensagem `role="tool"` que apareça ANTES do início dos
últimos `N` rounds completos é substituída, só no payload enviado ao
provider, por um stub curto:

```
[resultado de ferramenta omitido (3000 caracteres) — já utilizado em resposta anterior]
```

`N=1` preserva sempre o round MAIS RECENTE (inclusive um round ainda em
andamento — histórico terminando em `assistant(tool_calls=...)` sem resposta
final ainda). Uma versão anterior desta poda ancorava em toda mensagem
`assistant` (contando também a que emitiu a tool call): isso fazia
`keep_turns=1` estubar o tool result do PRÓPRIO round mais recente — o
oposto do pretendido. Se seu código trazia `prune_tool_results_after = 2`
como contorno para esse bug, `N=1` agora é o valor correto.

Uma janela maior ou igual ao número de rounds completos no histórico não
poda nada (comportamento idempotente para conversas curtas). O corte é feito
por `ToolAgent._prune_history` (staticmethod pura, chamada no início de
`_build_agent`) e NUNCA muta a lista ou as mensagens recebidas — o histórico
do cliente (o que o Server/OnlyOffice/CLI guardam e reenviam) continua
completo; a poda é efêmera, recomputada a cada request a partir do histórico
original. Default `None` desliga a poda inteiramente (zero mudança de
comportamento). Valores `<= 0` (ou não-int) são configuração inválida,
rejeitada no registro da subclasse com `AixonError` — use `None` para
desligar ou um `int >= 1`.

### Hooks de tool call (#17)

Duas sobrescritas opcionais no `ToolAgent` dão um ponto de observação/controle
determinístico em CADA execução de tool, sem depender do modelo — útil para
guardrails de política (bloquear uma tabela proibida, redigir um argumento) e
para telemetria/logging estruturado (captura para evals, auditoria).

```python
class GuardedAgent(ToolAgent):
    llm = LLM("gpt-5.4", temperature=0.2)
    tools = [cropnet_query]

    def on_tool_start(self, name: str, args: dict):
        if name == "cropnet_query" and "tabela_proibida" in args.get("sql", ""):
            raise PermissionError("acesso a 'tabela_proibida' é bloqueado pela política")
        if name == "cropnet_query":
            return {**args, "sql": args["sql"].strip()}  # normaliza antes de rodar
        return None  # mantém os args sem alteração

    def on_tool_end(self, name, args, result, error):
        _log.info(f"tool={name} args={args} error={error!r}")
```

- `on_tool_start(self, name, args)` roda ANTES da tool. Um `dict` de retorno
  REESCREVE os kwargs da chamada (a memoização — #5 — usa a chave já
  reescrita, então uma reescrita determinística compartilha cache entre
  chamadas equivalentes); `None` mantém os args originais. Uma exceção aqui é
  tratada como a PRÓPRIA tool falhando: o shield (#9) converte em um `TOOL
  ERROR` devolvido ao modelo — o run não cai, só aquela chamada reporta erro.
  A exceção é um veredito do gateway (orçamento esgotado, limite de
  requisições): o shield o relança, o mesmo objeto, depois do `on_tool_end`
  (ver `shield_tool_errors`).
- `on_tool_end(self, name, args, result, error)` roda DEPOIS — inclusive em
  cache hit (`error=None`) e em falha da tool (`error` preenchido com a
  exceção, `result=None`). É só observação: qualquer exceção levantada aqui é
  logada como warning e engolida — telemetria nunca corrompe o resultado que
  o modelo recebe.
- Ambos são no-op por padrão (zero mudança de comportamento se você não
  sobrescrever nenhum). São passados para `coerce_tools`/`_guard` só quando a
  subclasse sobrescreve pelo menos um — o caso default não paga custo extra
  por chamada.
- Como os hooks do #9/#5, só se aplicam a entradas `AgentTool`/callable; um
  `BaseTool` cru passado em `tools` continua sem guard (sem shield, sem memo,
  sem hooks).
- Pela mesma razão, os proxies de client tools do `client_tools="merge"|
  "replace"` (#18c, abaixo) TAMBÉM ficam fora do alcance de `on_tool_start`/
  `on_tool_end`: a call de uma tool do cliente nunca executa server-side, então
  não há chamada de tool ali para o hook observar.

### Client tools mesclados no loop (#18c)

`LLMAgent(client_tools="passthrough")` (acima) repassa os tools do cliente crus — o
agente decide sozinho quando responder com `tool_calls`. `ToolAgent` tem um
caminho de primeira classe: `client_tools="merge"` (ou `"replace"`) injeta os
defs do cliente como tools de VERDADE no mesmo loop de tool-calling das tools
internas do agente, com uma diferença crítica — a call de uma tool interna
executa server-side e o loop continua normalmente; a call de uma tool do
CLIENTE encerra o turno na hora (via um proxy `return_direct`, o mesmo
mecanismo da ponte nativa do OnlyOffice) e o run devolve
`Message(role="assistant", content="", tool_calls=[...])`.

```python
class RedatorAgent(ToolAgent):
    llm = LLM("gpt-5.4", temperature=0.2)
    tools = [buscar_no_banco]          # tool interna — executa aqui
    client_tools = "merge"             # + os tools que o editor declarar
    client_tools_conflict = "error"    # default: nome colidindo é erro explícito

    def client_tools_filter(self, defs):
        # curadoria opcional: só expõe tools do cliente com um prefixo esperado
        return [d for d in defs if d["function"]["name"].startswith("doc_")]
```

- `client_tools`: `"ignore"` (default, zero mudança) | `"merge"` (soma aos
  tools internos) | `"replace"` (só os do cliente, para aquele request).
- `client_tools_conflict` resolve um nome que aparece tanto numa tool interna
  quanto num def do cliente: `"error"` (default — `AixonError` já na montagem
  do grafo, citando as tools em colisão), `"internal"` (descarta o def do
  cliente, a interna vence), `"client"` (remove a tool interna daquele
  request, só o proxy do cliente fica exposto sob aquele nome).
- `client_tools_filter(self, defs)` roda ANTES da política de conflito —
  curadoria de quais defs do cliente sequer entram na disputa. Default:
  identidade (todos).
- **Retomada**: o cliente executa a call localmente e faz um novo request com
  `assistant(tool_calls=[...])` + `role="tool"` (o resultado) no histórico —
  o mesmo round-trip neutro que qualquer resultado de tool usa
  (`to_langchain`/`from_langchain`); nada de especial do lado do cliente.
- **Turno misto (interna + cliente na MESMA resposta do modelo)**: o que
  surfaceia é o PRIMEIRO turno do run que chamou uma tool do cliente — as
  calls do cliente desse turno viram `Message.tool_calls`, e as internas do
  mesmo turno já executaram server-side (o `ToolNode` do LangGraph roda
  todas as calls do turno). Como o `return_direct` do LangChain só encerra o
  grafo quando TODAS as calls do turno são return-direct, um turno misto NÃO
  corta o loop: o modelo vê o "resultado" placeholder do proxy e pode gerar
  turnos adicionais — tudo que ele produziu DEPOIS daquela primeira call do
  cliente é descartado (esses turnos custam tokens mas nunca chegam ao
  cliente: o resultado real da tool do cliente ainda não existia, então
  qualquer texto construído sobre o placeholder seria fabricado). No request
  seguinte (com o resultado do cliente no histórico) o modelo replaneja com
  dados reais; uma call interna re-emitida simplesmente roda de novo.
  Recomendação: prompts que induzam o modelo a separar ações do documento
  (tools do cliente) em turno próprio, depois das consultas internas,
  reduzem o desperdício pós-call. Detalhe completo do request/response e
  tabela `client_tools` × `client_tools_conflict`:
  [server.md](server.md#openaiadapter) ("Client tools"). Demo executável:
  `examples/client_tools/merge_demo.py`.
  **EFEITOS COLATERAIS (além de custo/texto fabricado)**: os turnos gerados
  DEPOIS daquela primeira call do cliente são descartados como *resposta*,
  mas qualquer tool INTERNA que o modelo chame nesses turnos EXECUTA de
  verdade — o `ToolNode` do LangGraph roda a call antes de o texto do turno
  ser jogado fora. Uma tool interna com efeito colateral (gravar, exportar,
  notificar) baseada no "resultado" placeholder do proxy dispara mesmo
  assim, e não há como desfazer depois. Mitigação: em agentes com
  `client_tools` ativo, evite combinar tools internas com efeito colateral
  no mesmo `tools`, ou instrua no prompt que ações do cliente (documento)
  fiquem em turno próprio, ANTES de qualquer tool interna que
  grave/exporte/notifique.

### Nesting agents as tools

Any `Agent` exposes itself as a tool via `as_tool()`. The result is a neutral
`AgentTool` — coerced to a LangChain tool inside `ToolAgent` automatically.

```python
from aixon import ToolAgent, LLM

class OrchestratorAgent(ToolAgent):
    llm   = LLM("gpt-4o-mini")
    tools = [
        PlannerAgent().as_tool(description="Break the goal into steps"),
        ResearchAgent().as_tool(),
    ]
```

**Framing the callee as a subagent (`audience="agent"`, #15).** By default
(`audience="human"`, unchanged), the nested agent gets the caller's text
verbatim — as if a person had typed it — and tends to answer accordingly:
greetings, hedging, "let me know if you need anything else." That's fine when
a human really is the ultimate reader, but when the caller is itself an
agent, that human-facing prose is just noise burning the parent's context
window. `as_tool(audience="agent")` appends a fixed suffix
(`aixon.agent._AGENT_AUDIENCE_SUFFIX`) to each call's user text, asking the
callee to answer with dense, structured facts instead of human-facing prose:

```python
CobradorAgent().as_tool(name="clientes", audience="agent")
```

The frame is appended to the **user message text**, never sent as a leading
system message: a leading system message would override the subagent's own
prompt (see the `ToolAgent`/`_build_agent` contract — the leading system
message wins), which would defeat the callee's own instructions instead of
just adding context to them. An invalid `audience` (anything other than
`"human"`/`"agent"`) raises `AixonError` immediately.

**Reasoning propagation:** when a nested agent emits reasoning (via the
`ReasoningChannel`), that reasoning bubbles up through the outer `stream()` as
`Chunk(reasoning=...)` deltas — so callers see the full chain of thought even
across nesting levels.

**Model reasoning.** When `self.llm` has the [reasoning knob](#reasoning-extended-thinking--reasoning-effort)
turned on, a turn's own thinking/reasoning text (extracted per
`reasoning_from_message`, see [architecture.md](architecture.md)) is emitted
into the same `ReasoningChannel` *before* that turn's tool-call label(s) — the
model reasoned before deciding to call the tool, and the channel preserves
that order. `Message.reasoning` (`invoke`) / `Chunk.reasoning` (`stream`)
therefore interleave the model's own thinking with the `"Calling {name}..."`
step labels, in the order they occurred. Consecutive duplicate reasoning
lines are deduplicated the same way as tool-call labels.

---

## ReflectiveAgent — evaluator-optimizer loop

Use `ReflectiveAgent` when a single generation pass isn't reliable enough:
it wraps a worker `Agent` in a review loop — a judge LLM scores each answer
against an objective rubric, and a rejected answer goes back to the worker
together with the judge's critique, up to `max_rounds` attempts.

```python
from aixon import LLM, ReflectiveAgent
from agents.gerente import GerenteAgent

class GerenteRevisadoAgent(ReflectiveAgent):
    name = "gerente-revisado"
    agent = GerenteAgent                 # class OR instance (like Orchestrator nodes)
    judge_llm = LLM("gpt-5.4-mini", temperature=0)
    judge_rubric = (
        "1. Every SQL statement returned was validated (no non-existent column).\n"
        "2. Any number quoted matches what the tools returned.\n"
        "3. The answer addresses the entire question."
    )
    max_rounds = 3
```

**`ReflectiveAgent` attributes:**

| Attribute | Type | Required | Description |
|---|---|---|---|
| `agent` | `Agent` (class or instance) | **Yes** | The worker that produces answers. Resolved once, at `__init__`, with the same `_instantiate` helper `Orchestrator` uses for its nodes. |
| `judge_llm` | `LLM` | **Yes** | The model that grades each answer. Often a cheaper/faster model than the worker's — judging is a classification task, not generation. |
| `judge_rubric` | `str` | **Yes** | Objective approval criteria, non-empty. See "Write an objective rubric" below. |
| `max_rounds` | `int` | No (default `3`) | Worker attempts before giving up, `>= 1`. |
| `revision_mode` | `str` | No (default `"full"`) | `"full"` regenerates the whole answer on a rejected round. `"patch"` (opt-in) asks the retry for SEARCH/REPLACE edit blocks applied programmatically over the previous answer — output-cost saver for long answers; a patch that doesn't apply falls back to full regeneration for that round, and raw patch text never reaches the stream as content. |
| `judge_label` | `str` | No | Reasoning-channel label emitted before each judge call. Default: `"Avaliando a resposta…"`. |
| `retry_label` | `str` | No | Reasoning-channel label emitted before a retry. `{round}`/`{max}` are interpolated. Default: `"Refinando a resposta (rodada {round}/{max})…"`. |
| `exhausted_label` | `str` | No | Reasoning-channel label emitted when `max_rounds` is reached without approval. Default: `"Rodadas esgotadas — entregando a melhor tentativa."`. |
| `patch_fallback_label` | `str` | No | Label emitted when a `"patch"` retry didn't apply and the round falls back to full regeneration. |
| `judge_unavailable_label` | `str` | No | Label emitted when `judge_llm` is unavailable, i.e. raises something other than a gateway verdict (see "Judge indisponível" below). Default: `"Revisão indisponível — entregando sem conferência."`. |

Missing `agent`/`judge_llm`, an empty `judge_rubric`, or `max_rounds < 1` on a
concrete subclass raises `AixonError` at import time — before registration
(the same validate-before-register precept as every other subtype), so a
misconfigured `ReflectiveAgent` never leaves a ghost entry in the registry.

**How it works — the loop:**

1. `invoke` runs the worker (`agent.invoke`) to get a first answer.
2. `emit_reasoning(judge_label)`, then the judge grades it: `judge_llm.complete`
   is called with the rubric and the question/answer pair.
3. The verdict is a text sentinel, following the `DELEGAR`/`END` precedent: if
   its first line (after `strip()`) is exactly `APROVADO`, the answer is
   returned as-is.
4. Otherwise the verdict IS the critique. If rounds remain,
   `emit_reasoning(retry_label)` and the worker is re-invoked with the
   critique appended to the conversation (a new message list — the caller's
   is never mutated).
5. If `max_rounds` is reached without an `APROVADO`, `exhausted_label` is
   emitted and the **last attempt is returned** — exhausting the rounds is
   *not* an exception. A quality shortfall must not crash a run that produced
   an answer; the caller decides what to do with a possibly-imperfect result.

Between steps 1 and 2, `should_judge` decides whether the loop runs at all
(see below) — when it returns `False` the worker's first answer is returned
immediately, before any judge call.

`stream`/`astream` mirror `Orchestrator`: they run the loop under a fresh
reasoning channel, drain it as `Chunk(reasoning=...)` deltas, then yield the
final `Chunk(content=...)` and `Chunk(done=True)`. `ainvoke`/`astream` are
native (`agent.ainvoke` + `judge_llm.acomplete`), not thread-bridged.

**Cost and latency.** Each round re-runs the worker (and, on rejection, a
fresh judge call) — but since 0.1.19 the retries are far cheaper than naive
reruns, automatically:

- **Prompt caching** — retries only APPEND messages (the prefix is
  byte-identical across rounds, guaranteed by test), so OpenAI's automatic
  prompt caching bills the repeated prefix as cache hits. For Anthropic
  workers/judges, opt in with `LLM(..., cache=True)` (explicit `cache_control`
  breakpoints on the system + last message; each round's breakpoint becomes
  the next round's cached prefix).
- **Tool-call memoization** — a retry that re-issues a tool call with the same
  arguments (same DB query, same web search) reuses the first result instead
  of re-executing (see the ToolAgent section; opt-out per tool with
  `as_tool(memoize=False)` — recommended for write tools).
- **Predicted Outputs (OpenAI)** — the previous attempt is sent as the
  `prediction` on retries, so unchanged spans regenerate by speculative
  decoding (latency win; rejected predicted tokens still bill as output).
  Other providers ignore it.
- **`revision_mode = "patch"`** (opt-in) — retries emit SEARCH/REPLACE edits
  instead of rewriting the whole answer (output-cost saver for long answers),
  with automatic fallback to full regeneration when a patch doesn't apply.

Keep `max_rounds` as low as the rubric allows, and prefer a cheap `judge_llm`.

**Write an objective rubric.** `judge_rubric` should state checkable facts,
not vibes — "does it cite a source?", "do the numbers match the tool
results?", "is every requested field present?". A vague rubric ("sounds
right", "is helpful") degenerates into the judge approving on the first pass
regardless of quality, defeating the point of the loop.

A complete runnable example (scripted judge + worker, no API key needed) is
at [examples/reflective_review](../examples/reflective_review).

### should_judge — skipping the judge for cheap answers (#14)

```python
def should_judge(self, messages: list[Message], answer: Message) -> bool: ...
```

Override this method on your subclass to gate the review loop per answer.
Default is `True` — every answer goes through the judge, the historical
behavior. Return `False` and the worker's answer is returned as-is: **no**
`judge_llm` call, no retry, nothing logged (see the `reflective_run` note
below). Typical use: not every worker answer deserves a paid judge call —

```python
def should_judge(self, messages, answer):
    return len(answer.content) > 200   # saudações não pagam juiz
```

Latency note: a reasoning `judge_llm` (e.g. a "thinking" model) can cost more
wall-clock time than the worker call itself — a plain greeting doesn't need
to wait on that. On `stream`/`astream`, the gate is only consulted on the
**first** round (a retry's candidate answer is always judged, since it only
exists because a previous round was rejected).

### Medindo a taxa de fallback do modo patch (#12)

Cada run emite uma linha estruturada no logger `aixon.reflective`:

    reflective_run agent=Uniplus-DB rounds=2 patch_applied=1 patch_fallback=0 outcome=approved

Taxa de fallback = `patch_fallback / (patch_applied + patch_fallback)` agregada
por agente. Em Cloud Run/Logging: filtre por `reflective_run` e agrupe por
`agent=`. `revision_mode="full"` loga a mesma linha com `patch_*=0`. Essa
métrica decide a promoção de `patch` a default numa futura 0.2.0.

Runs pulados pelo gate `should_judge` (#14) não geram linha `reflective_run` —
não houve loop de julgamento a medir.

### Judge indisponível degrada graciosamente (#24)

Qualquer exceção levantada por `judge_llm.complete`/`acomplete` — em
qualquer rodada, inclusive numa rodada de retry — não derruba o run: a
resposta ATUAL do worker (a que estava sendo avaliada quando o juiz falhou)
é entregue sem revisão, com `judge_unavailable_label` emitido no canal de
reasoning e um `warning` no logger `aixon.reflective`. O `reflective_run`
dessa execução loga `outcome=judge_error` (linha única por run, como os
outros outcomes). Mesmo preceito do módulo: uma falha de qualidade — aqui, a
própria avaliação de qualidade — não pode derrubar um run que já produziu
uma resposta. Sem mudança de comportamento quando o juiz funciona.

**Exceção: veredito do gateway.** Orçamento esgotado ou limite de requisições
do gateway (`classify_gateway_error`) não é um juiz indisponível nem uma falha
de qualidade: é o fim da request, e uma resposta entregue "sem revisão"
esconderia o veredito de quem precisa respondê-lo (o `Server` o transforma na
mensagem de cota). O erro é relançado, o mesmo objeto, antes de qualquer log,
label ou linha `reflective_run`, nos 4 caminhos (`invoke`, `ainvoke`, `stream`,
`astream`) e também para o worker: os handlers do #19 (o timeout ou o limite de
recursão do worker numa rodada de retry, inclusive nas de patch) o relançam em
vez de entregá-lo como conteúdo. Só conta o erro cuja request foi à
`AIXON_GATEWAY_BASE_URL`: um 429 da OpenAI num juiz que roda direto nela segue
sendo um juiz indisponível e degrada como descrito acima.

---

## Agent.as_tool — the neutral tool descriptor

```python
@dataclass
class AgentTool:
    name: str
    description: str
    func: Callable[[str], str]
    coroutine: Callable[[str], Awaitable[str]] | None = None  # optional async path
```

```python
tool = agent.as_tool()
tool = agent.as_tool(name="planner", description="Decomposes goals")
tool = agent.as_tool(audience="agent")  # #15 — see "Nesting agents as tools"
```

`func` drives `agent.stream()` (and `coroutine` drives `agent.astream()`, #22)
to completion: each call creates a fresh `[Message(role="user",
content=text)]` — the agent's state never leaks between tool calls. Every
yielded `Chunk.content` is joined into the returned string — byte-identical
to what `invoke()`/`ainvoke()` would have returned **for a normal answer**.
A worker that blows its OWN `max_execution_time` with nothing accumulated is
the one case where this diverges: `stream()`/`astream()` emit `timeout_message`
as CONTENT instead of dying mute (#19 — direct callers must never get an
empty answer), but `as_tool()` cannot let that look like a legitimate
subagent response reaching the parent model as fact. It detects a final
content that byte-matches the wrapped agent's own formatted
`timeout_message` (or, through a `ReflectiveAgent`'s pass-through, its
wrapped worker's) and raises `AixonError` instead — the parent's tool-call
shield (#9) turns that into a visible `TOOL ERROR` result, restoring the
pre-#22 semantics where the same timeout reached `as_tool()` via a raising
`invoke()`/`ainvoke()` (#22/#24-sweep — see CHANGELOG `[Unreleased]`). Every
yielded `Chunk.reasoning` is re-emitted **live**, right away, onto the
CALLER's active `ReasoningChannel` — not buffered until the whole subagent
run finishes. That means a nested agent's tool-call labels, judge lines and
retries flow through the parent's stream in real time, so the #20 drain has
something to drain while a slow nested tool is still running instead of the
caller looking stuck for the subagent's entire wall-clock time. `as_tool()`
sets both `func` and `coroutine`: `coerce_tools` registers them dual, and the
tool runs on the sync (`invoke` → `func`) and async (`ainvoke` → `coroutine`)
parent paths. The same `AgentTool` shape is returned by `Retriever.as_tool()`,
so `ToolAgent.tools` handles both uniformly.

**Gateway verdicts cross the tool.** A budget or rate-limit verdict of the
gateway (`classify_gateway_error`) raised anywhere inside the wrapped agent is
not a failing tool but the end of the request: it propagates out of
`func`/`coroutine` and through the parent's shield (#9), which re-raises it
instead of handing the model a `TOOL ERROR`, and through every `as_tool` level
above, so the `Server` answers the whole request with its quota message. A
`ReflectiveAgent` judge or worker inside the subagent behaves the same way.
Only an error from a request to `AIXON_GATEWAY_BASE_URL` counts: any other
exception that escapes the subagent, a 429 from another provider included,
still reaches the parent as a `TOOL ERROR` (shield on).

A stream that yields BOTH content and tool_calls in the same run is not
expected from any built-in `stream()`/`astream()` (they yield either
content, or a `tool_calls`-without-content client-tool leak, never both) —
if a custom one does, `as_tool()` keeps the content and silently drops the
tool_calls (there is no channel to relay them back through a tool result),
logging a warning on `aixon.agent` naming the discarded call(s) so the drop
is visible in telemetry instead of a silent data loss.

A cache hit (`aixon.toolcache`, `memoize=True` default) is served by
`coerce_tools`' wrapper before `func`/`coroutine` ever run — so a memoized
repeat call produces **no** reasoning at all (the subagent's `stream()` never
executes a second time). That silence is correct: re-running for a result
you're about to throw away would burn the wall-clock time this feature exists
to avoid surfacing.

`audience` (default `"human"`, zero behavior change) controls the framing of
the text handed to the callee: `"agent"` appends the subagent frame
(`_AGENT_AUDIENCE_SUFFIX`) so the callee answers with dense facts for another
agent instead of human-facing prose. See "Nesting agents as tools" above for
the rationale and an example. Any other value raises `AixonError`.

### Structured briefing — `args_schema` + `render_input` (#23)

By default a nested agent's tool exposes a single free-text argument to the
calling model, and `func`/`coroutine` receive it positionally
(`text: str`) — the caller's LLM writes one blob of prose, and that's exactly
what the subagent gets. Pass `args_schema` (a neutral JSON-Schema dict — the
same shape `AgentTool.args_schema` already accepts from MCP tools) to expose
STRUCTURED fields instead:

```python
CobradorAgent().as_tool(
    name="cobranca",
    args_schema={
        "type": "object",
        "properties": {
            "objetivo": {"type": "string", "description": "O que resolver"},
            "contexto": {"type": "string", "description": "Contexto já apurado"},
        },
        "required": ["objetivo"],
    },
)
```

With `args_schema` set, the calling model fills in `objetivo`/`contexto` as
separate typed fields (visible on the coerced tool as
`StructuredTool.args_schema`/`.args`) instead of composing one string —
sturdier for a multi-field briefing than hoping the model's free text mentions
everything. `func`/`coroutine` then receive those fields as `**kwargs`, and
`render_input(kwargs) -> str` turns them into the actual text the subagent
reads. The **default** `render_input` (used whenever the argument is
omitted) renders one `"FIELD: value"` line per kwarg, key upper-cased, in
insertion order, dropping any field whose value is `None`/`""`/`False`:

```python
{"objetivo": "revisar contrato", "contexto": "cliente ACME"}
# -> "OBJETIVO: revisar contrato\nCONTEXTO: cliente ACME"
```

Pass your own `render_input` to control the wording instead:

```python
CobradorAgent().as_tool(
    name="cobranca",
    args_schema=SCHEMA,
    render_input=lambda kw: f"Resolva: {kw['objetivo']} (contexto: {kw.get('contexto', '—')})",
)
```

Composition with the rest of `as_tool` is unchanged: `audience="agent"`
appends its suffix to the ALREADY-rendered text (the frame wraps the
briefing, not a raw field), and the call still runs through the same
`stream()`/`astream()`-driven `func`/`coroutine` from #22 — internal
reasoning still flows live to the caller's channel regardless of whether the
tool takes free text or a structured schema. Without `args_schema` (the
default, `None`), everything above is inert — the tool keeps taking a single
positional `text` argument exactly as before.

---

## Suffix rule reference

| Base class | `_suffix` | Valid example | Invalid (raises `NamingError`) |
|---|---|---|---|
| `LLMAgent` | `"Agent"` | `PlannerAgent` | `Planner`, `PlannerLLM` |
| `ToolAgent` | `"Agent"` | `ResearchAgent` | `Research`, `ResearchTool` |
| `Orchestrator` | `"Orchestrator"` | `SupportOrchestrator` | `Support`, `SupportAgent` |

**Abstract subtypes** (your own base classes) bypass the suffix check by passing
`abstract=True`. Their concrete subclasses are then validated:

```python
class BaseSupportAgent(ToolAgent, abstract=True):
    llm   = LLM("gpt-4o-mini")
    tools = [check_ticket]

class BillingAgent(BaseSupportAgent):     # valid: ends with "Agent"
    prompt = "You handle billing issues."

class TechAgent(BaseSupportAgent):        # valid
    prompt = "You handle technical issues."
```

---

## Invoke and stream examples

```python
from aixon.message import Message

# invoke — returns a Message
reply = PlannerAgent().invoke([Message(role="user", content="Plan a product launch")])
print(reply.content)

# stream — yields Chunk deltas
for chunk in ResearchAgent().stream([Message(role="user", content="Latest on LLMs")]):
    if chunk.reasoning:
        print("[reasoning]", chunk.reasoning)
    elif chunk.content:
        print(chunk.content, end="", flush=True)
```

## Async — `ainvoke` / `astream`

Every agent also exposes async methods. **Sync is the default; async is purely
additive** — existing sync code is untouched, and you opt into async only where
you want it.

```python
reply = await PlannerAgent().ainvoke([Message(role="user", content="Plan a launch")])

async for chunk in ResearchAgent().astream([Message(role="user", content="...")]):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

- `LLMAgent`, `ToolAgent` and `Orchestrator` implement `ainvoke`/`astream`
  **natively** over LangGraph's async path (`ainvoke`/`astream`), so they never
  block the event loop.
- A purely sync custom `Agent` (one that only implements `invoke`/`stream`)
  still gets working `ainvoke`/`astream` for free — the base bridges them to a
  worker thread.
- The neutral types are unchanged: `ainvoke` returns a `Message`, `astream`
  yields `Chunk`s.

**Async tools.** A `ToolAgent` tool may be an `async def` callable — it runs on
the async path (`ainvoke`/`astream`) and does real non-blocking I/O (e.g. an MCP
call via `Connector.aget`). An async tool requires that path: calling it from
sync `invoke` raises `NotImplementedError` (it is never silently skipped). Sync
tool callables work on **both** paths (under `ainvoke` they run in a thread
executor). So: use **sync** tools if you need the agent to work via both `invoke`
and `ainvoke`; use **async** tools when you commit to the async path and want
non-blocking I/O.

**Real timeouts (cancellation).** On the async path, `ToolAgent.max_execution_time`
and `Orchestrator.timeout` wrap the run in `asyncio.wait_for`, so an overrun is
**cancelled at the next await point** — provided the chain is genuinely async
(an async model, async tools). Sync work bridged to a thread cannot be
interrupted mid-call; bound that at the tool/IO layer (e.g. `Connector.timeout`).
The server (`docs/server.md`) awaits `ainvoke`/`astream`, so concurrent requests
no longer serialize.

---

## Registry helpers

```python
from aixon import get_registry

registry = get_registry()
registry.public()           # list of non-hidden agents
registry.all()              # every registered agent
registry.resolve("planner") # by name or alias
```

Agents with `hidden = True` remain callable but are excluded from `public()` and
the `aixon chat` selection menu.

---

## See also

- [Architecture overview](architecture.md) — how agents, retrievers, and the server compose
- [Retrieval](retrieval.md) — `Retriever.as_tool()` and the same `AgentTool` contract
