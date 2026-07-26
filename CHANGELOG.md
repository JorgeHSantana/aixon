# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.21] - 2026-07-26

### Added
- **Client tools de ponta a ponta — passthrough, transporte de `tool_choice`
  e merge de primeira classe no `ToolAgent` (#18a+#18b+#18c).**
  - **#18a** — `LLM.complete`/`acomplete`/`stream`/`astream` aceitam
    `tools`/`tool_choice`. `tools` no shape wire OpenAI
    (`{"type": "function", "function": {...}}` — o mesmo que
    `ParsedRequest.tools` já normaliza) é bindado no chat model via
    `model.bind_tools(tools, **extra)` só para aquela chamada; `tool_choice`,
    quando dado, vai junto. `stream`/`astream` acumulam os `tool_calls` do
    modelo ao longo dos chunks (via `AIMessageChunk` somável) e emitem
    `Chunk(tool_calls=[...])` — shape neutro `{"name", "args", "id"}` — antes
    do `Chunk(done=True)`. Sem `tools`, o caminho é byte-idêntico ao anterior.
    `LLMAgent.client_tools: bool = False` — quando `True`, lê
    `current_client_tools()`/`current_tool_choice()` (`aixon/runtime.py`) e
    repassa aos 4 métodos (passthrough cru — o agente decide sozinho quando
    responder com `tool_calls`); sem tools declaradas no request, é um no-op.
  - **#18b** — `tool_choice` do adapter OpenAI virou campo de transporte, não
    passthrough descartado: saiu do params-passthrough e entrou em
    `_TRANSPORT_FIELDS`/`ParsedRequest.tool_choice`; o `Server` publica
    `tool_choice_scope(pr.tool_choice)` (novo contextvar, espelhando o par
    `client_tools`/`current_client_tools()` já existente) junto de
    `client_tools(pr.tools)` nos dois caminhos (sync e stream), então
    `LLMAgent(client_tools=True)` e qualquer agente que leia
    `current_tool_choice()` agora o recebem de fato. Adapter Anthropic fica
    de fora desta task — dialeto de `tool_choice` da Anthropic tem shape
    próprio (`{"type": "auto"|"any"|"tool", "name": ...}`); registrado como
    follow-up da issue #18.
  - **#18c** — `ToolAgent(client_tools="ignore"|"merge"|"replace")`: caminho
    de primeira classe onde os defs do cliente entram no MESMO loop de
    tool-calling das tools internas do agente, via um proxy `return_direct`
    por def (`_client_proxy_tools`, mesmo mecanismo da ponte nativa do
    OnlyOffice; o corpo do proxy é um stub — a tool do cliente nunca executa
    server-side). Uma call de tool INTERNA executa server-side e o loop
    continua normalmente; quando o modelo chama uma tool do CLIENTE, o run
    devolve `Message(role="assistant", content="", tool_calls=[...])` com
    usage somado do run inteiro; em `stream`/`astream`,
    `Chunk(tool_calls=...)` seguido de `Chunk(done=True)`, sem content. A
    detecção pós-run (`_surface_client_calls`, staticmethod pura —
    mensagens novas do run + set de nomes como PARÂMETROS, nenhum estado de
    request na instância singleton do registry, então requests concorrentes
    não se corrompem) surfaceia o PRIMEIRO turno do run que chamou tool do
    cliente. Turno misto (interna + cliente na MESMA resposta do modelo):
    as internas do turno já executaram server-side; como o `return_direct`
    do LangChain só corta o grafo quando TODAS as calls do turno são
    return-direct, o modelo pode gerar turnos após o sentinel do proxy —
    esses turnos pós-sentinel são DESCARTADOS (custo gasto, mas nenhuma
    resposta fabricada sobre um resultado que ainda não existia chega ao
    cliente); prompts que separem ações do documento em turno próprio
    reduzem o desperdício. Novo `client_tools_conflict: str = "error"`
    resolve colisão de nome entre uma tool interna e um def do cliente:
    `"error"` (default — `AixonError` já na montagem, citando as tools em
    colisão), `"internal"` (descarta o def do cliente) ou `"client"` (remove
    a tool interna daquele request). Novo hook
    `client_tools_filter(self, defs) -> defs` (default identidade) para
    curadoria antes da política de conflito. Retomada: a request seguinte
    traz `assistant(tool_calls=...)` + `role="tool"` no histórico — já
    coberto pelo `to_langchain`/`from_langchain` existentes, sem mudança no
    `_interop` (e o scan restrito às mensagens NOVAS do run garante que a
    call já executada do histórico não re-surfaceia). Ambos os novos
    atributos são validados em `_validate_subclass` (valor fora do enum →
    `AixonError` no registro). Documentado em `docs/agents.md` ("Client
    tools mesclados no loop (#18c)") e `docs/server.md` ("Client tools", com
    a tabela `client_tools` × `client_tools_conflict`); demo executável
    `examples/client_tools/merge_demo.py`.

  **Nota de migração: nenhuma** — tudo opt-in (`client_tools="ignore"` e
  `LLMAgent.client_tools=False` são os defaults, byte-idênticos ao
  comportamento anterior). Única mudança de comportamento observável:
  `tool_choice` sem `tools` no mesmo request agora responde `400` (antes:
  descarte silencioso).
- **`ToolAgent`: hooks `on_tool_start`/`on_tool_end` no loop de tool-calling (#17).**
  Duas sobrescritas opcionais, no-op por padrão (zero mudança de
  comportamento). `on_tool_start(self, name, args)` roda ANTES de cada tool
  call, dentro do error shield (#9): um `dict` de retorno REESCREVE os kwargs
  (a memoização #5 usa a chave já reescrita); uma exceção levantada aqui é
  tratada como a própria tool falhando (`TOOL ERROR` devolvido ao modelo, run
  não cai). `on_tool_end(self, name, args, result, error)` roda DEPOIS —
  inclusive em cache hit e em falha (`error` preenchido) — e é
  observação-only: exceções são logadas como warning e engolidas, nunca
  corrompem o resultado. `coerce_tools`/`_guard` (`aixon/_interop/tools.py`)
  ganham `on_tool_start`/`on_tool_end`/`on_start`/`on_end` como kwargs
  opcionais (default `None`, comportamento byte-idêntico); `ToolAgent._build_agent`
  só repassa os hooks quando a subclasse sobrescreve pelo menos um. Como o
  shield/memo, só cobre entradas `AgentTool`/callable — `BaseTool` cru
  continua sem guard. Casos de uso: guardrails determinísticos (bloquear
  tabela proibida, normalizar argumento) e telemetria/logging estruturado
  (captura para evals). Documentado em `docs/agents.md` ("Hooks de tool call
  (#17)").
- **`ToolAgent`: poda opt-in de tool results antigos do payload (#16).** Novo
  atributo `prune_tool_results_after: int | None = None` (default `None`,
  zero mudança de comportamento). Com um `int N`, mensagens `role="tool"`
  ANTES das últimas `N` mensagens `role="assistant"` do histórico têm o
  content substituído por um stub curto (`[resultado de ferramenta omitido
  (M caracteres) — já utilizado em resposta anterior]`) só no payload enviado
  ao provider — o histórico do cliente não é tocado. Helper puro e testável
  `ToolAgent._prune_history(messages, keep_turns)` (staticmethod, nunca muta a
  lista/mensagens do chamador), chamado no início de `_build_agent`. Útil para
  agentes de banco (`CropnetDB`, `sql_static`) cujas queries tendem a devolver
  payloads grandes que já foram consumidos e resumidos em turnos anteriores.
  Documentado em `docs/agents.md` ("Poda de tool results antigos (#16)").
- **`Agent.as_tool(audience="agent")`: moldura de subagente opt-in (#15).**
  Novo parâmetro `audience` (default `"human"`, zero mudança de
  comportamento). Com `audience="agent"`, cada chamada anexa um sufixo fixo
  (`aixon.agent._AGENT_AUDIENCE_SUFFIX`) ao texto do usuário enviado ao
  subagente, pedindo fatos densos e estruturados em vez de prosa
  humano-orientada — reduz ruído no contexto do agente pai quando o callee é
  outro agente, não uma pessoa. O frame vai como sufixo do texto do usuário,
  NUNCA como mensagem system (uma system à frente sobrescreveria o prompt do
  próprio subagente). Valor inválido de `audience` levanta `AixonError` com
  mensagem clara. Exemplo em `examples/tool_shield_memo/main.py`; documentado
  em `docs/agents.md` ("Nesting agents as tools").
- **`ReflectiveAgent`: gate `should_judge` — juiz opcional por resposta (#14).**
  Novo método sobrescrevível `should_judge(self, messages, answer) -> bool`
  (default `True`, zero mudança de comportamento). Retornar `False` devolve a
  resposta do worker direto, SEM nenhuma chamada ao `judge_llm` e sem retry —
  cobre os 4 caminhos (`invoke`/`stream`/`ainvoke`/`astream`; nos streams, o
  gate só é consultado na primeira rodada). Um run pulado pelo gate NÃO emite
  a linha `reflective_run` do #12 (não houve loop a medir). Uso típico: nem
  toda resposta do worker merece pagar um juiz — ex. saudações curtas.
  Exemplo em `examples/reflective_review` (`GatedReviewedWriterAgent`);
  documentado em `docs/agents.md`.
- **Tool calls do mesmo turno em paralelo, garantido por teste (#13).** Spike
  confirmou que o `ToolNode` interno do `create_agent` (langchain 1.x /
  langgraph 1.2) já paraleliza tool calls emitidas num mesmo turno do modelo
  nos DOIS caminhos — async (`ainvoke`/`astream`, o usado pelo `Server`) via
  `asyncio.gather`, e sync (`invoke`/`stream`) via thread pool
  (`get_executor_for_config().map`) — sem nenhuma mudança em
  `aixon/agents/tool_agent.py`. Novo teste de regressão
  `tests/test_tool_parallel.py` fixa o caminho async por timing (2 tools de
  0.4s completam em ~0.4s, não ~0.8s). Documentado em `docs/agents.md`
  ("Paralelismo de tool calls").
- **`ReflectiveAgent`: log estruturado por run (#12).** Cada run (`invoke`,
  `stream`, `ainvoke`, `astream`) emite uma linha `reflective_run agent=<name>
  rounds=<n> patch_applied=<n> patch_fallback=<n> outcome=<approved|exhausted>`
  no logger `aixon.reflective` — grep-friendly para medir a taxa de fallback
  do `revision_mode="patch"` antes de promovê-lo a default. Zero mudança de
  comportamento.

### Docs
- Backfill de exemplos: `examples/providers_grok/` (provider xAI/Grok) e
  apontador para o debug tap (`AIXON_DEBUG_REQUESTS`) em
  `examples/tracing/README.md`.

## [0.1.20] - 2026-07-23

### Added
- **Provider xAI (Grok).** `LLM("grok-4.5")` resolve e constrói sozinho —
  novo provider `xai` (OpenAI-compatível em `https://api.x.ai/v1`, override
  via `XAI_BASE_URL`), chave `XAI_API_KEY` obrigatória (erro claro em vez de
  vazar `OPENAI_API_KEY` para o endpoint), regra de inferência `^grok`,
  knob `reasoning` traduzido para `reasoning_effort` verbatim, extra
  `aixon[xai]`.

## [0.1.19] - 2026-07-19

Eficiência do loop reflexivo (evaluator-optimizer) + robustez de tools. Na
prática: rodadas de retry ficam mais baratas e mais rápidas sem perder rigor,
e falha de infraestrutura numa tool vira mensagem explicada em vez de derrubar
o run com erro opaco.

### Added
- **Tool error shield (#9)**: qualquer exceção levantada por uma tool
  (AgentTool/callable) vira um resultado `TOOL ERROR (...)` legível devolvido
  ao modelo (`str(e) or repr(e)` — cobre `httpx.ReadTimeout`, cujo `str()` é
  vazio); o agente relata a indisponibilidade e/ou prossegue. Opt-out estrito
  por agente: `ToolAgent.shield_tool_errors = False`. `BaseTool` cru passa sem
  shield (documentado em `coerce_tools`)
- **Memoização de tool calls por request (#5)** — `aixon.toolcache`
  (ContextVar): dentro de uma request servida (e de um run do
  ReflectiveAgent), tool chamada de novo com os MESMOS argumentos devolve o
  primeiro resultado sem re-executar; erros nunca são cacheados; cache morre
  com a request. Opt-out por tool: `as_tool(memoize=False)` (Agent e
  Retriever) ou atributo `aixon_memoize = False` num callable
- **Prompt caching entre rodadas (#4)**: teste fixa que retries só ACRESCENTAM
  mensagens (prefixo byte-idêntico — o caching automático da OpenAI aplica);
  `LLM(..., cache=True)` marca `cache_control` (system + última mensagem) em
  providers com `supports_prompt_cache` (Anthropic) para caching incremental
  por rodada
- **Predicted Outputs no retry (#6)**: o ReflectiveAgent publica a resposta
  anterior via `prediction_scope` e o LLM anexa `prediction` na invocação
  quando o provider declara `supports_prediction` (OpenAI) — trechos
  inalterados regeneram por decodificação especulativa (ganho de latência);
  demais providers ignoram
- **`revision_mode = "patch"` no ReflectiveAgent (#7, opt-in)**: o retry emite
  blocos SEARCH/REPLACE aplicados programaticamente sobre a resposta anterior
  (economia de output em respostas longas); patch que não casa → fallback
  automático para regeneração completa; texto de patch nunca vaza como
  content. Default `"full"` byte-idêntico ao comportamento anterior

## [0.1.18] - 2026-07-15

### Added
- Debug tap com **allowlist de agentes**: `AIXON_DEBUG_REQUESTS` aceita, além
  de `1`/`true`/`yes` (grava tudo), uma lista `"Agente1,Agente2"` — só esses
  agentes são gravados, contendo o blast radius em servidores compartilhados
  (agentes com conversas sensíveis nunca tocam o disco)

## [0.1.17] - 2026-07-15

### Added
- **`Agent.thought_mode`** (`"custom" | "content" | "hidden"`): modo de
  reasoning POR AGENTE, com precedência request `thought_stream_mode` >
  agente > `default_thought_mode` do adapter. Protege clientes programáticos
  (parsers de protocolo, plugins de editor) do `<think>` no content sem mudar
  o default do servidor voltado a chat UIs. Não-stream honra só modos
  explícitos (request/agente) — shape histórico preservado
- **Debug tap de requests** (`AIXON_DEBUG_REQUESTS=1`): 1 registro JSONL por
  POST de chat (body verbatim + agente resolvido + resposta/linhas SSE) em
  `AIXON_DEBUG_REQUESTS_DIR` (default `./aixon-debug/`); headers nunca
  gravados; falha do tap nunca derruba a request; zero overhead desligado

## [0.1.16] - 2026-07-14

### Added
- `MCPConnector`: MCP servers as declarative tool sources — catalog discovery with `include`/`exclude`, sync/async calls, `isError` → `AixonError`, `as_tools()`/`aas_tools()`, and the deferred `toolset()` marker (zero I/O at class-body/import time; discovery runs lazily at the first agent invoke, so an unreachable MCP server can never fail server boot). New `[mcp]` extra and offline `examples/mcp_tools/` demo
- `AgentTool.args_schema`: tools may publish a JSON schema — the LLM sees the server's own contract (`**kwargs`) instead of a free-text wrapper; existing schema-less tools are byte-for-byte unchanged

### Fixed
- MCP review wave: `as_tools()` raises a clean `AixonError` inside a running event loop (was a bare RuntimeError + leaked coroutine); connector `timeout` is now passed through to the transport; catalog cache is exactly-once under concurrent first-use without holding a lock across awaits (same-loop deadlock avoided)

### Docs
- `RAG_KNOWLEDGE_BASE.md` refreshed from 0.1.1 to 0.1.15+ (ReflectiveAgent, usage tracking, reasoning, client tools, MCP)

## [0.1.15] - 2026-07-13

### Added
- `LLM(model, reasoning=...)`: declarative reasoning/extended-thinking knob (`None`/`False` off — byte-for-byte unchanged behavior; `True` ≡ `{"effort": "medium"}`; `dict` with `budget_tokens`/`effort`, normalized low=1024/medium=4096/high=16384) translated per provider — Anthropic `thinking` (temperature forced to 1 with a warning, `max_tokens` raised to fit the budget), OpenAI `reasoning_effort`, z.AI/GLM `extra_body.thinking`, Google `thinking_budget`/`include_thoughts` (graceful degradation + warning on an older `langchain-google-genai`); a custom provider without `supports_reasoning = True` has the knob ignored with a warning instead of a broken build (R1)
- Reasoning extraction: Claude `thinking` blocks and the `reasoning_content` convention (zai/GLM) surface on `Message.reasoning` (non-stream) and `Chunk.reasoning` (stream — reasoning delta yielded before the content delta of the same chunk); `to_langchain` does not reconstruct thinking blocks on the way back in — the internal LangGraph loop keeps native provider messages across turns instead (R2)
- `ToolAgent` emits a turn's own model reasoning into the live `ReasoningChannel` *before* that turn's tool-call label(s), so `Message.reasoning`/`Chunk.reasoning` carry the model's thinking ahead of the "Calling {name}..." steps it led to; per-request `reasoning_effort` (allow-listed generation param) overrides the class-level `reasoning=` knob for that one build (R3)

### Docs
- Documented the reasoning knob, the per-provider translation table, and the honesty notes: OpenAI's API returns no raw chain-of-thought (`reasoning_effort` only improves the answer — visible reasoning text comes from Anthropic, and Gemini with `include_thoughts`), the installed `langchain-openai` does not yet populate `reasoning_content` from Chat Completions (so GLM reasoning text doesn't surface despite thinking being enabled — a provider-side gap), and thinking/reasoning tokens bill as output tokens already counted in `Message.usage` (R4)

## [0.1.14] - 2026-07-13

### Added
- Full client-tools round-trip on the Anthropic dialect: responses emit `tool_use` blocks (non-stream and stream, `stop_reason: "tool_use"`), and `tool_result`/`tool_use` history blocks are parsed back into neutral form (N1)
- `aixon.usage` module (`merge_usage`, thread-safe `UsageAccumulator`, `usage_scope`): Orchestrator runs now report usage summed over EVERY model turn (Tier-1 supervisor + all workers, Tier-2 nodes), and ReflectiveAgent sums worker + judge turns across retries (N2)
- Regression tests: Anthropic provider builds without `ANTHROPIC_API_KEY`; accumulator thread-safety under fan-out contention (N2)

### Changed
- `publish.yml` now runs the bare-install smoke job and gates publishing on it — a broken bare install can no longer reach PyPI (N2)

### Fixed
- Usage totals no longer mutate the worker's returned `Message` in place (copies via `dataclasses.replace`) and no longer drop turns under LangGraph's threaded fan-out (`threading.Lock`) (N2)

## [0.1.13] - 2026-07-12

### Added
- Real provider usage tracking via `Message.usage` — provider-reported usage wins on non-streaming responses; the tiktoken estimate remains the fallback when the provider reports none (and for streaming) (M1)
- Production stream session support for Anthropic with interleaved blocks and error envelope closing (M3)
- mypy CI gate in both workflows plus a bare-install smoke job on PRs; `[tool.mypy]` config in pyproject.toml (M4)

### Changed
- `ParsedRequest.tools` is now always OpenAI-shaped: the Anthropic adapter normalizes inbound tool defs, so `current_client_tools()` is dialect-neutral (M2)

### Fixed
- Anthropic provider no longer passes `api_key=None` to ChatAnthropic when the env var is unset (raised a pydantic ValidationError) — pre-existing bug surfaced by the M4 mypy gate

## [0.1.12] - 2026-07-10

### Fixed
- **Server (S1-S11):** Parse 400s on malformed tool arguments and unknown roles; developer role mapping; anthropic SSE envelope handling
- **Agents (A1-A6):** Request params handling in ToolAgent, label history, client system message, supervisor collision, client_tools deepcopy, duplicate tool deduplication
- **Vendors (P1, R1-R6, C1-C2):** zai key guard, weaviate race conditions and stale chunks, Ragie merge order for awrite, connector pooling and kwargs
- **Weaviate (R3):** Purge is best-effort (warn instead of fail); empty string source_id treated as no-source in deduplication
- **Infrastructure (I1-I8):** Scaffold buildability, click core, dev extra completeness, logging deduplication, serve autodiscovery, reasoning stderr, orphaned history
- **CLI (I8 follow-up):** Errored turns return None instead of partial assistant message
- **Final-review wave:** Loop-affine async client, developer role parity, request-model cache, anthropic parse guard

## [0.1.11] - 2026-07-09

### Fixed
- Ragie write now uses `data` field (Ragie SDK 2.0) instead of `content`

## [0.1.10] - 2026-07-09

### Fixed
- Reflective agent streaming now provides live reasoning output during worker loop; reasoning chunks stream immediately while attempt content is buffered
- Removed httpx2 deprecation in dev dependencies (TestClient compatibility with starlette)

## [0.1.9] - 2026-07-09

### Added
- z.AI (GLM) provider support via OpenAI-compatible endpoint
- Client-declared tools on the wire (tool_calls)
- Configurable `default_thought_mode` on OpenAIAdapter

### Fixed
- z.AI build() now returns pure ChatOpenAI (BaseChatModel contract)

## [0.1.7] - 2026-07-07

### Added
- ReflectiveAgent: declarative evaluator-optimizer loop with native async support (ainvoke/astream)

### Changed
- Documentation and examples for ReflectiveAgent

## [0.1.6] - 2026-07-07

### Fixed
- CORS middleware now wraps auth instead of sitting inside it, answering preflight before auth challenge (fixes 401 on OPTIONS)

## [0.1.5] - 2026-07-04

### Fixed
- Agent tool-call reasoning labels now deduplicate consecutive duplicates

## [0.1.4] - 2026-07-01

### Added
- Test suite runs on pull requests

### Fixed
- astream bridge accepts non-generator iterators without deadlock
- astream bridge stops the sync producer on consumer break
- Orchestrator supervisor routing uses whole-word matching with one strict retry

## [0.1.3] - 2026-07-01

### Fixed
- Hardened streaming, error boundaries, and registry state (audit sweep)

## [0.1.2] - 2026-06-30

### Fixed
- Provider streams bounded with timeout to prevent indefinite hangs

## [0.1.1] - 2026-06-29

### Added
- Editable tool-call reasoning label via `tool_call_label` attribute (declarative, templated, overridable per subclass for i18n)

### Fixed
- Gemini structured content (list) flattened to plain text in interop and streaming paths (ToolAgent stream/astream)

### Changed
- Documentation for tool_call_label attribute

## [0.1.0] - 2026-06-27

Initial release of aixon framework with core declarative agent system (ToolAgent, LLMAgent, Orchestrator), streaming support, multi-provider compatibility (OpenAI, Anthropic, Gemini, Cohere, Ollama), and OpenAI-compatible API adapter.
