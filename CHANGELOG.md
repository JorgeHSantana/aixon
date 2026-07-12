# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Real provider usage tracking via `Message.usage` (M1: tiktoken fallback for batch-only providers)
- Production stream session support for Anthropic with interleaved blocks and error envelope closing (M3)
- `ParsedRequest.tools` now always OpenAI-shaped, with anthropic definitions normalized (M2)
- mypy CI gate in both workflows plus a bare-install smoke job on PRs; `[tool.mypy]` config in pyproject.toml (M4)

### Changed
- Request tools dialect unified to OpenAI format across all providers (M2)

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
