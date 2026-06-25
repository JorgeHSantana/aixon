# aixon Multimodal Input (Vision) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Status:** DRAFT — for future implementation. Not started.

**Goal:** Let agents accept **media input** (images first) through the neutral
boundary, so multimodal models (gpt-4o, claude-3.5-sonnet, gemini-*) can see
images sent by a client. The work is concentrated in four places — the rest of
the stack is already multimodal-ready because aixon talks to vendors only
through LangChain.

**Why this is bounded (architecture works in our favor):**
- **Providers / LLM:** zero change. `langchain_openai/anthropic/google` already
  send image content blocks to the vendor APIs; aixon never calls those APIs
  directly.
- **`to_langchain`:** nearly free — LangChain message constructors already accept
  `content` as a list of content blocks; today aixon passes `msg.content`
  straight through.
- **`from_langchain`:** one line is the blocker — `aixon/_interop/messages.py`
  currently does `content = msg.content if isinstance(msg.content, str) else str(msg.content)`,
  which **stringifies** structured content (flagged as a known limitation in the
  Plan 2/3 ledgers). It must preserve list content instead.
- **The real work:** (1) a neutral content-part model + widening `Message.content`;
  (2) the two interop edges; (3) per-dialect adapter translation (the bulk);
  (4) auditing the call-sites that assume `content` is a `str`.

**Architecture:**
- `aixon/content.py` (NEW) — a tiny neutral content model: `TextPart`,
  `ImagePart`. `ContentPart = TextPart | ImagePart`. No LangChain / vendor type
  imported here (neutral boundary).
- `aixon/message.py` — widen `content: str | list[ContentPart]` (default `""`,
  so the string case is byte-identical and backward compatible). Add
  `Message.text() -> str` that concatenates text parts (and returns `content`
  verbatim when it is already a `str`), for the many call-sites that only care
  about text.
- `aixon/_interop/messages.py` — `to_langchain` maps neutral parts → LangChain
  content blocks; `from_langchain` preserves list content (deletes the `str()`
  coercion).
- `aixon/server/adapters/openai.py` / `anthropic.py` — translate the dialect's
  multimodal wire shape (OpenAI `image_url` array; Anthropic `source/base64`
  block array) ↔ neutral parts.

**Scope:**
- **IN scope:** image input (vision), end to end (client → adapter → neutral →
  LangChain → vendor), through `LLMAgent` and `ToolAgent`.
- **Incremental (same mechanism, later tasks/plan):** audio input
  (`AudioPart`) — gpt-4o-audio / Gemini.
- **Out of scope here:** video input (Gemini-only wire specifics); **media
  OUTPUT** (image/audio generation) — that needs a different vendor API, output
  fields on `Message`/`Chunk`, and different streaming semantics → a SEPARATE
  plan. `Chunk` stays text-only (`content`/`reasoning` remain `str`).

**Tech Stack:** Python 3.11+, existing deps (langchain, langgraph; server extra
for the adapters). Hermetic tests only — a 1×1-px base64 image, a fake
multimodal model, no network, no API keys.

## Global Constraints

- **Backward compatible (BINDING):** `content` defaults to `""` and the `str`
  case must be byte-identical end to end. Every existing test (currently 324
  passed) keeps passing unchanged. A `str` content is NOT silently converted to
  a one-element list.
- **Neutral boundary (BINDING):** `aixon/content.py` imports no LangChain and no
  vendor SDK. `ContentPart` is a small neutral union mirroring only the common
  subset OpenAI/Anthropic/LangChain share. No wire/vendor field names leak
  inward (translate at the adapter; map at the interop edge).
- **`Message.text()` is the str path:** call-sites that only need text (provider
  classification, `Retriever.as_tool` query, CLI display, logging, the demo
  provider) must use `Message.text()`, never assume `content` is a `str`.
- **Tools stay text:** tool arguments/results remain text. Images flow as
  message content, not as tool args. `AgentTool.func` stays `Callable[[str], str]`.
- **`Chunk` unchanged:** output is text; media-out is a separate plan.
- **Hermetic tests:** a shared tiny PNG (1×1 transparent pixel) base64 constant
  in `tests/_fakes.py`; a `FakeMultimodalModel` that asserts it received image
  content and returns a deterministic text answer. No network, no keys.
- Error messages: state what was got and how to fix it (restmcp tone), e.g. an
  unknown `ContentPart` type names the offending dict.
- Exports: `TextPart`, `ImagePart` (and `ContentPart`) exported from `aixon`.

---

### Task 1: Neutral content model + `Message.content` widening + `Message.text()`

**Files:** Create `aixon/content.py`; modify `aixon/message.py`, `aixon/__init__.py`; test `tests/test_content.py`.

**Interfaces produced:**
- `aixon.content.TextPart(text: str)`, `aixon.content.ImagePart(url: str | None, base64: str | None, mime_type: str | None)` (a small frozen dataclass each; exactly one of `url`/`base64` set).
- `ContentPart = TextPart | ImagePart` (type alias).
- `Message.content: str | list[ContentPart]` (default `""`).
- `Message.text() -> str` — returns `content` if it is a `str`; else the
  newline/space-joined text of its `TextPart`s (ignoring images).
- `Message.to_dict`/`serialize` updated to render list content as a list of
  neutral dicts (string content still renders as a plain string — backward
  compatible).

**Acceptance:**
- [ ] `Message(role="user", content="hi").text() == "hi"` and serializes to a string `content` exactly as today.
- [ ] `Message(role="user", content=[TextPart("describe"), ImagePart(url="http://x/i.png")])` — `.text() == "describe"`; serialization is a list of neutral dicts.
- [ ] `ImagePart` requires exactly one of `url`/`base64` (else `ValueError` naming the problem).
- [ ] Exports importable from `aixon`.

> Design note: keep `ContentPart` minimal — `TextPart` + `ImagePart` only. Resist
> adding fields vendors don't share. `AudioPart` is a later task; do not add it now (YAGNI).

---

### Task 2: Interop — preserve multimodal across `to_langchain` / `from_langchain`

**Files:** modify `aixon/_interop/messages.py`; test `tests/test_interop_multimodal.py`.

**Work:**
- `to_langchain`: when `msg.content` is a list, map each `ContentPart` to a
  LangChain content block:
  - `TextPart(text)` → `{"type": "text", "text": text}`
  - `ImagePart(url=...)` → `{"type": "image_url", "image_url": {"url": url}}`
  - `ImagePart(base64=..., mime_type=...)` → `{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}`
  (This is the LangChain-canonical multimodal block shape, accepted across providers.)
  When `msg.content` is a `str`, behavior is unchanged.
- `from_langchain`: **delete the `str(msg.content)` coercion.** When the
  LangChain message's `content` is a list, map blocks back to `ContentPart`s;
  when it is a `str`, keep the str. Text-only AI replies stay `str` (the common
  output case), so assistant messages are unaffected.

**Acceptance:**
- [ ] `to_langchain([Message(content=[TextPart("hi"), ImagePart(base64=PNG_1PX, mime_type="image/png")])])` produces a `HumanMessage` whose `content` is a 2-element list with a `text` block and an `image_url` data-URI block.
- [ ] Round-trip: `from_langchain(to_langchain([m])[0])` preserves the parts (no stringification).
- [ ] A plain-string message still round-trips to a `str` (regression guard).
- [ ] The existing tool_call_id/name round-trip test still passes.

---

### Task 3: Audit str-assuming call-sites; route text through `Message.text()`

**Files:** modify `aixon/agents/llm_agent.py` (verify `_with_prompt` is content-agnostic — it only prepends, no `.content` read, so likely no change), `aixon/cli.py` (display), `aixon/logging` call-sites, `aixon/retriever.py` (`as_tool` builds a query from a `str` — unaffected, tools are text), `aixon/server/server.py` (logging). Test `tests/test_text_helper_callsites.py`.

**Work:** Grep for `.content` reads that assume `str` and that should tolerate
list content by using `Message.text()`. Known: CLI chat display, server request
logging, any `in`/`.lower()`/slicing on `content`. The agents' core paths
(`invoke`/`ainvoke`) pass `content` through to `to_langchain` untouched, so they
need no change.

**Acceptance:**
- [ ] A `Message` with list content can be passed to `agent.invoke`/`ainvoke` without `AttributeError`/`TypeError` anywhere in the core path.
- [ ] CLI display and server logging render list content via `.text()` (no crash, shows the text part).
- [ ] No behavior change for string content (regression).

---

### Task 4: OpenAI adapter — parse vision requests

**Files:** modify `aixon/server/adapters/openai.py`; test `tests/test_adapter_openai_vision.py`.

**Work:** `parse_request` currently does `content=m.get("content") or ""`. When a
message's `content` is a **list** (OpenAI vision format), map each part:
- `{"type": "text", "text": ...}` → `TextPart`
- `{"type": "image_url", "image_url": {"url": ...}}` → `ImagePart(url=...)` (or
  `base64`+`mime` if the url is a `data:` URI — parse it).
A `str` content stays a `str` (unchanged). `format_response` is unchanged
(output is text).

**Acceptance:**
- [ ] Posting `{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,<PNG_1PX>"}}]}]}` yields a `ParsedRequest` whose message has `[TextPart, ImagePart]` content.
- [ ] A normal string-content request is unchanged.

---

### Task 5: Anthropic adapter — parse vision requests

**Files:** modify `aixon/server/adapters/anthropic.py`; test `tests/test_adapter_anthropic_vision.py`.

**Work:** Anthropic's content blocks differ: text is
`{"type":"text","text":...}`; image is
`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<b64>"}}`.
The adapter's `_flatten_content` currently collapses arrays to a string — replace
with a mapper to neutral `ContentPart`s (text → `TextPart`, image → `ImagePart(base64=..., mime_type=...)`).

**Acceptance:**
- [ ] An Anthropic-shaped vision request parses to `[TextPart, ImagePart(base64=...)]`.
- [ ] Text-only Anthropic requests unchanged; the existing system-prompt-hoist behavior is preserved.

---

### Task 6: End-to-end hermetic vision test + fake multimodal model

**Files:** modify `tests/_fakes.py` (add `PNG_1PX` constant + `FakeMultimodalModel`/provider); test `tests/test_vision_end_to_end.py`.

**Work:** `FakeMultimodalModel._generate` inspects the incoming LangChain
messages, asserts an `image_url`/image block is present, and returns
`AIMessage(content="I see an image.")`. Drive it through an `LLMAgent` and a
`ToolAgent` with list content; assert the model actually received the image
block (proving the path carries media, not just text) and the agent returns the
text answer. Async variant via `ainvoke`.

**Acceptance:**
- [ ] `LLMAgent.invoke([Message(content=[TextPart("what is this?"), ImagePart(base64=PNG_1PX, mime_type="image/png")])])` reaches the fake model **with** the image block and returns text.
- [ ] Same via `ainvoke`.
- [ ] A server round-trip (`POST /v1/chat/completions` with the OpenAI vision body) returns a text completion (uses Task 4 + the fake model).

---

### Task 7: Example + documentation

**Files:** `examples/support_assistant/` (a small vision touch or a dedicated mini-example), `docs/multimodal.md` (or sections in `docs/agents.md` + `docs/server.md`), README mention, `docs/olympus-vs-aixon.md` note.

**Work:**
- A runnable snippet: an agent that receives an image (offline: the demo
  provider returns a canned "I see ..." for image content) so `python` runs
  without keys; with a real key it actually describes the image.
- Docs: how to send images (Python `ImagePart`; OpenAI/Anthropic wire bodies),
  what is supported (image-in; audio/video noted as future; media-out out of
  scope), and the `Message.text()` helper.

**Acceptance:**
- [ ] Example runs offline.
- [ ] Docs show both the Python and wire-format ways to send an image, and state the scope boundaries.

---

## Risks / landmines (pre-chew for the implementer)

1. **The `from_langchain` `str()` line (Task 2)** is the single behavioral
   blocker — make sure deleting it doesn't change text-only output (assistant
   replies are strings; only *input* user messages carry lists).
2. **ContentPart shape:** map to LangChain's canonical `image_url`/`data:` block
   shape in `to_langchain` — it is the portable form accepted by all three
   providers. Don't invent a custom shape.
3. **Backward compat:** never convert a `str` into `[TextPart(str)]` implicitly —
   keep `str` as `str` so serialization and every existing test are unchanged.
4. **Adapter dialects differ** (OpenAI `image_url` vs Anthropic `source/base64`)
   — Tasks 4 and 5 are genuinely separate; don't share a parser.
5. **Tests must carry a real image block, not text** — assert the fake model
   *received* an image block, else the test would pass on a stringified path
   (the very bug being fixed). Use a 1×1-px base64 PNG constant; no network.
6. **`Chunk` and tools stay text** — resist widening them; media-out and
   media-tools are out of scope.

## Suggested execution order

Tasks 1 → 2 → 3 establish the neutral path (testable without the server). Task 4
(OpenAI) delivers the first user-visible capability; Task 5 (Anthropic) mirrors
it. Task 6 is the end-to-end proof. Task 7 ships example + docs. Audio-input is a
future increment (add `AudioPart` + the two adapter mappings, same shape).
