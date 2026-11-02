"""The aixon ASGI Server.

A singleton FastAPI app that mounts one or more ProtocolAdapters over the agent
Registry. Request flow:

    ASGI -> adapter.parse_request -> get_registry().resolve(model)
         -> agent.invoke|stream  (NEUTRAL Message[]/Chunk only)
         -> adapter.format_*      -> HTTP / SSE

The Server is dialect-agnostic: every wire detail lives in the adapter. Bearer
auth (AUTH_API_KEY env) wraps the whole app when set; /health and each
adapter's model-list route stay public. Mirrors restmcp's FastAPI + ASGI-auth
construction."""

from __future__ import annotations

import contextlib
import datetime as dt
import hmac
import json
import os
import threading
from types import MappingProxyType
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from aixon.exceptions import (
    AgentNotFoundError,
    AixonError,
    GatewayBudgetExceeded,
    RequestRejected,
    classify_gateway_error,
)
from aixon.logging import Logger
from aixon.message import Chunk
from aixon.observability import observe_request
from aixon.registry import get_registry
from aixon.runtime import (
    client_tools_scope,
    generation_params,
    imposed_model_scope,
    tool_choice_scope,
    user_email_scope,
)
from aixon.toolcache import tool_call_cache
from aixon.usage import usage_for
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.compat import open_webui_request_hook
from aixon.server.context import RequestDecision, RequestHookLike, make_context
from aixon.server.protocol import ParsedRequest, ProtocolAdapter

if TYPE_CHECKING:
    from aixon.agent import Agent

_log = Logger("aixon.server")

# What the user is told when the gateway refuses the call: the text of the chat
# message on a stream, the "message" of the 429 body without one. Each can be
# replaced per Server (budget_exceeded_message / rate_limited_message).
DEFAULT_BUDGET_EXCEEDED_MESSAGE = (
    "Your usage budget has been exhausted. Please try again after it resets "
    "or contact your administrator."
)
DEFAULT_RATE_LIMITED_MESSAGE = (
    "Too many requests to the model right now. Please wait a moment and try again."
)


def _valid_token(raw: str) -> bool:
    keys = [k.strip() for k in os.getenv("AUTH_API_KEY", "").split(",") if k.strip()]
    return bool(raw) and any(hmac.compare_digest(raw, k) for k in keys)


# --- debug tap (AIXON_DEBUG_REQUESTS) --------------------------------------
# Opt-in diagnostic recorder for client integrations: one JSONL record per
# chat POST with the verbatim body, the resolved agent and the response (the
# non-stream payload, or the raw SSE lines of a stream). Headers are NEVER
# recorded — Authorization can't leak by construction. Env is checked per
# request (like auth) so tests/hot-reload can toggle it without a rebuild.

@dataclass
class _ChatCall:
    """O que uma request de chat já resolvida carrega até a resposta: o par
    adapter/rota, o agente, o pedido neutro, o corpo cru (só para o tap) e a
    decisão do ``request_hook`` (imposição, e-mail e identidade do trace)."""

    adapter: ProtocolAdapter
    path: str
    agent: "Agent"
    pr: ParsedRequest
    body: dict
    model: str
    tap: bool
    decision: RequestDecision


def _tap_enabled(agent_name: str) -> bool:
    """Truthy values ("1"/"true"/"yes") record EVERY agent; anything else is a
    comma-separated allowlist of agent names — only those are recorded. The
    allowlist keeps the blast radius contained on shared servers: agents
    handling sensitive conversations never touch the disk while one specific
    integration is under diagnosis."""
    raw = os.getenv("AIXON_DEBUG_REQUESTS", "").strip()
    if not raw:
        return False
    if raw.lower() in ("1", "true", "yes"):
        return True
    return agent_name in {n.strip() for n in raw.split(",") if n.strip()}


def _tap_write(record: dict) -> None:
    """Append one JSONL record; tap failures must never break the request."""
    try:
        dirpath = Path(os.getenv("AIXON_DEBUG_REQUESTS_DIR") or "./aixon-debug")
        dirpath.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc)
        record = {"ts": stamp.isoformat(), **record}
        path = dirpath / f"requests-{stamp.strftime('%Y%m%d')}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        _log.info(f"debug tap: request recorded at {path}")
    except Exception as exc:  # diagnóstico nunca derruba a request
        _log.error(f"debug tap failed: {exc}")


class _AuthMiddleware:
    """Pure-ASGI Bearer middleware. No-op when AUTH_API_KEY is unset. Does not
    buffer the body, so SSE streaming is unaffected. ``public_paths`` are
    matched by exact path AND read-only method (GET/HEAD): the public set only
    ever holds GET routes (/health + model lists), so an adapter that mounts a
    POST on the same path keeps that POST guarded."""

    def __init__(self, app, public_paths):
        self.app = app
        self.public = frozenset(public_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not os.getenv("AUTH_API_KEY"):
            return await self.app(scope, receive, send)
        if scope.get("method", "").upper() in ("GET", "HEAD") \
                and scope.get("path", "") in self.public:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"").decode("utf-8", "ignore")
        token = auth.split(" ", 1)[1] if auth.startswith("Bearer ") else None
        if not token or not _valid_token(token):
            from starlette.responses import JSONResponse

            resp = JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await resp(scope, receive, send)
        return await self.app(scope, receive, send)


class Server:
    _instance: Optional["Server"] = None
    # Declared here (not just assigned in __new__/__init__) so mypy can
    # resolve its type — inferring it purely from `cls._instance._initialized
    # = False` inside __new__ is circular (it needs _instance's own attribute
    # type first).
    _initialized: bool = False
    # Declared for the same reason: `__init__` reads them back (singleton check)
    # above the line that assigns them on the first construction.
    budget_exceeded_message: str
    rate_limited_message: str
    request_hook: RequestHookLike

    # Trava de classe (server-10): a primeira construção (__new__/__init__) e
    # o build lazy do app acontecem uma vez só, mesmo com threads concorrentes.
    # Reentrante: _build_app() roda adapter.routes() e overrides de subclasse,
    # que podem construir Server() ou ler .app sob a mesma trava.
    _lock = threading.RLock()

    def __new__(
        cls,
        adapters: list[ProtocolAdapter] | None = None,
        *,
        request_hook: RequestHookLike | None = None,
        budget_exceeded_message: str = DEFAULT_BUDGET_EXCEEDED_MESSAGE,
        rate_limited_message: str = DEFAULT_RATE_LIMITED_MESSAGE,
    ):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        adapters: list[ProtocolAdapter] | None = None,
        *,
        request_hook: RequestHookLike | None = None,
        budget_exceeded_message: str = DEFAULT_BUDGET_EXCEEDED_MESSAGE,
        rate_limited_message: str = DEFAULT_RATE_LIMITED_MESSAGE,
    ):
        with type(self)._lock:
            self._init_locked(
                adapters,
                request_hook=request_hook,
                budget_exceeded_message=budget_exceeded_message,
                rate_limited_message=rate_limited_message,
            )

    def _init_locked(
        self,
        adapters: list[ProtocolAdapter] | None,
        *,
        request_hook: RequestHookLike | None,
        budget_exceeded_message: str,
        rate_limited_message: str,
    ) -> None:
        """Corpo do __init__, sempre sob ``_lock``: assim ``_initialized`` não
        é lido por uma thread enquanto outra ainda o está montando."""
        # Singleton: adapters are fixed on first construction. A later
        # Server(adapters=[...]) that asks for a DIFFERENT set raises instead of
        # silently ignoring the argument (which used to lock callers into
        # OpenAI-only without warning) — call Server._reset() first to rebuild.
        # The gateway messages follow the same rule: the first construction
        # fixes them, and a later one that asks for another text raises. The
        # DEFAULT counts as "no opinion" (a bare Server() must keep working
        # after the app configured the messages), so it keeps what was stored.
        if self._initialized:
            if adapters is not None and self._adapter_signature(adapters) != \
                    self._adapter_signature(self._protocol_adapters):
                raise AixonError(
                    "Server is already constructed with a different adapter set "
                    f"({[type(a).__name__ for a in self._protocol_adapters]}). "
                    "The singleton's adapters are fixed; call Server._reset() "
                    "before constructing it with a new adapter set."
                )
            for argument, asked, default, kept in (
                ("budget_exceeded_message", budget_exceeded_message,
                 DEFAULT_BUDGET_EXCEEDED_MESSAGE, self.budget_exceeded_message),
                ("rate_limited_message", rate_limited_message,
                 DEFAULT_RATE_LIMITED_MESSAGE, self.rate_limited_message),
            ):
                if asked not in (kept, default):
                    raise AixonError(
                        f"Server is already constructed with a different {argument} "
                        f"({kept!r}). The singleton's messages are fixed; call "
                        "Server._reset() before constructing it with a new "
                        f"{argument}."
                    )
            # O request_hook idem: None é "sem opinião" (é o que `aixon serve`
            # passa depois do autodiscover); outro objeto (por identidade)
            # levanta.
            if request_hook is not None and request_hook is not self.request_hook:
                raise AixonError(
                    "Server is already constructed with a different request_hook "
                    f"({self.request_hook!r}). The singleton's request_hook is "
                    "fixed; call Server._reset() before constructing it with a "
                    "new request_hook."
                )
            return
        self._protocol_adapters: list[ProtocolAdapter] = adapters or [OpenAIAdapter()]
        # Sem hook da aplicação, o padrão reproduz a 0.1.28 (headers do Open
        # WebUI). Um hook próprio o EXCLUI: não há fallback para os headers.
        self.request_hook = request_hook or open_webui_request_hook
        self.budget_exceeded_message = budget_exceeded_message
        self.rate_limited_message = rate_limited_message
        self._app = None
        self._initialized = True

    @staticmethod
    def _adapter_signature(adapters: list[ProtocolAdapter]) -> list[tuple[str, str]]:
        """Identity of an adapter set for the 'already constructed' check:
        (class name, mount prefix) per adapter. Two equivalent sets compare
        equal even though the instances differ."""
        return [(type(a).__name__, getattr(a, "mount_prefix", "")) for a in adapters]

    # --- app construction ------------------------------------------------
    @property
    def app(self):
        with type(self)._lock:
            if self._app is None:
                self._app = self._build_app()
            return self._app

    @staticmethod
    def _full_path(adapter: ProtocolAdapter, path: str) -> str:
        """Apply the adapter's mount prefix (default ``""``) to a route path.
        ``getattr`` guards custom adapters written before mount_prefix existed."""
        return getattr(adapter, "mount_prefix", "") + path

    def _public_paths(self) -> set[str]:
        public = {"/health"}
        for adapter in self._protocol_adapters:
            for method, path in adapter.routes():
                if method.upper() == "GET":
                    # model-list routes stay public (at their mounted path)
                    public.add(self._full_path(adapter, path))
        return public

    def _build_app(self):
        from fastapi import FastAPI
        from starlette.middleware.cors import CORSMiddleware

        app = FastAPI()

        @app.get("/health")
        def health():
            return {
                "status": "healthy",
                "server": "aixon",
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

        self._check_route_collisions()
        for adapter in self._protocol_adapters:
            self._mount_adapter(app, adapter)

        # Wrap unconditionally so the middleware can react to AUTH_API_KEY set
        # AFTER construction (tests, hot-reload). The middleware is a no-op when
        # the env is unset, so this is safe and matches restmcp's per-request check.
        guarded = _AuthMiddleware(app, public_paths=self._public_paths())
        # CORS must be OUTSIDE auth: preflight OPTIONS never carries
        # Authorization (per spec), so with auth outermost every preflight
        # would 401 and browsers would block the real request. Outermost CORS
        # also stamps allow-origin onto 401s, so the browser surfaces the auth
        # error instead of an opaque network failure.
        cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
        return CORSMiddleware(
            guarded, allow_origins=cors, allow_methods=["*"], allow_headers=["*"]
        )

    def _check_route_collisions(self) -> None:
        """Fail loudly if two adapters claim the same (method, mounted path).
        Without this, FastAPI keeps both registrations and the first silently
        shadows the second (e.g. OpenAI + Anthropic both at GET /v1/models).
        Give one adapter a ``mount_prefix`` to disambiguate."""
        seen: dict[tuple[str, str], str] = {}
        for adapter in self._protocol_adapters:
            for method, path in adapter.routes():
                key = (method.upper(), self._full_path(adapter, path))
                if key in seen:
                    raise AixonError(
                        f"Route {key[0]} {key[1]} is claimed by both "
                        f"'{seen[key]}' and '{adapter.name}'. Give one adapter a "
                        f"mount_prefix, e.g. "
                        f"{type(adapter).__name__}(mount_prefix='/{adapter.name}')."
                    )
                seen[key] = adapter.name

    def _mount_adapter(self, app, adapter: ProtocolAdapter) -> None:
        for method, path in adapter.routes():
            full = self._full_path(adapter, path)
            if method.upper() == "GET":
                self._mount_models(app, adapter, full)
            else:
                self._mount_chat(app, adapter, full)

    def _mount_models(self, app, adapter: ProtocolAdapter, path: str) -> None:
        async def list_models():
            _log.info(f"{adapter.name}: GET {path} (model list)")
            return adapter.format_models(get_registry().public())

        app.add_api_route(path, list_models, methods=["GET"])

    def _gateway_refusal(self, verdict: type[AixonError]) -> tuple[str, str, str]:
        """``(message, type, code)`` the user gets when the gateway refused the
        call with ``verdict``, one of the two ``classify_gateway_error`` gives.
        The message is the same in a stream and in the 429 body; ``type`` and
        ``code`` only exist in the body."""
        if verdict is GatewayBudgetExceeded:
            return self.budget_exceeded_message, "insufficient_quota", "budget_exceeded"
        return self.rate_limited_message, "rate_limit_exceeded", "rate_limited"

    def _request_scopes(
        self, *, agent, adapter, pr, decision: RequestDecision, stream: bool,
    ) -> contextlib.ExitStack:
        """A pilha de escopos de uma request de chat, igual nos caminhos stream
        e não stream. Ordem fixa, de fora para dentro: observe_request,
        generation_params, user_email_scope, imposed_model_scope (só quando o
        hook decidiu ``impose``), client_tools_scope, tool_choice_scope,
        tool_call_cache. Se um enter falhar, o ExitStack fecha os que já
        entraram."""
        stack = contextlib.ExitStack()
        with stack:
            stack.enter_context(observe_request(
                agent.name,
                user_id=decision.trace_user_id or decision.user_email,
                session_id=decision.trace_session_id,
                metadata={"adapter": adapter.name, "stream": stream},
            ))
            stack.enter_context(generation_params(pr.params))
            stack.enter_context(user_email_scope(decision.user_email))
            if decision.impose is not None:
                stack.enter_context(imposed_model_scope(*decision.impose, by="server"))
            stack.enter_context(client_tools_scope(pr.tools))
            stack.enter_context(tool_choice_scope(pr.tool_choice))
            stack.enter_context(tool_call_cache())
            return stack.pop_all()

    def _mount_chat(self, app, adapter, path) -> None:
        from fastapi import Request
        from starlette.responses import JSONResponse

        async def chat(request: Request):
            pr = await self._parse_body(request, adapter, path)
            if isinstance(pr, JSONResponse):
                return pr
            agent = self._resolve_agent(pr)
            if isinstance(agent, JSONResponse):
                return agent
            _log.info(f"{adapter.name}: {path} -> agent '{agent.name}' (stream={pr.stream})")
            # O Starlette guarda o JSON já lido: esta é a mesma dict do parse.
            body = await request.json()
            model = pr.model or agent.name
            decision = self._decide(request, adapter, agent, pr)
            if isinstance(decision, JSONResponse):
                return decision
            call = _ChatCall(
                adapter=adapter, path=path, agent=agent, pr=pr, body=body,
                model=model, tap=_tap_enabled(agent.name),
                decision=RequestDecision(),
            )
            if isinstance(decision, RequestRejected):
                if pr.stream:
                    return self._stream_response(call, refusal=str(decision))
                _log.warning(
                    f"{adapter.name}: request for agent '{agent.name}' rejected "
                    f"by the request hook (HTTP {decision.status})"
                )
                return JSONResponse(
                    {"error": {"message": str(decision), "type": "request_rejected",
                               "code": decision.code}},
                    status_code=decision.status,
                )
            call.decision = decision
            if pr.stream:
                return self._stream_response(call)
            return await self._sync_response(call)

        # `from __future__ import annotations` (module-level) turns
        # `request: Request` into the *string* "Request". FastAPI resolves
        # string annotations via `typing.get_type_hints(chat)`, which looks the
        # name up in `chat.__globals__` (this module's globals) — but `Request`
        # was only imported into this function's local scope above, so the
        # lookup would fail and FastAPI would silently treat `request` as a
        # query param (422 "field required"). Overwrite the annotation with the
        # real class object so FastAPI's introspection needs no string lookup.
        chat.__annotations__["request"] = Request
        app.add_api_route(path, chat, methods=["POST"])

    def _decide(self, request, adapter, agent, pr):
        """Chama o ``request_hook`` uma vez, antes de qualquer escopo. Devolve a
        decisão (``None`` vira ``RequestDecision()``), a ``RequestRejected``
        que o hook levantou, ou a JSONResponse 500 para qualquer outra falha
        (o log nomeia só o tipo: o texto pode trazer dado do usuário)."""
        from starlette.responses import JSONResponse

        ctx = make_context(
            model=pr.model or agent.name, agent_name=agent.name,
            raw_headers=request.headers.items(),
            # Cópia somente leitura: o hook não altera os params da request.
            params=MappingProxyType(dict(pr.params or {})),
            adapter=adapter.name, stream=bool(pr.stream),
        )
        try:
            return self.request_hook(ctx) or RequestDecision()
        except RequestRejected as exc:
            return exc
        except Exception as exc:
            _log.error(
                f"{adapter.name}: request hook failed for agent '{agent.name}' "
                f"({type(exc).__name__})"
            )
            return JSONResponse(
                {"error": {"message": "The request hook failed.",
                           "type": "server_error"}},
                status_code=500,
            )

    async def _parse_body(self, request, adapter, path):
        """Lê e valida o corpo: devolve o ParsedRequest do adapter, ou a
        JSONResponse 400 quando o corpo não é um objeto JSON ou o adapter o
        recusa."""
        from starlette.responses import JSONResponse

        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return JSONResponse(
                {
                    "error": {
                        "message": "Request body must be a JSON object.",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )
        try:
            return adapter.parse_request(body, path=path)
        except Exception as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "invalid_request_error"}},
                status_code=400,
            )

    def _resolve_agent(self, pr):
        """O agente de ``pr.model`` (com o thought_mode do agente injetado em
        ``pr.params``), ou a JSONResponse 404 quando o modelo não existe."""
        from starlette.responses import JSONResponse

        try:
            agent = get_registry().resolve(pr.model)
        except AgentNotFoundError as exc:
            return JSONResponse(
                {"error": {"message": exc.message, "type": "model_not_found"}},
                status_code=404,
            )
        # Agent-level thought mode: fills the gap between the per-request
        # param (which always wins — only injected when absent) and the
        # adapter-wide default. Injected into params so both the stream
        # session and format_response see one resolved knob. Safe for the
        # provider: thought_stream_mode is not in GENERATION_PARAMS.
        agent_mode = getattr(agent, "thought_mode", None)
        if agent_mode and "thought_stream_mode" not in (pr.params or {}):
            pr.params = {**(pr.params or {}), "thought_stream_mode": agent_mode}
        return agent

    def _scopes_for(self, call: "_ChatCall", *, stream: bool) -> contextlib.ExitStack:
        return self._request_scopes(
            agent=call.agent, adapter=call.adapter, pr=call.pr,
            decision=call.decision, stream=stream,
        )

    @staticmethod
    def _write_tap(call: "_ChatCall", *, stream: bool, response) -> None:
        """O registro do debug tap, igual nos dois caminhos salvo ``stream``
        e ``response``."""
        _tap_write({
            "adapter": call.adapter.name, "path": call.path,
            "agent": call.agent.name, "model": call.model,
            "stream": stream, "body": call.body, "response": response,
        })

    def _stream_response(self, call: "_ChatCall", *, refusal: Optional[str] = None):
        """O caminho stream: StreamingResponse SSE que nunca deixa uma falha
        no meio do stream chegar ao Starlette. Com ``refusal`` (a recusa do
        ``request_hook``), o agente não roda: o texto sai como a última
        mensagem de um stream que termina como de costume, sem escopos."""
        from starlette.responses import StreamingResponse

        adapter, agent = call.adapter, call.agent
        session = adapter.open_stream(model=call.model, request=call.pr)

        async def gen():
            # A mid-stream failure (provider hard-wall timeout, usage
            # counting, adapter bug) fires AFTER the 200/text/event-stream
            # headers went out, so it must not propagate into Starlette
            # (which would abort the response with a truncated stream).
            # Emit an error event instead, then still close the stream.
            sse_lines: list[str] = []  # collected only when tap is on

            def emit(line: str) -> str:
                if call.tap:
                    sse_lines.append(line)
                return line

            async def drain(chunks):
                # chunk → linha SSE para cada chunk, depois a cauda da sessão.
                async for chunk in chunks:
                    line = session.chunk(chunk)
                    if line:
                        yield emit(line)
                tail = session.finish()
                if tail:
                    yield emit(tail)

            async def _aiter(items):
                for item in items:
                    yield item

            try:
                if refusal is not None:
                    _log.warning(
                        f"{adapter.name}: stream for agent '{agent.name}' "
                        f"rejected by the request hook; answered in the "
                        f"stream (HTTP 200)"
                    )
                    async for line in drain(
                        _aiter((Chunk(content=refusal), Chunk(done=True)))
                    ):
                        yield line
                else:
                    with self._scopes_for(call, stream=True):
                        async for line in drain(agent.astream(call.pr.messages)):
                            yield line
            except Exception as exc:
                verdict = classify_gateway_error(exc)
                if verdict is None:
                    _log.error(
                        f"{adapter.name}: stream via agent '{agent.name}' "
                        f"failed: {exc}"
                    )
                    yield emit(session.error(exc))
                else:
                    # The gateway said no (budget spent, rate limit): an
                    # answer for the user, not a server failure. It goes
                    # out as the last message of a stream that ends as
                    # usual — the content delta closes a <think> block
                    # the failed run left open, then the finish chunk
                    # and [DONE] follow — so a chat UI shows it instead
                    # of a generic error event.
                    # The verdict only: the gateway's own text may name
                    # the end user and the spend.
                    _log.warning(
                        f"{adapter.name}: stream via agent '{agent.name}' "
                        f"refused by the gateway ({verdict.__name__}); "
                        f"answered in the stream (HTTP 200)"
                    )
                    try:
                        text = self._gateway_refusal(verdict)[0]
                        async for line in drain(
                            _aiter((Chunk(content=text), Chunk(done=True)))
                        ):
                            yield line
                    except Exception as inner:
                        # Same rule as any mid-stream failure: it must
                        # not reach Starlette. Generic error, then done.
                        _log.error(
                            f"{adapter.name}: stream via agent "
                            f"'{agent.name}' failed while reporting the "
                            f"gateway refusal: {inner}"
                        )
                        yield emit(session.error(inner))
            try:
                yield emit(session.done())
            except Exception as exc:
                # Nunca relança pelo Starlette no meio do stream, mas registra.
                _log.error(f"{adapter.name}: session.done() failed: {exc}")
            if call.tap:
                self._write_tap(call, stream=True, response=sse_lines)

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def _sync_response(self, call: "_ChatCall"):
        """O caminho não stream: o payload do adapter, ou 429/500 em JSON."""
        from starlette.responses import JSONResponse

        adapter, agent = call.adapter, call.agent
        # await ainvoke (async-native or threaded bridge) so the LLM call
        # never blocks the event loop. Request generation params are active
        # for the duration of the call via the runtime contextvar.
        try:
            with self._scopes_for(call, stream=False):
                message = await agent.ainvoke(call.pr.messages)
        except Exception as exc:
            verdict = classify_gateway_error(exc)
            if verdict is not None:
                # The gateway said no: tell the client so with a 429 it can
                # act on (the type/code pair is what tells the two apart).
                # The verdict only in the log: the gateway's own text may
                # name the end user and the spend.
                _log.warning(
                    f"{adapter.name}: agent '{agent.name}' refused by the "
                    f"gateway ({verdict.__name__}); answered HTTP 429"
                )
                text, err_type, err_code = self._gateway_refusal(verdict)
                return JSONResponse(
                    {"error": {"message": text, "type": err_type, "code": err_code}},
                    status_code=429,
                    # The openai SDK retries a 429 unless told not to, and
                    # each retry would run the whole tree again.
                    headers={"x-should-retry": "false"},
                )
            _log.error(f"{adapter.name}: agent '{agent.name}' failed: {exc}")
            return JSONResponse(
                {"error": {"message": "The agent failed to process the request.",
                          "type": "server_error"}},
                status_code=500,
            )
        # Usage real do provedor, senão a estimativa do tiktoken (usage_for).
        # O stream segue só com a estimativa da sessão.
        usage = usage_for(call.model, call.pr.messages, message)
        payload = adapter.format_response(
            model=call.model, message=message, usage=usage, params=call.pr.params
        )
        if call.tap:
            self._write_tap(call, stream=False, response=payload)
        return payload

    # --- lifecycle -------------------------------------------------------
    def serve(self, host: str = "0.0.0.0", port: int = 8000):
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)

    @classmethod
    def get_instance(cls) -> "Server":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset(cls):
        with cls._lock:
            cls._instance = None
