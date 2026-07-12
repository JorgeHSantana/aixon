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

import datetime as dt
import hmac
import os
from typing import Optional

from aixon.exceptions import AgentNotFoundError, AixonError
from aixon.logging import Logger
from aixon.registry import get_registry
from aixon.runtime import client_tools, generation_params
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.usage import build_usage
from aixon.server.protocol import ProtocolAdapter

_log = Logger("aixon.server")


def _valid_token(raw: str) -> bool:
    keys = [k.strip() for k in os.getenv("AUTH_API_KEY", "").split(",") if k.strip()]
    return bool(raw) and any(hmac.compare_digest(raw, k) for k in keys)


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

    def __new__(cls, adapters: list[ProtocolAdapter] | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, adapters: list[ProtocolAdapter] | None = None):
        # Singleton: adapters are fixed on first construction. A later
        # Server(adapters=[...]) that asks for a DIFFERENT set raises instead of
        # silently ignoring the argument (which used to lock callers into
        # OpenAI-only without warning) — call Server._reset() first to rebuild.
        if self._initialized:
            if adapters is not None and self._adapter_signature(adapters) != \
                    self._adapter_signature(self._protocol_adapters):
                raise AixonError(
                    "Server is already constructed with a different adapter set "
                    f"({[type(a).__name__ for a in self._protocol_adapters]}). "
                    "The singleton's adapters are fixed; call Server._reset() "
                    "before constructing it with a new adapter set."
                )
            return
        self._protocol_adapters: list[ProtocolAdapter] = adapters or [OpenAIAdapter()]
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

    def _mount_chat(self, app, adapter, path) -> None:
        from fastapi import Request
        from starlette.responses import JSONResponse, StreamingResponse

        async def chat(request: Request):
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
                pr = adapter.parse_request(body, path=path)
            except Exception as exc:
                return JSONResponse(
                    {"error": {"message": str(exc), "type": "invalid_request_error"}},
                    status_code=400,
                )
            try:
                agent = get_registry().resolve(pr.model)
            except AgentNotFoundError as exc:
                return JSONResponse(
                    {"error": {"message": exc.message, "type": "model_not_found"}},
                    status_code=404,
                )
            _log.info(f"{adapter.name}: {path} -> agent '{agent.name}' (stream={pr.stream})")
            model = pr.model or agent.name

            if pr.stream:
                session = adapter.open_stream(model=model, request=pr)

                async def gen():
                    # A mid-stream failure (provider hard-wall timeout, usage
                    # counting, adapter bug) fires AFTER the 200/text/event-stream
                    # headers went out, so it must not propagate into Starlette
                    # (which would abort the response with a truncated stream).
                    # Emit an error event instead, then still close the stream.
                    try:
                        with generation_params(pr.params), client_tools(pr.tools):
                            async for chunk in agent.astream(pr.messages):
                                line = session.chunk(chunk)
                                if line:
                                    yield line
                        tail = session.finish()
                        if tail:
                            yield tail
                    except Exception as exc:
                        _log.error(
                            f"{adapter.name}: stream via agent '{agent.name}' "
                            f"failed: {exc}"
                        )
                        yield session.error(exc)
                    try:
                        yield session.done()
                    except Exception:
                        pass  # never re-raise through Starlette mid-stream

                return StreamingResponse(gen(), media_type="text/event-stream")

            # await ainvoke (async-native or threaded bridge) so the LLM call
            # never blocks the event loop. Request generation params are active
            # for the duration of the call via the runtime contextvar.
            try:
                with generation_params(pr.params), client_tools(pr.tools):
                    message = await agent.ainvoke(pr.messages)
            except Exception as exc:
                _log.error(f"{adapter.name}: agent '{agent.name}' failed: {exc}")
                return JSONResponse(
                    {"error": {"message": "The agent failed to process the request.",
                              "type": "server_error"}},
                    status_code=500,
                )
            # Provider-real usage (Message.usage, already OpenAI-shaped) wins;
            # the tiktoken estimate is the fallback for agents whose provider
            # reported none. Streaming keeps the estimate-only path.
            usage = message.usage
            if not usage:
                prompt_text = "\n".join(m.content for m in pr.messages)
                completion_text = message.content
                if message.reasoning:
                    completion_text += "\n" + message.reasoning
                usage = build_usage(model, prompt_text, completion_text)
            return adapter.format_response(model=model, message=message, usage=usage)

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
        cls._instance = None
