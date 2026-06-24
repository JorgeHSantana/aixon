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

from aixon.exceptions import AgentNotFoundError
from aixon.logging import Logger
from aixon.registry import get_registry
from aixon.server.adapters.openai import OpenAIAdapter
from aixon.server.protocol import ProtocolAdapter

_log = Logger("aixon.server")


def _valid_token(raw: str) -> bool:
    keys = [k.strip() for k in os.getenv("AUTH_API_KEY", "").split(",") if k.strip()]
    return bool(raw) and any(hmac.compare_digest(raw, k) for k in keys)


class _AuthMiddleware:
    """Pure-ASGI Bearer middleware. No-op when AUTH_API_KEY is unset. Does not
    buffer the body, so SSE streaming is unaffected. ``public_paths`` are
    matched by exact path."""

    def __init__(self, app, public_paths):
        self.app = app
        self.public = frozenset(public_paths)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not os.getenv("AUTH_API_KEY"):
            return await self.app(scope, receive, send)
        if scope.get("path", "") in self.public:
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

    def __new__(cls, adapters: list[ProtocolAdapter] | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, adapters: list[ProtocolAdapter] | None = None):
        # Singleton: adapters are fixed on first construction. A later
        # Server(adapters=[...]) call silently reuses the first instance's
        # adapters — call Server._reset() first if a different set is needed
        # (e.g. between tests).
        if self._initialized:
            return
        self._adapters: list[ProtocolAdapter] = adapters or [OpenAIAdapter()]
        self._app = None
        self._initialized = True

    # --- app construction ------------------------------------------------
    @property
    def app(self):
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _public_paths(self) -> set[str]:
        public = {"/health"}
        for adapter in self._adapters:
            for method, path in adapter.routes():
                if method.upper() == "GET":
                    public.add(path)  # model-list routes stay public
        return public

    def _build_app(self):
        from fastapi import FastAPI
        from starlette.middleware.cors import CORSMiddleware

        app = FastAPI()
        cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
        app.add_middleware(
            CORSMiddleware, allow_origins=cors, allow_methods=["*"], allow_headers=["*"]
        )

        @app.get("/health")
        def health():
            return {
                "status": "healthy",
                "server": "aixon",
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

        for adapter in self._adapters:
            self._mount_adapter(app, adapter)

        # Wrap unconditionally so the middleware can react to AUTH_API_KEY set
        # AFTER construction (tests, hot-reload). The middleware is a no-op when
        # the env is unset, so this is safe and matches restmcp's per-request check.
        return _AuthMiddleware(app, public_paths=self._public_paths())

    def _mount_adapter(self, app, adapter: ProtocolAdapter) -> None:
        for method, path in adapter.routes():
            if method.upper() == "GET":
                self._mount_models(app, adapter, path)
            else:
                self._mount_chat(app, adapter, path)

    def _mount_models(self, app, adapter: ProtocolAdapter, path: str) -> None:
        async def list_models():
            _log.info(f"{adapter.name}: GET {path} (model list)")
            return adapter.format_models(get_registry().public())

        app.add_api_route(path, list_models, methods=["GET"])

    def _mount_chat(self, app, adapter, path) -> None:
        from fastapi import Request
        from starlette.responses import JSONResponse, StreamingResponse

        async def chat(request: Request):
            body = await request.json()
            pr = adapter.parse_request(body, path=path)
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
                def gen():
                    for chunk in agent.stream(pr.messages):
                        line = adapter.format_stream_chunk(model=model, chunk=chunk)
                        if line:
                            yield line
                    yield adapter.format_stream_done(model=model)

                return StreamingResponse(gen(), media_type="text/event-stream")

            message = agent.invoke(pr.messages)
            return adapter.format_response(model=model, message=message, usage={})

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
