"""Contexto por request entregue ao ``request_hook`` do Server.

Tipos puros: nenhum import de fastapi/starlette, para que a aplicação possa
escrever e testar seu hook sem o extra ``server`` instalado."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

from aixon.exceptions import AixonError


@dataclass(frozen=True)
class RequestContext:
    """O que o hook enxerga de uma request: modelo pedido, agente resolvido,
    cabeçalhos (chaves minúsculas, somente leitura), parâmetros, adapter e
    se a resposta é em stream."""

    model: str
    agent_name: str
    headers: Mapping[str, str]
    params: Mapping[str, Any]
    adapter: str
    stream: bool


@dataclass(frozen=True)
class RequestDecision:
    """A decisão do hook. Todo campo ``None`` significa "não mexer"."""

    impose: tuple[str, str] | None = None
    user_email: str | None = None
    trace_user_id: str | None = None
    trace_session_id: str | None = None


RequestHookLike = Callable[[RequestContext], "RequestDecision | None"]
"""O que o Server aceita como ``request_hook``: qualquer chamável síncrono que
devolve uma decisão, ``None``, ou levanta ``RequestRejected``. Função solta
continua valendo; ``RequestHook`` (abaixo) é só uma forma organizada de
escrever um."""


@dataclass(frozen=True)
class Identity:
    """Quem fez a request: e-mail e identidade do trace. ``None`` é "não mexer"."""

    user_email: str | None = None
    trace_user_id: str | None = None
    trace_session_id: str | None = None


class RequestHook:
    """Hook em fases. ``__call__`` roda ``check`` → ``identity`` →
    ``imposition``, nessa ordem, e monta a ``RequestDecision``. Qualquer fase
    pode levantar ``RequestRejected``; a recusa em ``check`` impede as demais.

    A base não tem opinião (decisão vazia); sobrescreva só as fases que
    interessam. Não é obrigatório herdar: uma função solta que satisfaça
    ``RequestHookLike`` continua aceita pelo Server."""

    def check(self, ctx: RequestContext) -> None:
        """Recusa a request levantando ``RequestRejected``. Base: aceita."""

    def identity(self, ctx: RequestContext) -> Identity:
        """Identidade da request. Base: ``Identity()`` (não mexer)."""
        return Identity()

    def imposition(self, ctx: RequestContext) -> tuple[str, str] | None:
        """``(modelo, provider)`` a impor, ou ``None``. Base: ``None``."""
        return None

    def __call__(self, ctx: RequestContext) -> RequestDecision:
        self.check(ctx)
        ident = self.identity(ctx)
        impose = self.imposition(ctx)
        return RequestDecision(
            impose=impose,
            user_email=ident.user_email,
            trace_user_id=ident.trace_user_id,
            trace_session_id=ident.trace_session_id,
        )


class HeaderIdentityHook(RequestHook):
    """Identidade lida de cabeçalhos configuráveis. ``user_email`` é o valor
    cru de ``email_header``; ``trace_user_id`` é o e-mail ou, se vazio/ausente,
    ``user_id_header`` (quando configurado); ``trace_session_id`` vem de
    ``session_header`` (quando configurado). Nomes comparados em minúsculas."""

    def __init__(
        self,
        *,
        email_header: str,
        user_id_header: str | None = None,
        session_header: str | None = None,
    ) -> None:
        if not email_header or not email_header.strip():
            raise AixonError("HeaderIdentityHook: email_header must be non-empty")
        self.email_header = email_header.lower()
        self.user_id_header = user_id_header.lower() if user_id_header else None
        self.session_header = session_header.lower() if session_header else None

    def identity(self, ctx: RequestContext) -> Identity:
        h = ctx.headers
        email = h.get(self.email_header)
        user_id = email or (h.get(self.user_id_header) if self.user_id_header else None)
        session = h.get(self.session_header) if self.session_header else None
        return Identity(
            user_email=email, trace_user_id=user_id, trace_session_id=session
        )


class ChainRequestHook(RequestHook):
    """Compõe hooks em ordem. Um ``RequestRejected`` de qualquer um para a
    cadeia. As decisões se mesclam campo a campo: o primeiro valor não-``None``
    vence. Hook que devolve ``None`` conta como ``RequestDecision()``."""

    def __init__(self, hooks: Sequence[RequestHookLike]) -> None:
        self.hooks = tuple(hooks)

    def __call__(self, ctx: RequestContext) -> RequestDecision:
        merged: dict[str, Any] = {}
        for hook in self.hooks:
            decision = hook(ctx) or RequestDecision()
            for f in fields(decision):
                if merged.get(f.name) is None:
                    merged[f.name] = getattr(decision, f.name)
        return RequestDecision(**merged)


def make_context(
    *,
    model: str,
    agent_name: str,
    raw_headers: Iterable[tuple[str, str]],
    params: Mapping[str, Any],
    adapter: str,
    stream: bool,
) -> RequestContext:
    """Monta o contexto normalizando as chaves dos cabeçalhos para minúsculas;
    em duplicata, o PRIMEIRO valor vence (como ``starlette.Headers.get``)."""
    headers: dict[str, str] = {}
    for k, v in raw_headers:
        headers.setdefault(k.lower(), v)
    return RequestContext(
        model=model,
        agent_name=agent_name,
        headers=MappingProxyType(headers),
        params=params,
        adapter=adapter,
        stream=stream,
    )
