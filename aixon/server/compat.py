"""Hook padrão de compatibilidade do ``Server``.

Tipos puros, como ``aixon.server.context``: sem fastapi/starlette."""

from __future__ import annotations

from aixon.server.context import RequestContext, RequestDecision

OPEN_WEBUI_EMAIL_HEADER = "x-openwebui-user-email"
OPEN_WEBUI_USER_ID_HEADER = "x-openwebui-user-id"
OPEN_WEBUI_CHAT_ID_HEADER = "x-openwebui-chat-id"


def open_webui_request_hook(ctx: RequestContext) -> RequestDecision:
    """O que o aixon fazia até a 0.1.28. É o padrão quando
    `Server(request_hook=...)` não é passado; vira opt-in na 0.2.

    Lê os headers que o Open WebUI manda com
    ``ENABLE_FORWARD_USER_INFO_HEADERS=true``. O e-mail sai cru (vazio ou com
    espaços inclusive): quem o trata é o ``user_email_scope``. Para o trace, o
    e-mail vazio cai no user-id, como antes."""
    headers = ctx.headers
    email = headers.get(OPEN_WEBUI_EMAIL_HEADER)
    return RequestDecision(
        user_email=email,
        trace_user_id=email or headers.get(OPEN_WEBUI_USER_ID_HEADER),
        trace_session_id=headers.get(OPEN_WEBUI_CHAT_ID_HEADER),
    )
