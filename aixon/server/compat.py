"""Hook padrão de compatibilidade do ``Server``.

Tipos puros, como ``aixon.server.context``: sem fastapi/starlette."""

from __future__ import annotations

from aixon.server.context import HeaderIdentityHook

OPEN_WEBUI_EMAIL_HEADER = "x-openwebui-user-email"
OPEN_WEBUI_USER_ID_HEADER = "x-openwebui-user-id"
OPEN_WEBUI_CHAT_ID_HEADER = "x-openwebui-chat-id"


class OpenWebUIRequestHook(HeaderIdentityHook):
    """O que o aixon fazia até a 0.1.28. A instância ``open_webui_request_hook``
    é o padrão quando `Server(request_hook=...)` não é passado; vira opt-in na
    0.2.

    Lê os headers que o Open WebUI manda com
    ``ENABLE_FORWARD_USER_INFO_HEADERS=true``. O e-mail sai cru (vazio ou com
    espaços inclusive): quem o trata é o ``user_email_scope``. Para o trace, o
    e-mail vazio cai no user-id, como antes. Herde para acrescentar ``check``
    ou ``imposition`` mantendo a identidade."""

    def __init__(self) -> None:
        super().__init__(
            email_header=OPEN_WEBUI_EMAIL_HEADER,
            user_id_header=OPEN_WEBUI_USER_ID_HEADER,
            session_header=OPEN_WEBUI_CHAT_ID_HEADER,
        )


open_webui_request_hook = OpenWebUIRequestHook()
