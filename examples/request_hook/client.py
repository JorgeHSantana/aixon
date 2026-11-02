"""Cliente do exemplo request_hook: a MESMA pergunta ao MESMO agente com
``x-tier: fast`` e ``x-tier: deep``.

O corpo da resposta não expõe o modelo imposto (o campo ``model`` ecoa o
agente), então o cliente imprime o tier enviado ao lado da resposta.

    python main.py          # num terminal
    python client.py        # noutro (AIXON_URL opcional, padrão localhost:8000)
"""

from __future__ import annotations

import os

import httpx

QUESTION = "Em uma frase: o que é um request hook?"
TIERS = ("fast", "deep")


def main(http: httpx.Client | None = None) -> None:
    """Manda uma request por tier. ``http`` permite injetar outro cliente
    (o teste passa um TestClient); sem ele, usa ``AIXON_URL``."""
    own = http is None
    if http is None:
        http = httpx.Client(
            base_url=os.environ.get("AIXON_URL", "http://localhost:8000"), timeout=120,
        )
    try:
        for tier in TIERS:
            r = http.post(
                "/v1/chat/completions",
                headers={"x-user": "ana@example.com", "x-tier": tier},
                json={"model": "helper", "messages": [{"role": "user", "content": QUESTION}]},
            )
            r.raise_for_status()
            body = r.json()
            answer = body["choices"][0]["message"]["content"]
            print(f"x-tier: {tier} (agente {body.get('model')}) -> {answer}")
    finally:
        if own:
            http.close()


if __name__ == "__main__":
    main()
