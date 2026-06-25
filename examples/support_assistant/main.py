"""Acme support assistant — an OpenAI-compatible API server over aixon.

Run it:
    cd examples/support_assistant
    pip install -r requirements.txt
    python main.py                 # http://localhost:8000  (set PORT to change)

Then (OpenAI wire format, no SDK needed):
    curl http://localhost:8000/health
    curl http://localhost:8000/v1/models
    curl -X POST http://localhost:8000/v1/chat/completions \\
      -H 'content-type: application/json' \\
      -d '{"model":"support","messages":[{"role":"user","content":"where is my order 1002?"}]}'

Set OPENAI_API_KEY to use gpt-4o-mini; leave it unset to run fully offline on
the bundled demo provider. See README.md for auth and the full catalog.
"""

from __future__ import annotations

import os

from aixon import Server, autodiscover

# Import every module under agents/, registering each Agent (and, transitively,
# the demo provider, retriever and connector) at class-definition time.
autodiscover("agents")

# OpenAI-compatible server (the default adapter). Set AUTH_API_KEY to require a
# Bearer token; /health and /v1/models stay public.
server = Server()
app = server.app  # ASGI app — production: `uvicorn main:app --workers 4`


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    if os.getenv("AUTH_API_KEY"):
        print("auth: ON (send 'Authorization: Bearer <key>')")
    else:
        print("auth: OFF (set AUTH_API_KEY to require a token)")
    mode = "gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "offline demo provider"
    print(f"model: {mode}")
    server.serve(host="0.0.0.0", port=port)
