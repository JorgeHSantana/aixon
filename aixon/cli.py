"""CLI entry point for aixon.

Commands
--------
list    List registered agents.
chat    Interactive chat with an agent (in-process or remote).
new     Scaffold a consumer project.
serve   Start the aixon server (delegates to Plan 5 Server).
"""
from __future__ import annotations

import os
import sys

import click

from aixon import __version__
from aixon.discovery import autodiscover
from aixon.registry import get_registry


def _ensure_cwd_on_path() -> None:
    """Put the current working directory on ``sys.path`` so ``autodiscover``
    can import a project-local package (e.g. ``agents/``).

    A ``python main.py`` run already has the script's dir on ``sys.path[0]``,
    but the installed ``aixon`` console-script has its launcher dir there
    instead — so a CWD-local ``agents`` package would not import. Add the CWD to
    ``sys.path`` (only if absent) so the CLI behaves like the script."""
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _autodiscover_quietly(package: str) -> None:
    """Autodiscover ``package``, staying quiet ONLY when the package itself is
    absent (the normal "no agents/ directory here" case). An ImportError raised
    INSIDE a user's agent module (e.g. agents/weather.py importing a missing
    lib) is surfaced — silently swallowing it would print "No agents
    registered." with zero diagnostics or serve a partial agent list."""
    try:
        autodiscover(package)
    except ModuleNotFoundError as exc:
        name = exc.name or ""
        if name and (package == name or package.startswith(name + ".")):
            return  # graceful: no agents package in this directory
        click.echo(f"Warning: error importing agents from '{package}': {exc}", err=True)
    except ImportError as exc:
        click.echo(f"Warning: error importing agents from '{package}': {exc}", err=True)
    except ValueError:
        pass  # graceful: name exists but is not a package

# ---------------------------------------------------------------------------
# OpenAI — module-level import so tests can patch aixon.cli.OpenAI
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------
_DIM = "\033[2m"
_RESET = "\033[0m"


def _supports_ansi() -> bool:
    """Return True when the terminal likely supports ANSI (stdout is a TTY)."""
    return sys.stdout.isatty()


def _print_dim(text: str) -> None:
    if _supports_ansi():
        click.echo(f"{_DIM}{text}{_RESET}", nl=False)
    else:
        click.echo(text, nl=False)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------
@click.group()
@click.version_option(version=__version__, prog_name="aixon", message="%(prog)s %(version)s")
def app() -> None:
    """aixon — declarative AI-agent framework."""


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
@app.command(name="list")
@click.option("--package", "-p", default="agents", show_default=True,
              help="Package to autodiscover before listing.")
def list_command(package: str) -> None:
    """List registered agents."""
    _ensure_cwd_on_path()
    _autodiscover_quietly(package)

    agents = get_registry().public()
    if not agents:
        click.echo("No agents registered.")
        return

    for agent in agents:
        agent_type = type(agent).__name__
        desc = agent.description or ""
        click.echo(f"{agent.name}  [{agent_type}]  {desc}")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
@app.command(name="chat")
@click.option("--package", "-p", default="agents", show_default=True,
              help="Package to autodiscover (ignored when --url is set).")
@click.option("--url", default=None,
              help="Remote server URL (e.g. http://localhost:8000). "
                   "When set, routes messages via the OpenAI wire format.")
def chat_command(package: str, url: str | None) -> None:
    """Interactive chat with an agent."""
    if url:
        _chat_remote(url)
    else:
        _chat_inprocess(package)


def _pick_agent() -> object | None:
    """Display a menu and return the chosen agent, or None to exit."""
    agents = get_registry().public()
    if not agents:
        click.echo("No agents registered.")
        return None

    click.echo("\n--- Agent Menu ---")
    for i, agent in enumerate(agents, 1):
        agent_type = type(agent).__name__
        desc = f" — {agent.description}" if agent.description else ""
        click.echo(f"  {i}. {agent.name}  [{agent_type}]{desc}")
    click.echo("  0. Exit")
    click.echo()

    while True:
        raw = click.prompt("Choose", default="1")
        if raw.strip() == "0":
            return None
        try:
            idx = int(raw.strip()) - 1
            if 0 <= idx < len(agents):
                return agents[idx]
        except ValueError:
            pass
        click.echo("Invalid choice, try again.")


def _stream_inprocess(agent: object, messages: list) -> str:
    """Stream agent.stream(messages) to the terminal and RETURN the assistant
    content collected from the stream.

    Returning the streamed content (rather than re-running the agent via
    invoke() to build the history message) means the saved history is exactly
    what the user saw, and each turn costs ONE inference call, not two."""
    parts: list[str] = []
    try:
        for chunk in agent.stream(messages):
            if chunk.reasoning:
                _print_dim(chunk.reasoning)
            if chunk.content:
                click.echo(chunk.content, nl=False)
                parts.append(chunk.content)
            if chunk.done:
                click.echo()  # final newline
    except KeyboardInterrupt:
        click.echo()  # ensure newline after interrupted output
    except Exception as exc:
        # A provider error must not crash the CLI and lose the conversation —
        # report it and return to the prompt (same pattern as remote mode).
        click.echo(f"\nError: {exc}", err=True)
    return "".join(parts)


def _chat_inprocess(package: str) -> None:
    from aixon.message import Message

    _ensure_cwd_on_path()
    _autodiscover_quietly(package)

    agent = _pick_agent()
    if agent is None:
        return

    messages: list[Message] = []

    while True:
        try:
            user_input = click.prompt("\nYou", prompt_suffix="> ")
        except click.Abort:
            # Ctrl+C at empty prompt -> back to menu
            click.echo()
            agent = _pick_agent()
            if agent is None:
                return
            messages = []
            continue

        stripped = user_input.strip()

        if stripped == "/exit":
            click.echo("Goodbye.")
            return

        if stripped == "/menu":
            agent = _pick_agent()
            if agent is None:
                return
            messages = []
            continue

        if not stripped:
            continue

        messages.append(Message(role="user", content=stripped))

        click.echo()
        # ONE inference per turn: the streamed chunks ARE the response. Build
        # the history message from what was just displayed — never re-invoke
        # the agent (a second call would double cost/latency and could diverge
        # from the shown output under temperature > 0).
        content = _stream_inprocess(agent, messages)
        if content:
            messages.append(Message(role="assistant", content=content))


def _chat_remote(url: str) -> None:
    """Chat via OpenAI-compatible wire protocol against a remote aixon server."""
    if OpenAI is None:
        click.echo(
            "The 'openai' package is required for remote mode. "
            "Install with: pip install 'aixon[cli]'",
            err=True,
        )
        raise SystemExit(1)

    client = OpenAI(api_key="local", base_url=f"{url.rstrip('/')}/v1")

    # Fetch available models from the remote server
    try:
        models_response = client.models.list()
        remote_agents = [m.id for m in models_response.data]
    except Exception as exc:
        click.echo(f"Could not reach server at {url}: {exc}", err=True)
        raise SystemExit(1)

    if not remote_agents:
        click.echo("No agents available on the remote server.")
        return

    click.echo(f"\nConnected to {url}")

    # Outer loop: /menu breaks back here instead of recursing into
    # _chat_remote (recursion grew a stack frame per /menu -> RecursionError
    # in long sessions). Mirrors the in-process chat's loop pattern.
    while True:
        click.echo("--- Remote Agents ---")
        for i, name in enumerate(remote_agents, 1):
            click.echo(f"  {i}. {name}")
        click.echo("  0. Exit\n")

        while True:
            raw = click.prompt("Choose", default="1")
            if raw.strip() == "0":
                return
            try:
                idx = int(raw.strip()) - 1
                if 0 <= idx < len(remote_agents):
                    chosen_model = remote_agents[idx]
                    break
            except ValueError:
                pass
            click.echo("Invalid choice, try again.")

        messages: list[dict] = []
        back_to_menu = False

        while not back_to_menu:
            try:
                user_input = click.prompt("\nYou", prompt_suffix="> ")
            except click.Abort:
                click.echo()
                return

            stripped = user_input.strip()
            if stripped == "/exit":
                click.echo("Goodbye.")
                return
            if stripped == "/menu":
                back_to_menu = True  # re-enter menu via the outer loop
                continue
            if not stripped:
                continue

            messages.append({"role": "user", "content": stripped})

            click.echo()
            try:
                stream = client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    stream=True,
                )
                collected = []
                for event in stream:
                    delta = event.choices[0].delta if event.choices else None
                    if delta and delta.content:
                        click.echo(delta.content, nl=False)
                        collected.append(delta.content)
                click.echo()
                messages.append({"role": "assistant", "content": "".join(collected)})
            except KeyboardInterrupt:
                click.echo()
            except Exception as exc:
                click.echo(f"\nError: {exc}", err=True)


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------
_AGENTS_INIT = """\
# agents/__init__.py — bare package marker.
# Drop .py files in this directory; aixon autodiscover() registers them
# automatically on startup. No list to maintain.
"""

_EXAMPLE_AGENT = """\
from aixon import LLMAgent, LLM


class GreeterAgent(LLMAgent):
    description = "Friendly greeter"
    # Replace 'gpt-4o-mini' with any supported model.
    llm = LLM("gpt-4o-mini", temperature=0.7)
    prompt = "You are a friendly assistant. Greet the user warmly."
"""

_MAIN_PY = """\
from aixon import Server, autodiscover

# Import every module in agents/, registering each Agent at startup.
autodiscover("agents")

# OpenAI-compatible API server. Set AUTH_API_KEY to require a Bearer token.
server = Server()
app = server.app  # ASGI app — for production: `uvicorn main:app --workers 4`

if __name__ == "__main__":
    server.serve(host="0.0.0.0", port=8000)
"""

_PYPROJECT = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "aixon[server,cli]",   # the server extra already pulls in uvicorn[standard]
]

[project.optional-dependencies]
all = ["aixon[all]"]
"""


@app.command(name="new")
@click.argument("name")
def new_command(name: str) -> None:
    """Scaffold a new consumer project."""
    import os

    sep = os.path.sep
    alt = os.path.altsep or ""
    if os.path.isabs(name) or name in (".", "..") or sep in name or (alt and alt in name):
        click.echo(f"Error: '{name}' is not a valid project name (must be a single path component).", err=True)
        raise SystemExit(1)

    base = os.path.join(os.getcwd(), name)
    if os.path.exists(base):
        click.echo(f"Error: directory '{name}' already exists.", err=True)
        raise SystemExit(1)

    os.makedirs(base)
    agents_dir = os.path.join(base, "agents")
    os.makedirs(agents_dir)

    with open(os.path.join(agents_dir, "__init__.py"), "w") as f:
        f.write(_AGENTS_INIT)

    with open(os.path.join(agents_dir, "greeter.py"), "w") as f:
        f.write(_EXAMPLE_AGENT)

    with open(os.path.join(base, "main.py"), "w") as f:
        f.write(_MAIN_PY)

    with open(os.path.join(base, "pyproject.toml"), "w") as f:
        f.write(_PYPROJECT.format(name=name))

    click.echo(f"Project '{name}' created.")
    click.echo(f"  cd {name}")
    click.echo(f"  pip install -e .")
    click.echo(f"  python main.py")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command(name="serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--package", "-p", default="agents", show_default=True,
              help="Package to autodiscover before serving.")
@click.option("--anthropic", "-a", is_flag=True, default=False,
              help="Also serve the Anthropic dialect under /anthropic "
                   "(in addition to OpenAI at /v1).")
def serve_command(host: str, port: int, package: str, anthropic: bool) -> None:
    """Start the aixon server."""
    # Probe the extra's deps explicitly: aixon.server.server itself imports
    # only stdlib at module level (fastapi/uvicorn are lazy), so the import
    # below SUCCEEDS on a bare install and the user would otherwise get a raw
    # ModuleNotFoundError traceback deep inside serve().
    import importlib.util

    missing = [m for m in ("fastapi", "uvicorn") if importlib.util.find_spec(m) is None]
    if missing:
        click.echo(
            "The server extra is required for 'serve' "
            f"(missing: {', '.join(missing)}). "
            "Install with: pip install 'aixon[server]'",
            err=True,
        )
        raise SystemExit(1)

    try:
        from aixon.server.server import Server
        from aixon.server.adapters.openai import OpenAIAdapter
    except ImportError:
        click.echo(
            "The server extra is required for 'serve'. "
            "Install with: pip install 'aixon[server]'",
            err=True,
        )
        raise SystemExit(1)

    _ensure_cwd_on_path()
    try:
        autodiscover(package)
    except (ImportError, ModuleNotFoundError, ValueError):
        click.echo(f"Warning: could not autodiscover package '{package}'.", err=True)

    adapters = [OpenAIAdapter()]
    if anthropic:
        from aixon.server.adapters.anthropic import AnthropicAdapter

        adapters.append(AnthropicAdapter(mount_prefix="/anthropic"))

    # Construct the singleton with the chosen adapters (raises if one was
    # already built with a different set — e.g. a main.py imported first).
    server = Server(adapters=adapters)
    dialects = "OpenAI + Anthropic" if anthropic else "OpenAI"
    click.echo(f"Starting aixon server ({dialects}) on {host}:{port} ...")
    server.serve(host=host, port=port)
