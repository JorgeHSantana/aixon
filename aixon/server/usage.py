"""Token counting for OpenAI-style ``usage``.

Lives in the Server layer, not on the neutral Message/Chunk types: the neutral
boundary carries no token counts. tiktoken is an optional extra
(``aixon[tiktoken]``); when absent, build_usage returns {} and the server omits
usage (graceful degradation, never an error)."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=32)
def _encoding(model: str):
    import tiktoken  # optional; raises ImportError if the extra is not installed

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(model: str, text: str) -> int | None:
    """Token count for ``text`` under ``model``'s encoding, or None if tiktoken
    is unavailable."""
    try:
        enc = _encoding(model)
        # disallowed_special=() treats special-token markers in user text
        # (e.g. "<|endoftext|>") as plain text; tiktoken's default raises
        # ValueError on them, which would break the graceful-degradation
        # promise for user-controlled content.
        return len(enc.encode(text or "", disallowed_special=()))
    except Exception:
        return None


def build_usage(model: str, prompt_text: str, completion_text: str) -> dict:
    """OpenAI-style usage dict, or {} when counting is unavailable."""
    prompt = count_tokens(model, prompt_text)
    completion = count_tokens(model, completion_text)
    if prompt is None or completion is None:
        return {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
