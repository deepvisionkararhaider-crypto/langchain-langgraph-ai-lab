"""Shared LLM helper for the langchain-langgraph-ai-lab projects.

Provides a Groq-backed LangChain chat model plus a robust text extractor.

Why the extractor? Groq's ``gpt-oss`` models are *reasoning* models: the raw
completion sometimes arrives with an empty ``message.content`` while the actual
text lives in ``message.reasoning``. :func:`extract_text` normalises both cases
so every project gets usable text regardless of the model.

Keys are read from (in order):
  1. ``st.secrets``  (Streamlit Community Cloud / local ``secrets.toml``)
  2. environment variables (``.env`` / shell)
"""
from __future__ import annotations

import os
from typing import Any

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from Streamlit secrets first, then the environment."""
    try:  # pragma: no cover - depends on runtime
        import streamlit as st

        if name in st.secrets:  # type: ignore[operator]
            value = st.secrets[name]
            if value not in (None, ""):
                return str(value)
    except Exception:
        pass
    env = os.environ.get(name)
    return env if env not in (None, "") else default


def get_llm(
    temperature: float = 0.3,
    model: str | None = None,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
):
    """Return a configured ``ChatGroq`` instance.

    Raises a clear error when the API key is missing so the Streamlit UI can
    show a friendly message instead of a stack trace.
    """
    from langchain_groq import ChatGroq

    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to Streamlit secrets "
            "(Settings → Secrets) or export it as an environment variable."
        )
    model = model or get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    kwargs: dict[str, Any] = dict(
        model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key
    )
    # Only attach reasoning_effort when explicitly requested. Some models
    # reject the parameter, so this stays opt-in to remain model-agnostic.
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def _flatten(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "".join(parts)
    return str(content)


def extract_text(message: Any) -> str:
    """Return plain text from a LangChain ``AIMessage`` (or a raw string)."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message

    text = _flatten(getattr(message, "content", message))
    if text.strip():
        return text

    extra = getattr(message, "additional_kwargs", {}) or {}
    for key in ("reasoning", "reasoning_content"):
        if extra.get(key):
            return _flatten(extra[key])

    meta = getattr(message, "response_metadata", {}) or {}
    for key in ("reasoning", "reasoning_content"):
        if meta.get(key):
            return _flatten(meta[key])

    return text


def llm_text(llm, prompt: str, **kwargs) -> str:
    """Invoke ``llm`` on ``prompt`` and return the extracted text."""
    return extract_text(llm.invoke(prompt, **kwargs))


def missing_key_message() -> str:
    return (
        "🔑 **GROQ_API_KEY not configured.**\n\n"
        "Set it as a Streamlit secret (Settings → Secrets) or an env var, then "
        "reload. Get a free key at https://console.groq.com/keys."
    )
