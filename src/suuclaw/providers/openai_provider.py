"""``create_model()`` — build the LangChain chat model used by the agent loop.

Configuration comes from the environment (``.env`` is loaded automatically):

===========================  ==========================================
``OPENAI_API_KEY``           required; the only mandatory variable
``OPENAI_MODEL``             default model id (fallback: ``gpt-4o-mini``)
``OPENAI_BASE_URL``          optional; point at any OpenAI-compatible gateway
``OPENAI_TIMEOUT``           optional request timeout in seconds (default 60)
``OPENAI_MAX_RETRIES``       optional retry count (default 2)
===========================  ==========================================

The API key is read lazily inside :func:`create_model`, never at import time, so
``--help`` / ``--list-tools`` keep working on a machine with no credentials.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_openai import ChatOpenAI

__all__ = ["DEFAULT_MODEL", "create_model", "load_environment"]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2

_env_loaded = False


def load_environment(
    dotenv_path: str | os.PathLike[str] | None = None,
    *,
    reload: bool = False,
) -> None:
    """Load ``.env`` into ``os.environ`` (idempotent; never overrides real env vars)."""
    global _env_loaded
    if _env_loaded and not reload:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        _env_loaded = True
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)
    _env_loaded = True


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def create_model(
    model: str | None = None,
    *,
    temperature: float = 0.0,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` instance ready for ``.bind_tools(...)``.

    Raises:
        RuntimeError: when ``OPENAI_API_KEY`` is missing, or a numeric env var is
            not parseable.
    """
    load_environment()

    resolved_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
    if not resolved_key or not resolved_key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export OPENAI_API_KEY before running."
        )

    resolved_model = (model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    resolved_base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
    resolved_timeout = (
        timeout if timeout is not None else _env_float("OPENAI_TIMEOUT", DEFAULT_TIMEOUT)
    )
    resolved_retries = max_retries
    if resolved_retries is None:
        resolved_retries = _env_int("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES)

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise RuntimeError("langchain-openai is not installed. Run: pip install -e .") from exc

    return ChatOpenAI(
        model=resolved_model,
        api_key=resolved_key.strip(),
        base_url=resolved_base_url,
        temperature=temperature,
        timeout=resolved_timeout,
        max_retries=resolved_retries,
        **kwargs,
    )
