"""``web_search`` tool — Tavily-backed web search for the searchAgent.

This is the only external capability :mod:`suuclaw.agents.search_agent` has, so
the entire network surface of the multi-agent flow lives in this one file: one
``TavilyClient.search()`` call per query, mapped into a JSON-serialisable
envelope::

    {"ok": True,
     "query": "python iso 8601 duration parser",
     "answer": "isodate and durationpy are the two maintained options ...",
     "results": [{"title": "isodate", "url": "https://...", "content": "...",
                  "score": 0.97}],
     "error": ""}

Two entry points, and the split is deliberate:

``search()``
    Returns that dict — parsed data, not a string a caller would have to parse
    back out of a repr. It is what the searchAgent collects into ``queries`` /
    ``sources`` / ``answers``.
``run()``
    The :class:`~suuclaw.tools.registry.SuuClawTool` contract, which is
    text-in/text-out because a tool's return value travels to the model as a
    ``ToolMessage`` body. It serialises the same envelope as JSON.

Failures are data, never exceptions. A missing key, a missing package, a timeout
or a 401 from Tavily each come back as ``ok: False`` plus a reason, which is what
lets a caller tell the model "the search failed, say so" instead of the run dying
inside the tool.

Credentials are read at call time, never at import time, so importing this module
works on a machine that has no key at all:

* ``state.env["TAVILY_API_KEY"]`` — a per-run override, the same way
  ``RuntimeState`` configures everything else it owns;
* ``TAVILY_API_KEY`` in the environment, ``.env`` included.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from .registry import SuuClawTool

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_SEARCH_DEPTH",
    "MAX_RESULTS_LIMIT",
    "MISSING_API_KEY_ERROR",
    "MISSING_PACKAGE_ERROR",
    "RESULT_CONTENT_CHARS",
    "WebSearchInput",
    "WebSearchTool",
    "failure_envelope",
]

#: Environment variable holding the Tavily key.
API_KEY_ENV = "TAVILY_API_KEY"

#: The exact text promised when no key is configured. The searchAgent surfaces it
#: as its own failure reason, so it is a constant rather than a literal here.
MISSING_API_KEY_ERROR = "missing TAVILY_API_KEY"

MISSING_PACKAGE_ERROR = "tavily-python is not installed (pip install tavily-python)"

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 20

#: ``basic`` costs one Tavily credit per query and still returns the synthesised
#: answer plus page extracts; ``advanced`` costs two for a deeper crawl. Research
#: steps ask several focused questions rather than one deep one, so the cheap mode
#: is the right default — pass ``search_depth="advanced"`` to override it.
DEFAULT_SEARCH_DEPTH = "basic"

#: Per-result extract budget. Tavily returns a few thousand characters per page,
#: and five of those unclipped would crowd the plan out of the next prompt.
RESULT_CONTENT_CHARS = 2000


def failure_envelope(query: str, error: str) -> dict[str, Any]:
    """The envelope for a search that could not run.

    Kept uniform with the success shape on purpose: a caller can read
    ``answer`` / ``results`` without checking ``ok`` first, and a model reading
    the JSON sees precisely which key is missing.
    """
    return {"ok": False, "query": query, "answer": "", "results": [], "error": error}


class WebSearchInput(BaseModel):
    """Arguments for :class:`WebSearchTool`."""

    query: str = Field(
        description=(
            "What to search for, phrased as a question or a keyword phrase. One "
            "focused query per call beats one broad one."
        )
    )
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_RESULTS_LIMIT,
        description=f"Maximum sources to return (1-{MAX_RESULTS_LIMIT}).",
    )


class WebSearchTool(SuuClawTool):
    """Search the public web through Tavily; return an answer plus its sources."""

    name = "web_search"
    description = (
        "Search the public web and return a synthesised answer plus the source "
        "pages, each with a title, URL, text extract and relevance score. Use it "
        "for facts that the workspace cannot answer, and prefer official or "
        "encyclopedia-style sources when several agree. Read-only: it cannot "
        "change the workspace."
    )
    args_schema = WebSearchInput

    # -- programmatic entry point -----------------------------------------
    def search(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        **options: Any,
    ) -> dict[str, Any]:
        """Run one Tavily search and return the envelope described in the module docstring.

        Never raises: every failure mode — empty query, missing key, missing
        package, transport error, a payload that is not an object — comes back as
        ``ok: False`` with a reason a caller can act on or pass to the model.

        Extra keyword arguments go through to ``TavilyClient.search`` untouched
        (``topic``, ``include_domains``, ``days``, ...), so the schema the model
        sees stays small while a programmatic caller keeps the full API.
        """
        clean = str(query or "").strip()
        if not clean:
            return failure_envelope("", "query must not be empty")

        budget = _as_count(max_results, DEFAULT_MAX_RESULTS)
        api_key = self._api_key()
        if not api_key:
            return failure_envelope(clean, MISSING_API_KEY_ERROR)

        client, problem = self._client(api_key)
        if client is None:
            return failure_envelope(clean, problem)

        try:
            payload = client.search(
                clean,
                max_results=budget,
                include_answer=True,
                search_depth=DEFAULT_SEARCH_DEPTH,
                **options,
            )
        except Exception as exc:  # noqa: BLE001 - the envelope is how a caller hears about it
            return failure_envelope(clean, f"{type(exc).__name__}: {exc}")

        if not isinstance(payload, Mapping):
            kind = type(payload).__name__
            return failure_envelope(clean, f"Tavily returned {kind}, expected an object")

        return {
            "ok": True,
            "query": clean,
            "answer": str(payload.get("answer") or "").strip(),
            "results": self._rows(payload.get("results")),
            "error": "",
        }

    # -- SuuClawTool contract ---------------------------------------------
    def run(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> str:
        """The registry's text-in/text-out entry point: the envelope, as JSON.

        A tool's return value becomes the body of a ``ToolMessage``, so it has to
        be text — but the object inside it is exactly what :meth:`search` returns,
        which is the entry point a programmatic caller should use instead of
        re-parsing this.
        """
        envelope = self.search(query, max_results=max_results)
        return self.state.truncate(
            json.dumps(envelope, ensure_ascii=False, default=str),
            note="search result truncated",
        )

    # -- internals ---------------------------------------------------------
    def _api_key(self) -> str:
        """The key for this run: the state's own env first, then the environment."""
        injected = str(self.state.env.get(API_KEY_ENV) or "").strip()
        if injected:
            return injected

        # Lazy, and for the same reason `create_model` is: importing this module
        # must not depend on `.env` existing or parse it twice.
        from ..providers.openai_provider import load_environment

        load_environment()
        return (os.getenv(API_KEY_ENV) or "").strip()

    @staticmethod
    def _client(api_key: str) -> tuple[Any | None, str]:
        """Build a ``TavilyClient``, reporting a missing package or key shape as data."""
        try:
            from tavily import TavilyClient
        except ImportError:
            return None, MISSING_PACKAGE_ERROR
        try:
            return TavilyClient(api_key=api_key), ""
        except Exception as exc:  # noqa: BLE001 - constructor validation is the client's call
            return None, f"{type(exc).__name__}: {exc}"

    def _rows(self, raw: Any) -> list[dict[str, Any]]:
        """Normalise Tavily's result rows, clipping each extract to the budget."""
        rows: list[dict[str, Any]] = []
        for item in raw or []:
            if not isinstance(item, Mapping):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            rows.append(
                {
                    "title": str(item.get("title") or url).strip(),
                    "url": url,
                    "content": self.state.truncate(
                        str(item.get("content") or "").strip(),
                        limit=RESULT_CONTENT_CHARS,
                        note="result extract truncated",
                    ),
                    "score": _as_score(item.get("score")),
                }
            )
        return rows


def _as_count(raw: Any, default: int) -> int:
    """A result budget read off loose input, clamped to what the API accepts."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return min(max(value, 1), MAX_RESULTS_LIMIT)


def _as_score(raw: Any) -> float:
    """Tavily's relevance score, or ``0.0`` when it is missing or not a number."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0
