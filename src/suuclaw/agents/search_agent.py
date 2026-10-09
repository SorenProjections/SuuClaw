"""The searchAgent node: the research specialist of SuuClaw's multi-agent flow.

Stage 3 replaces the single do-everything actor with specialists, and this is
the research half — a model whose only capability is searching the web. The
split exists because research and implementation want different instincts: a
researcher reads widely and quotes sources, an implementer edits a workspace and
runs commands, and one prompt asked to do both does neither well.

Where it sits
-------------
The supervisor (the graph's planner node) delegates to this agent through its
``call_search_agent`` tool; several specialists can run inside one planner
superstep::

    planner ──▶ [ searchAgent · codeAgent ] ──▶ verifier ──▶ final

The function a graph node calls is :func:`run_search_agent`.

Why the loop is written out here rather than delegated
------------------------------------------------------
:func:`~suuclaw.core.agent.stream_react_events` would run the same
model⇄tool⇄model shape, but it keeps only the model's closing prose. A research
step's value is also in the *structured* half — the queries it ran, the sources
it found, the answer Tavily synthesised — which the planner and the code agent
read afterwards. So the loop is written out here, and the tool is invoked
directly (:meth:`~suuclaw.tools.web_search_tool.WebSearchTool.search`) rather
than through the generic loop's serialise-then-reparse envelope path. The result
is that one function produces both the prose summary and the source list.

What it emits
-------------
On ``writer`` — LangGraph's custom stream, so a front end can render progress
live:

==================  ==========================================================
``tool_call``       A search was requested. Keys: ``node``, ``loop``, ``id``,
                    ``name``, ``args``.
``memory``          The layered-memory snapshot this step's brief was built
                    with. Keys: ``node``, ``memory`` (the full three-layer
                    snapshot). Emitted once, before the first model call.
``search_results``  A search finished. Keys: ``node``, ``loop``, ``id``,
                    ``query``, ``ok``, ``answer``, ``sources`` (title, url and
                    score per source), ``result_count``, ``error``.
``final_answer``    The loop closed its own turn. The workflow driver relabels
                    this to ``node_summary``, because a node's last words are
                    not the run's answer — only the final node writes that.
``retry``, ``error``  A model call failed transiently and is being retried, or
                    the loop could not continue.
==================  ==========================================================

What it returns
---------------
A dict, never a state update: the caller decides how to fold it into graph state
(usually ``research_notes`` for the next node, and ``messages`` through the
channel's reducer).

``messages`` holds only the loop's own turns — the system/human scaffolding is
rebuilt from state on every call, so replaying it would splice half
conversations together.

No files, no commands: the tool list is the sandbox here. A search agent cannot
write, which is what makes it safe to hand it a wide-open instruction.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..core.agent import (
    DEFAULT_RETRIES,
    DEFAULT_RETRY_BACKOFF,
    is_transient_error,
    message_text,
)
from ..core.state import RuntimeState
from ..prompts.stage3 import SEARCH_AGENT_PROMPT
from ..tools.web_search_tool import (
    DEFAULT_MAX_RESULTS,
    WebSearchTool,
    failure_envelope,
)

__all__ = [
    "DEFAULT_SEARCH_LOOPS",
    "SEARCH_AGENT_PROMPT",
    "SEARCH_NODE_NAME",
    "SearchAgentError",
    "make_search_agent",
    "run_search_agent",
]

#: ReAct loops one research step gets. Lower than the code agent's ten on
#: purpose: a search agent spends one loop per query, so four rounds of
#: search → read → refine is already a thorough answer, and the fifth would only
#: re-read what it already has.
DEFAULT_SEARCH_LOOPS = 4

#: Tag on every event this agent emits, so a renderer can label its section.
SEARCH_NODE_NAME = "search"

#: Per-block cap on what state hands the agent. The task and the inherited notes
#: both end up in one prompt, and neither should crowd the other out.
MAX_BLOCK_CHARS = 4000

#: How many sources the deterministic summary lists before it starts counting.
MAX_SUMMARY_SOURCES = 10


class SearchAgentError(ValueError):
    """Raised when the state handed to the agent is missing something it cannot invent."""


# --------------------------------------------------------------------------- #
# state access
# --------------------------------------------------------------------------- #
def _runtime(state: Mapping[str, Any]) -> RuntimeState:
    """The ``RuntimeState`` the web search tool needs, or a clear complaint about it."""
    if not isinstance(state, Mapping):
        kind = type(state).__name__
        raise SearchAgentError(f"search agent needs a graph state mapping, got {kind}")
    runtime = state.get("runtime")
    if runtime is None:
        raise SearchAgentError(
            "search agent state has no 'runtime' — seed the run with "
            "{'task': ..., 'runtime': RuntimeState.create(workspace)}"
        )
    return runtime


def _default_model(state: Mapping[str, Any]) -> Any:
    """Build a model from the runtime's metadata (the CLI records the model id there).

    Imported lazily so that merely importing this module never requires an API key.
    """
    from ..providers.openai_provider import create_model

    metadata = getattr(_runtime(state), "metadata", None) or {}
    return create_model(metadata.get("model"), temperature=0.0)


def _stream_writer() -> Any | None:
    """LangGraph's custom-stream writer, or ``None`` outside a running graph.

    Mirrors the helper in :mod:`suuclaw.graph.nodes` instead of importing it: an
    agent package reaching into a graph module's privates is exactly the coupling
    that would make a stage-3 flow hard to lift out of its stage-2 home.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001 - no run, no stream, no problem
        return None


def _emit(writer: Any | None, event: Mapping[str, Any]) -> None:
    """Forward one event to the custom stream, tagged with this agent's node.

    A broken consumer must not abort real research, hence the swallow.
    """
    if writer is None:
        return
    payload = dict(event)
    payload.setdefault("node", SEARCH_NODE_NAME)
    try:
        writer(payload)
    except Exception:  # noqa: BLE001 - a dead stream is not a reason to stop
        pass


# --------------------------------------------------------------------------- #
# prompt assembly
# --------------------------------------------------------------------------- #
def _clip(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... (+{len(text) - limit} more character(s) withheld)"


def _search_agent_input(
    state: Mapping[str, Any],
    instruction: str,
    memory: Mapping[str, Any],
) -> str:
    """The agent's brief: task, what to research now, and the layered memory.

    ``memory`` is the assembled three-layer snapshot the caller built; its
    ``working_memory.research_notes`` already carries whatever earlier search
    steps established, so the notes block the brief used to render by hand is
    gone — one source of truth, the same trade the code agent made.
    """
    from ..graph.memory import append_memory_block  # lazy: see the note in _research

    task = _clip(str(state.get("task") or ""), 2000) or "(no task given)"
    ask = _clip(str(instruction or ""), 2000) or "(no specific instruction — answer the task above)"
    return append_memory_block(
        "\n\n".join(
            [
                f"# Task\n{task}",
                f"# What to research now\n{ask}",
                "# Your job\n"
                "The only tool available to you is `web_search` (the WebSearchTool "
                "mentioned in your instructions). Search for the facts the task needs — "
                "several focused queries beat one broad one — and read the answers and "
                "extracts that come back before deciding whether anything is still "
                "missing. Then reply with a CONCISE research summary and the useful "
                "source URLs. Do not write files and do not produce application code.",
            ]
        ),
        memory,
    )


# --------------------------------------------------------------------------- #
# one loop
# --------------------------------------------------------------------------- #
def _positive_int(raw: Any, default: int) -> int:
    """A count read off loose input: the caller's number when it is usable."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 1 else default


def _invoke(
    agent: Any,
    messages: list[Any],
    *,
    loop: int,
    writer: Any | None,
    retries: int,
    retry_backoff: float,
) -> tuple[Any | None, str]:
    """One model call, retrying only *transient* failures.

    Returns ``(response, "")`` or ``(None, reason)``. The policy is the actor's:
    a dropped connection, a 5xx or a 429 is worth another attempt, while a bad
    model name or a bad key fails identically every time and so is reported
    immediately. Retries are announced on the stream, so a stalled run looks
    stalled rather than hung.
    """
    for attempt in range(retries + 1):
        try:
            return agent.invoke(messages), ""
        except Exception as exc:  # noqa: BLE001 - surfaced as an event, not a traceback
            message = f"{type(exc).__name__}: {exc}"
            if attempt >= retries or not is_transient_error(exc):
                _emit(writer, {"type": "error", "loop": loop, "message": message})
                return None, message

            delay = retry_backoff * (2**attempt)
            _emit(
                writer,
                {
                    "type": "retry",
                    "loop": loop,
                    "attempt": attempt + 1,
                    "max_retries": retries,
                    "delay": delay,
                    "message": message,
                },
            )
            time.sleep(delay)

    return None, "the model call did not complete"  # pragma: no cover - loop returns or raises


def _search_once(tool: WebSearchTool, raw_args: Any) -> dict[str, Any]:
    """Run one ``web_search`` call and return its envelope — never raising.

    Model-supplied arguments are trusted no further than the tool's own schema
    would trust them: a missing or empty ``query`` becomes a readable failure the
    model can correct on its next loop, rather than an exception that ends the
    step.
    """
    if not isinstance(raw_args, Mapping):
        kind = type(raw_args).__name__
        return failure_envelope("", f"web_search arguments must be an object, got {kind}")

    query = str(raw_args.get("query") or "").strip()
    if not query:
        return failure_envelope("", "web_search needs a non-empty 'query' argument")

    budget = _positive_int(raw_args.get("max_results"), DEFAULT_MAX_RESULTS)
    try:
        return tool.search(query, max_results=budget)
    except Exception as exc:  # noqa: BLE001 - a backstop; the tool reports its own failures
        return failure_envelope(query, f"{type(exc).__name__}: {exc}")


def _source_rows(envelope: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The title/url/score view of a search's results, for events and the summary."""
    rows: list[dict[str, Any]] = []
    for item in envelope.get("results") or []:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        rows.append(
            {
                "title": str(item.get("title") or url).strip(),
                "url": url,
                "score": item.get("score"),
            }
        )
    return rows


def _fallback_summary(
    *,
    queries: Sequence[str],
    answers: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
    error: str,
    loops: int,
    answered: bool,
) -> str:
    """The step's findings in the step's own words, for when the model produced none.

    A research step that spent its loops searching still owes the next node what
    it found, and one that failed owes it the reason. A loop that ends silently
    would leave the planner to guess which of the two happened.
    """
    lines: list[str] = []
    if error:
        lines.append(f"The web search step failed: {error}")
    if queries:
        noun = "query" if len(queries) == 1 else "queries"
        lines.append(f"Searched {len(queries)} {noun}: {'; '.join(queries)}")
    if answers:
        lines.append(answers[0])
    if sources:
        lines.append(f"{len(sources)} source(s) collected:")
        for row in sources[:MAX_SUMMARY_SOURCES]:
            lines.append(f"- {row.get('title', '')} — {row.get('url', '')}")
        if len(sources) > MAX_SUMMARY_SOURCES:
            lines.append(f"- (+{len(sources) - MAX_SUMMARY_SOURCES} more)")
    if not answered and not error:
        lines.append(
            f"The agent used all {loops} of its loops without a closing summary; "
            "the findings above are what it collected."
        )
    if not lines:
        return "No research was gathered."
    return "\n".join(lines)


def _research(
    state: Mapping[str, Any],
    instruction: str,
    *,
    model: Any,
    writer: Any | None,
    max_loops: int,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> dict[str, Any]:
    """The loop itself. See :func:`run_search_agent` for the contract."""
    runtime = _runtime(state)
    tool = WebSearchTool(runtime)
    agent = model.bind_tools([tool.as_langchain_tool()])

    # The assembled three-layer snapshot, announced before the first call.
    # Imported here rather than at module level: graph/__init__ pulls this
    # module in via the nodes, so a top-level import would be the cycle the
    # lazy-import rule exists for (same trade as the code agent).
    from ..graph.memory import build_layered_memory, memory_event

    memory = build_layered_memory(state, node="searchAgent")
    _emit(writer, memory_event(memory, node=SEARCH_NODE_NAME))

    messages: list[Any] = [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(content=_search_agent_input(state, instruction, memory)),
    ]

    produced: list[Any] = []  # the loop's own turns; the scaffolding is not replayed
    tool_events: list[dict[str, Any]] = []
    queries: list[str] = []
    sources: list[dict[str, Any]] = []
    answers: list[str] = []
    seen_urls: set[str] = set()
    searches_ok = 0
    calls_made = 0
    first_failure = ""
    last_content = ""
    last_error: str | None = None
    answered = False
    loops_used = 0

    for loop in range(1, max_loops + 1):
        loops_used = loop
        response, failure = _invoke(
            agent,
            messages,
            loop=loop,
            writer=writer,
            retries=retries,
            retry_backoff=retry_backoff,
        )
        if response is None:
            last_error = failure
            break

        messages.append(response)
        produced.append(response)
        content = message_text(getattr(response, "content", "")).strip()
        if content:
            last_content = content

        calls = list(getattr(response, "tool_calls", None) or [])
        invalid = list(getattr(response, "invalid_tool_calls", None) or [])
        if not calls:
            if invalid:
                # The model's tool-call JSON did not survive the hop. Say so rather
                # than pass the turn off as a clean close.
                preview = [str(item)[:200] for item in invalid]
                last_error = f"model produced malformed tool call(s): {preview}"
                _emit(writer, {"type": "error", "loop": loop, "message": last_error})
            else:
                answered = True
            break

        for position, call in enumerate(calls, start=1):
            name = str(call.get("name") or tool.name)
            raw_args = call.get("args")
            call_id = str(call.get("id") or f"{SEARCH_NODE_NAME}-{loop}-{position}")
            shown_args = dict(raw_args) if isinstance(raw_args, Mapping) else {}
            _emit(
                writer,
                {
                    "type": "tool_call",
                    "loop": loop,
                    "id": call_id,
                    "name": name,
                    "args": shown_args,
                },
            )

            envelope = _search_once(tool, raw_args)
            calls_made += 1
            ok = bool(envelope.get("ok"))
            query = str(envelope.get("query") or "").strip()
            answer = str(envelope.get("answer") or "").strip()
            error = str(envelope.get("error") or "")
            found = _source_rows(envelope)

            if ok:
                searches_ok += 1
                if query and query not in queries:
                    queries.append(query)
                if answer and answer not in answers:
                    answers.append(answer)
                for row in found:
                    url = str(row.get("url") or "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append(row)
            elif not first_failure:
                first_failure = error or "the web search failed"

            _emit(
                writer,
                {
                    "type": "search_results",
                    "loop": loop,
                    "id": call_id,
                    "query": query,
                    "ok": ok,
                    "answer": answer,
                    "sources": found,
                    "result_count": len(found),
                    "error": error,
                },
            )

            reply = ToolMessage(
                content=json.dumps(envelope, ensure_ascii=False, default=str),
                tool_call_id=call_id,
            )
            messages.append(reply)
            produced.append(reply)
            tool_events.append(
                {
                    "loop": loop,
                    "id": call_id,
                    "name": name,
                    "args": shown_args,
                    "ok": ok,
                    "query": query,
                    "answer": answer,
                    "sources": found,
                    "result_count": len(found),
                    "error": error,
                }
            )

    if last_error is not None:
        ok, error = False, last_error
    elif calls_made and not searches_ok:
        # Every query failed. `missing TAVILY_API_KEY` is the common case, and it
        # has to travel out: a step that reports "ok" here would leave the planner
        # treating an unconfigured key as "the web knows nothing".
        ok, error = False, first_failure or "every web_search call failed"
    elif not (searches_ok or last_content):
        ok, error = (
            False,
            (
                f"the search agent used all {loops_used} of its loops without a search "
                "result or a closing summary"
            ),
        )
    else:
        ok, error = True, ""

    # A closing turn is a report; anything the model said *before* an unfinished
    # tool call is not — replaying that out of context reads as a conclusion the
    # step never reached. So an exhausted loop is summarised from the data it
    # actually collected instead. (``cli/app.py`` draws the same distinction when
    # it decides what to show for an inner loop's closing turn.)
    summary = (
        last_content
        if (answered and last_content)
        else _fallback_summary(
            queries=queries,
            answers=answers,
            sources=sources,
            error=error,
            loops=loops_used,
            answered=answered,
        )
    )

    if not answered and last_error is None:
        # The loop ran out of budget with the turn still open. The run's driver
        # needs to hear that too, so the node does not look like it finished.
        _emit(
            writer,
            {
                "type": "final_answer",
                "content": summary,
                "loops": loops_used,
                "exhausted": True,
                "error": "",
                "interrupted": False,
            },
        )
    else:
        _emit(
            writer,
            {
                "type": "final_answer",
                "content": summary,
                "loops": loops_used,
                "exhausted": False,
                "error": last_error or "",
                "interrupted": last_error is not None,
            },
        )

    return {
        "ok": ok,
        "summary": summary,
        "queries": queries,
        "sources": sources,
        "messages": produced,
        "tool_events": tool_events,
        "error": error,
    }


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #
def run_search_agent(
    state: Mapping[str, Any],
    instruction: str,
    *,
    writer: Any | None = None,
    max_loops: int = DEFAULT_SEARCH_LOOPS,
    model: Any | None = None,
) -> dict[str, Any]:
    """Research ``instruction`` on the web and report what was found.

    Args:
        state: The graph state. ``runtime`` (the policy every tool needs) and
            ``task`` (the user's original request) are read from it, as is
            ``research_notes`` — whatever an earlier search step established.
        instruction: What *this* step should research. Usually one planner todo,
            e.g. "which Python libraries parse ISO 8601 durations, and what are
            their licences?".
        writer: LangGraph's custom-stream writer. When omitted it is picked up from
            the running graph, so a node can call this without passing one, and a
            plain script call works with no stream at all.
        max_loops: Model⇄tool iterations before the step gives up. One loop per
            search, plus the closing turn that writes the summary.
        model: Pre-built chat model. Built from the runtime's metadata when
            omitted; the supervisor passes its own so one model instance serves
            the whole graph, and a test can inject a fake.

    Returns:
        ``{"ok", "summary", "queries", "sources", "messages", "tool_events",
        "error"}``.

        ``ok`` is ``False`` when the model call failed, when every search failed
        (``missing TAVILY_API_KEY`` is the usual one), or when the step spent all
        its loops with nothing to show for it; ``error`` then carries the reason
        and is ``""`` otherwise.

        ``summary`` is the model's closing prose when it wrote one, and a
        deterministic digest of what was collected when it did not. ``queries``
        and ``sources`` are de-duplicated in first-seen order, and per-query
        answers live in ``tool_events`` (and inside ``summary``).

    Raises:
        SearchAgentError: when ``state`` has no ``runtime``. A missing Tavily key
            is *not* an error of this kind — the step reports ``ok: False``
            instead, because "not configured" is an outcome the planner can route
            around.
    """
    return _research(
        state,
        instruction,
        model=model if model is not None else _default_model(state),
        writer=writer if writer is not None else _stream_writer(),
        max_loops=_positive_int(max_loops, DEFAULT_SEARCH_LOOPS),
    )


def make_search_agent(
    model: Any,
    *,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> Any:
    """Search agent bound to one pre-built model (the form a compiled graph wants).

    Mirrors the ``make_*_node`` factories in :mod:`suuclaw.graph.nodes`: a graph
    builds its model once and hands it to every node, and a test can hand it a
    fake without an API key or a network.
    """

    def agent(
        state: Mapping[str, Any],
        instruction: str,
        *,
        writer: Any | None = None,
        max_loops: int = DEFAULT_SEARCH_LOOPS,
    ) -> dict[str, Any]:
        return _research(
            state,
            instruction,
            model=model,
            writer=writer if writer is not None else _stream_writer(),
            max_loops=_positive_int(max_loops, DEFAULT_SEARCH_LOOPS),
            retries=retries,
            retry_backoff=retry_backoff,
        )

    agent.__name__ = "run_search_agent"
    return agent
