"""Tool infrastructure: the tool contract plus the ``build_tools`` registry.

Design notes
------------
* Each tool is a small class with a ``ClassVar`` name/description/schema and a
  plain ``run(**kwargs) -> str`` method. Keeping tools class-based makes them
  stateless-per-call and trivial to unit test without a model.
* :func:`build_tools` converts them into ``StructuredTool`` objects, which is
  exactly what ``ChatOpenAI.bind_tools(...)`` wants.
* Concrete tools are imported lazily inside the registry functions, so
  ``registry`` -> ``file_tools`` -> ``registry`` never becomes a circular import
  at module load time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ..core.state import RuntimeState

__all__ = [
    "SuuClawTool",
    "READ_ONLY_TOOL_NAMES",
    "as_langchain_tools",
    "build_read_only_tools",
    "build_tools",
    "describe_tools",
    "tool_names",
]

#: Tools safe for a *verifier* to use. An allowlist rather than an exclusion
#: list, on purpose: a tool added to the registry later must not silently become
#: available to the node whose entire job is to check work without touching it.
READ_ONLY_TOOL_NAMES: tuple[str, ...] = ("file_read", "grep")


class SuuClawTool(ABC):
    """Common contract for every SuuClaw tool.

    Subclasses declare three class attributes and implement :meth:`run`:

    >>> class EchoTool(SuuClawTool):            # doctest: +SKIP
    ...     name = "echo"
    ...     description = "Echo the input back."
    ...     args_schema = EchoInput
    ...     def run(self, text: str) -> str:
    ...         return text
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    args_schema: ClassVar[type[BaseModel]]

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return a plain-text result.

        Tools return error *text* (``"Error: ..."``) instead of raising, so the
        model can read the failure and retry with better arguments.
        """

    def as_langchain_tool(self) -> StructuredTool:
        """Wrap this tool as a ``StructuredTool`` for ``model.bind_tools()``.

        The Python class name is carried in ``metadata`` because the wrapper hides
        it, and the actor prompt addresses tools by class name (``FileWriteTool``).
        """
        if not self.name or not self.description:
            raise ValueError(f"{type(self).__name__} must define name and description")
        return StructuredTool(
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
            func=self.run,
            metadata={"suuclaw_class": type(self).__name__},
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"


def _tool_classes() -> tuple[type[SuuClawTool], ...]:
    """Import and return the default tool set (kept lazy on purpose)."""
    from .bash_tool import BashTool
    from .file_tools import FileEditTool, FileReadTool, FileWriteTool
    from .grep_tool import GrepTool

    return (FileReadTool, FileWriteTool, FileEditTool, GrepTool, BashTool)


def _select_classes(
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> tuple[type[SuuClawTool], ...]:
    classes: Sequence[type[SuuClawTool]] = _tool_classes()
    if include is not None:
        wanted = {item.strip().lower() for item in include if item and item.strip()}
        unknown = wanted - {cls.name.lower() for cls in classes}
        if unknown:
            raise ValueError(f"unknown tool(s) in include: {sorted(unknown)}")
        classes = tuple(cls for cls in classes if cls.name.lower() in wanted)
    if exclude:
        unwanted = {item.strip().lower() for item in exclude if item and item.strip()}
        classes = tuple(cls for cls in classes if cls.name.lower() not in unwanted)
    return classes


def tool_names() -> list[str]:
    """Names of every tool registered by default, in registration order."""
    return [cls.name for cls in _tool_classes()]


def build_tools(
    state: RuntimeState,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[StructuredTool]:
    """Bind the tool set to ``state`` and return LangChain ``StructuredTool``s.

    >>> from suuclaw.core import RuntimeState      # doctest: +SKIP
    >>> state = RuntimeState.create("./sandbox")     # doctest: +SKIP
    >>> tools = build_tools(state)                   # doctest: +SKIP
    >>> sorted(t.name for t in tools)                # doctest: +SKIP
    ['bash', 'file_edit', 'file_read', 'file_write', 'grep']
    """
    tools = [cls(state).as_langchain_tool() for cls in _select_classes(include, exclude)]
    if not tools:
        raise ValueError("no tools selected; check include/exclude")
    names = [tool.name for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate tool names: {names}")
    return tools


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """The verifier's tool set: read, grep, the notepad, and web search — never write.

    Verification commands are run by the verifier *node*, not by the model, so
    the model needs no ``bash`` — and a verifier that can edit the workspace is
    not verifying anything. The stage-3 verifier prompt also lets it read the
    notepad for durable context and search the web for facts the workspace
    cannot answer, so those two ride along here rather than in
    :func:`build_tools`: putting ``web_search`` in the default registry would
    hand it to the code agent too, and the search agent is supposed to be the
    only researcher in the building.
    """
    from .notepad_tools import NotepadReadTool
    from .web_search_tool import WebSearchTool

    tools = build_tools(state, include=READ_ONLY_TOOL_NAMES)
    tools.append(NotepadReadTool(state).as_langchain_tool())
    tools.append(WebSearchTool(state).as_langchain_tool())
    return tools


def as_langchain_tools(tools: Iterable[Any]) -> list[StructuredTool]:
    """Normalise a mixed sequence of ``SuuClawTool``s and ready tools.

    Lets a caller write ``build_tools(state) + [TodoUpdateTool(state, store)]``
    without tracking which half still needs wrapping — the todo tools take an
    extra collaborator, so they cannot come out of :func:`build_tools`.

    Raises:
        TypeError: when an item is neither a ``SuuClawTool`` nor tool-like.
        ValueError: when two tools end up sharing a name, which would make
            ``execute_tool`` resolve the wrong one.
    """
    normalised: list[StructuredTool] = []
    for tool in tools:
        if isinstance(tool, SuuClawTool):
            normalised.append(tool.as_langchain_tool())
        elif hasattr(tool, "invoke") and hasattr(tool, "name"):
            normalised.append(tool)
        else:
            raise TypeError(f"cannot use {type(tool).__name__} as a tool")
    names = [tool.name for tool in normalised]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate tool names: {names}")
    return normalised


def describe_tools(
    state: RuntimeState,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return ``(name, description, signature)`` rows for CLI/GUI display."""
    rows: list[tuple[str, str, str]] = []
    for cls in _select_classes(include, exclude):
        params = []
        for field_name, model_field in cls.args_schema.model_fields.items():
            annotation = model_field.annotation
            raw_name = getattr(annotation, "__name__", None) or str(annotation)
            type_name = raw_name.replace("typing.", "")
            if model_field.is_required():
                params.append(f"{field_name}: {type_name}")
            else:
                params.append(f"{field_name}: {type_name} = {model_field.default!r}")
        rows.append((cls.name, cls.description, ", ".join(params)))
    return rows
