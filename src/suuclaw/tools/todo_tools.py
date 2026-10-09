"""Todo tools: the plan as *data* the model reads and rewrites.

Why these are not like the other tools
--------------------------------------
``file_read`` / ``bash`` operate on the workspace, so they take only a
:class:`~suuclaw.core.state.RuntimeState`. The todo tools operate on the
plan, which lives in the *graph state*, not on disk — so they are constructed
with an extra collaborator, a :class:`TodoStore`, and the owning node reads the
store back into ``SuuClawGraphState`` after the model is done.

``TodoWriteTool`` plays two roles at once, and they agree by construction:

1. **Structured-output schema.** The planner binds it and reads the arguments
   off the emitted tool call — no prose parsing, no regex. Which is the whole
   point: a plan is data, so it should never have to survive a round trip
   through markdown.
2. **A real tool.** ``run`` writes those same arguments into the store, so the
   tool is honest — anyone who *does* invoke it gets the plan recorded.

Layering note: this module deliberately does not import from ``suuclaw.graph``,
even for the ``TodoItem`` type — it does so under ``TYPE_CHECKING`` only. A row
is just a dict at runtime, and keeping the runtime dependency one-way
(``graph`` → ``tools``) is what lets ``graph/__init__`` export the nodes without
a circular import.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..core.state import RuntimeState
from .registry import SuuClawTool

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime cycle
    from ..graph.state import TodoItem

__all__ = [
    "STATUSES",
    "TodoStore",
    "TodoUpdateInput",
    "TodoUpdateTool",
    "TodoWriteInput",
    "TodoWriteTool",
    "PlannedTodo",
    "build_todo_tools",
    "todo_update_tool",
    "todo_write_tool",
]

#: Valid values for ``TodoItem.status``, in rough lifecycle order.
STATUSES = ("pending", "in_progress", "completed", "blocked")

PENDING = "pending"

#: Cap on how much of the plan a prompt carries, so a runaway planner cannot
#: blow up every downstream prompt.
MAX_RENDERED_TODOS = 50


def _row(raw: Any) -> Mapping[str, Any]:
    """Coerce a todo row to a plain mapping.

    ``StructuredTool`` may hand ``run`` either dicts or pydantic models
    depending on how it validated the arguments, so accept both.
    """
    if isinstance(raw, Mapping):
        return raw
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        result = dump()
        if isinstance(result, Mapping):
            return result
    return {}


@dataclass(slots=True)
class TodoStore:
    """The live plan, mutated by the actor as it works.

    Holds plain dicts so that :meth:`snapshot` can be dropped straight into the
    ``todos`` channel of :class:`~suuclaw.graph.state.SuuClawGraphState`.
    """

    items: list[TodoItem] = field(default_factory=list)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_items(cls, items: Iterable[Any] | None = None) -> TodoStore:
        """Build a store seeded from an existing plan (e.g. from graph state)."""
        store = cls()
        store.replace(items or ())
        return store

    def replace(self, items: Iterable[Any]) -> list[TodoItem]:
        """Install ``items`` as the whole plan; returns the normalised rows.

        Ids are repaired rather than trusted: a blank id becomes the 1-based
        position *among the kept rows*, and a duplicate gets ``-<n>`` appended. A
        model that numbers two steps "1" is a normal failure mode, and it must not
        produce a plan whose rows cannot be addressed.

        Numbering by kept position rather than by input index matters because rows
        with no content are dropped — numbering by input index leaves gaps, so a
        five-row plan that loses a row ends up with ids like ``['3', '5']``.
        """
        self.items.clear()
        seen: set[str] = set()
        position = 0

        for raw in items:
            row = _row(raw)
            content = str(row.get("content") or "").strip()
            if not content:
                continue  # a step with no description is not a step

            position += 1
            todo_id = str(row.get("id") or "").strip() or str(position)
            if todo_id in seen:
                todo_id = f"{todo_id}-{position}"
            seen.add(todo_id)

            status = str(row.get("status") or PENDING).strip().lower()
            self.items.append(
                {
                    "id": todo_id,
                    "content": content,
                    "status": status if status in STATUSES else PENDING,
                    "note": str(row.get("note") or "").strip(),
                }
            )
        return list(self.items)

    # -- queries ----------------------------------------------------------
    def find(self, todo_id: str) -> TodoItem | None:
        """Look a todo up by id, tolerating case and surrounding whitespace."""
        wanted = str(todo_id or "").strip()
        if not wanted:
            return None
        for item in self.items:
            if item["id"] == wanted:
                return item
        folded = wanted.casefold()
        for item in self.items:
            if item["id"].casefold() == folded:
                return item
        return None

    def ids(self) -> list[str]:
        return [item["id"] for item in self.items]

    def snapshot(self) -> list[TodoItem]:
        """Copy of the plan, safe to store in graph state (no shared dicts)."""
        return [dict(item) for item in self.items]  # type: ignore[misc]

    def counts(self) -> dict[str, int]:
        tally = {status: 0 for status in STATUSES}
        for item in self.items:
            tally[item["status"]] = tally.get(item["status"], 0) + 1
        return tally

    def render(self) -> str:
        """Markdown-ish plan digest for a prompt, or a note that it is empty."""
        if not self.items:
            return "(no plan yet)"
        lines = []
        for item in self.items[:MAX_RENDERED_TODOS]:
            line = f"- [{item['status']}] {item['id']}: {item['content']}"
            if item["note"]:
                line += f"  (note: {item['note']})"
            lines.append(line)
        if len(self.items) > MAX_RENDERED_TODOS:
            lines.append(f"- ... (+{len(self.items) - MAX_RENDERED_TODOS} more, not shown)")
        return "\n".join(lines)

    # -- mutation ---------------------------------------------------------
    def update(self, todo_id: str, *, status: str | None = None, note: str | None = None) -> str:
        """Set ``status`` / ``note`` on one todo; returns a line for the model.

        Errors come back as text (``"Error: ..."``) like every other SuuClaw
        tool, so a model that guesses a wrong id can read the valid ids and try
        again instead of aborting the loop.
        """
        target = self.find(todo_id)
        if target is None:
            known = ", ".join(self.ids()) or "(the plan is empty)"
            return f"Error: no todo with id {todo_id!r}. Known ids: {known}"

        if status is not None:
            clean = str(status).strip().lower()
            if clean not in STATUSES:
                return f"Error: status must be one of {STATUSES}, got {status!r}"
            target["status"] = clean
        if note is not None:
            target["note"] = str(note).strip()

        rendered = f"todo {target['id']} is now {target['status']}"
        return f"{rendered} — {target['note']}" if target["note"] else rendered

    def mark_all(self, status: str) -> None:
        """Force every row to ``status`` (the verifier does this once work passes)."""
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        for item in self.items:
            item["status"] = status


# --------------------------------------------------------------------------- #
# tool schemas
# --------------------------------------------------------------------------- #
class PlannedTodo(BaseModel):
    """One step as the planner is asked to write it (no status — it is new)."""

    id: str = Field(
        description=(
            "Short, stable identifier for this step, e.g. '1' or 'inspect'. "
            "The actor uses it to report progress, so keep it unique."
        )
    )
    content: str = Field(description="One imperative step, concrete enough to finish and to check.")


class TodoWriteInput(BaseModel):
    """The planner's whole output: a plan that is checkable, not just plausible."""

    plan_summary: str = Field(
        description="One paragraph: what you will do and the shape of the result."
    )
    todos: list[PlannedTodo] = Field(
        description="Ordered steps. Prefer few and verifiable over many and vague."
    )
    acceptance_criteria: list[str] = Field(
        description=(
            "Plain-language statements that must be true when the task is done. "
            "These are what the verifier judges, so make them falsifiable."
        )
    )
    verification_commands: list[str] = Field(
        description=(
            "Shell commands that exit 0 exactly when the work is correct, run from "
            "the workspace root. Prefer `python -m pytest -q` style commands over "
            "interactive ones. Use an empty list only if nothing can be checked "
            "mechanically."
        )
    )


class TodoWriteTool(SuuClawTool):
    """Record the plan. Calling it replaces any previous plan."""

    name = "todo_write"
    description = (
        "Record the complete plan for the task: a summary, the ordered todos, the "
        "acceptance criteria, and the verification commands. Calling this replaces "
        "any previously recorded plan, so always send the full plan."
    )
    args_schema = TodoWriteInput

    def __init__(self, state: RuntimeState, store: TodoStore) -> None:
        super().__init__(state)
        self.store = store

    def run(  # type: ignore[override]
        self,
        plan_summary: str,
        todos: list[Any],
        acceptance_criteria: list[str],
        verification_commands: list[str],
    ) -> str:
        written = self.store.replace(todos)
        if not written:
            return "Error: the plan has no todos; every step needs a content field."
        return (
            f"Recorded a plan with {len(written)} todo(s) "
            f"[{', '.join(self.store.ids())}], {len(acceptance_criteria)} acceptance "
            f"criteria and {len(verification_commands)} verification command(s)."
        )


class TodoUpdateInput(BaseModel):
    """A progress report from the actor on one step of the plan."""

    todo_id: str = Field(description="id of the todo to update, exactly as it appears in the plan.")
    status: Literal["pending", "in_progress", "completed", "blocked"] = Field(
        description=(
            "New status. Use 'in_progress' when starting, 'completed' only once it "
            "is done and checkable, and 'blocked' when you cannot proceed."
        )
    )
    note: str = Field(
        default="",
        description=(
            "One short line of justification — required in spirit when the status is "
            "'blocked', since the planner reads it to revise the plan."
        ),
    )


class TodoUpdateTool(SuuClawTool):
    """Mark one todo as in progress / completed / blocked, with a note."""

    name = "todo_update"
    description = (
        "Update the status of one todo from the current plan. Call it when you start "
        "a step, when you finish one, and when you get stuck — the planner reads the "
        "notes to decide what to change when verification fails."
    )
    args_schema = TodoUpdateInput

    def __init__(self, state: RuntimeState, store: TodoStore) -> None:
        super().__init__(state)
        self.store = store

    def run(  # type: ignore[override]
        self,
        todo_id: str,
        status: str,
        note: str = "",
    ) -> str:
        return self.store.update(todo_id, status=status, note=note)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def todo_write_tool(state: RuntimeState, store: TodoStore) -> StructuredTool:
    """``TodoWriteTool`` bound to ``store``, ready for ``model.bind_tools``."""
    return TodoWriteTool(state, store).as_langchain_tool()


def todo_update_tool(state: RuntimeState, store: TodoStore) -> StructuredTool:
    """``TodoUpdateTool`` bound to ``store``, ready for ``model.bind_tools``."""
    return TodoUpdateTool(state, store).as_langchain_tool()


def build_todo_tools(state: RuntimeState, store: TodoStore) -> list[StructuredTool]:
    """Both todo tools bound to one store, in a stable order."""
    return [todo_write_tool(state, store), todo_update_tool(state, store)]
