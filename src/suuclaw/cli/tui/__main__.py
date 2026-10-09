"""``python -m suuclaw.cli.tui`` — open the TUI on ``./workspace``.

The same thing ``suuclaw --tui`` does, for a checkout that has not been
reinstalled as a console script. Options are deliberately absent: anything worth
configuring — the workspace, the model, the approval policy — is a flag on the
main command, and a second, thinner argument parser here would be one more place
for the two entry points to disagree.
"""

from __future__ import annotations

from pathlib import Path

from ...core import RuntimeState
from .app import run_tui


def main() -> int:
    state = RuntimeState.create(Path("workspace"))
    return run_tui(state=state, session_workspace=state.workspace)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
