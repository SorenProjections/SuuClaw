"""Lazy public entry points for SuuClaw's three local interfaces.

Only an exact leading ``dashboard`` is reserved. Ordinary task strings and
Typer options retain their existing behavior, including through the legacy alias.
Use ``suuclaw -- dashboard`` to submit the literal task ``dashboard``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

__all__ = ["dashboard_main", "feishu_main", "main"]


def dashboard_main(argv: Sequence[str] | None = None) -> int:
    """Start the browser dashboard without loading the optional TUI or Feishu SDK."""
    from suuclaw.dashboard.server import main as run_dashboard

    return run_dashboard(None if argv is None else list(argv))


def feishu_main(argv: Sequence[str] | None = None) -> int:
    """Start the optional Feishu gateway through its existing implementation."""
    from suuclaw.feishu.app import main as run_feishu

    return run_feishu(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the dashboard subcommand or preserve the one-task Typer CLI."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "dashboard":
        return dashboard_main(args[1:])

    from suuclaw.cli.app import app

    app(args=args, prog_name="suuclaw")
    return 0
