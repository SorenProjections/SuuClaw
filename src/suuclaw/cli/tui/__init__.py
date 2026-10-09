"""Textual front end for SuuClaw.

A second front end, not a replacement. :mod:`suuclaw.cli.app` renders a
transcript of a *finished* run — one task in, one event stream out — while this
package keeps a live widget tree and a conversation that survives across turns:
the same session, the same workspace, the same state object, one turn at a time.

Three modules, three layers:

* :mod:`suuclaw.cli.tui.app` — the application: the layout, the thread that
  drives a turn's blocking event stream, and the mapping from workflow events
  onto panels. Entry point: :func:`suuclaw.cli.tui.app.run_tui`.
* :mod:`suuclaw.cli.tui.approval` — the approval bridge: the gate a tool thread
  blocks on, the modal that answers it, and the message that carries it across.
* :mod:`suuclaw.cli.tui.logo` — the startup splash: the ASCII wordmark, its
  animation frames, and the widget that plays them.

``textual`` is an optional dependency (``pip install "suuclaw[tui]"``). The core
workflow and the one-shot CLI run without it, so nothing in
:mod:`suuclaw.cli` may import this package at module scope — the ``--tui``
branch imports it inside the command body, and that is the only edge into here.
"""

from __future__ import annotations

__all__: list[str] = []
