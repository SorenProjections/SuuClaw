"""``bash`` tool — run a shell command with the workspace as the working directory.

The command runs with ``cwd`` set to the workspace, a hard timeout, and no stdin
(so nothing can block waiting for input). On timeout the whole process *tree* is
killed, not just the shell, otherwise a spawned child could outlive the call.

Caveat worth knowing: the workspace boundary is enforced at the path layer
(``file_*`` / ``grep``). A shell command is arbitrary code by definition — a
determined ``cd /`` or ``rm`` can still leave the workspace. Treat this tool as
"trusted local execution", not as a security sandbox.

What the tool *does* add on top of that is an approval gate: a command whose
shape matches :data:`~suuclaw.core.approval.RISK_PATTERNS` (installs,
downloads, servers — the things that reach outside the workspace or outlive the
call) is routed through the state's ``approval_mode`` before it runs. See
:mod:`suuclaw.core.approval` for the policy; the gate lives in
:meth:`BashTool.run`, which is the entry point ``StructuredTool`` invokes.
"""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ..core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    classify_command_risk,
    normalize_approval_mode,
)
from .registry import SuuClawTool

__all__ = ["BashInput", "BashTool", "CommandResult", "resolve_shell", "run_command"]

DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_TIMEOUT_SECONDS = 600.0

#: A ``bash`` that is really the WSL launcher is *not* interchangeable with Git
#: Bash. It runs a Linux userland — paths become ``/mnt/<drive>/...``, ``python3``
#: is the distro's interpreter, and ``curses`` imports fine — so anything the
#: agent verifies there can be false for the user's own terminal. Prefer a native
#: Windows POSIX shell whenever one exists; WSL stays available via --shell or
#: SUUCLAW_SHELL for deliberate Linux work.
_NATIVE_BASH_GLOBS = (
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
    r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe",
    r"%ProgramFiles%\Git\usr\bin\bash.exe",
    r"%USERPROFILE%\scoop\apps\git\current\bin\bash.exe",
    r"%ProgramData%\chocolatey\bin\bash.exe",
    r"%USERPROFILE%\.workbuddy\binaries\PortableGit\versions\*\bin\bash.exe",
)


def _is_wsl_launcher(path: str) -> bool:
    """True for ``C:\\Windows\\System32\\bash.exe`` — the WSL entry point."""
    return any(part.lower() in {"system32", "syswow64"} for part in Path(path).parts)


def _find_native_bash() -> str | None:
    """First existing *native Windows* bash, skipping the WSL launcher."""
    if os.name != "nt":
        return None

    candidates: list[str] = []
    for pattern in _NATIVE_BASH_GLOBS:
        expanded = os.path.expandvars(pattern)
        if any(ch in expanded for ch in "*?"):
            candidates.extend(sorted(glob.glob(expanded), reverse=True))
        else:
            candidates.append(expanded)

    on_path = shutil.which("bash")
    if on_path:
        candidates.append(on_path)

    for candidate in candidates:
        if Path(candidate).is_file() and not _is_wsl_launcher(candidate):
            return candidate
    return None


def resolve_shell(explicit: str | None = None) -> list[str]:
    """Return the argv prefix used to run a command.

    Resolution order: explicit state config → ``$SUUCLAW_SHELL`` → a native
    Windows bash (Git Bash / PortableGit, **never** the WSL launcher) → any
    remaining bash (WSL on Windows) → platform default (``%COMSPEC% /c``,
    ``$SHELL -c``).

    Point ``SUUCLAW_SHELL`` at WSL when the task genuinely wants Linux.
    """
    if explicit:
        parts = shlex.split(explicit, posix=os.name != "nt")
        if not parts:
            raise ValueError("shell must not be blank")
        if len(parts) == 1:
            parts.append("-c")
        return parts

    override = os.environ.get("SUUCLAW_SHELL")
    if override and override.strip():
        parts = shlex.split(override.strip(), posix=os.name != "nt")
        if len(parts) == 1:
            parts.append("-c")
        return parts

    if os.name == "nt":
        native = _find_native_bash()
        if native:
            return [native, "-c"]
        # Nothing but WSL (or nothing at all) is installed. WSL still beats
        # cmd.exe for POSIX commands, so keep it, but the paths it sees differ.
        wsl = shutil.which("bash")
        if wsl:
            return [wsl, "-c"]
        return [os.environ.get("COMSPEC", "cmd.exe"), "/c"]

    bash = shutil.which("bash")
    if bash:
        return [bash, "-c"]
    return [os.environ.get("SHELL") or "/bin/sh", "-c"]


class BashInput(BaseModel):
    """Arguments for :class:`BashTool`."""

    command: str = Field(description="Shell command line to execute in the workspace directory.")
    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=MAX_TIMEOUT_SECONDS,
        description=f"Kill the command after this many seconds (max {MAX_TIMEOUT_SECONDS:g}).",
    )


@dataclass(slots=True)
class CommandResult:
    """Structured outcome of one shell command.

    :class:`BashTool` renders this into prose for a model to read; the graph's
    verifier stores it as a ``VerificationResult``. Both go through
    :func:`run_command`, so a timeout kills the process tree identically either
    way — the reason this exists at all instead of two copies of ``Popen``.

    ``requires_approval`` and ``approval_reason`` record the approval gate's
    verdict rather than its decision: they are set only when the command matched
    a risk pattern, and they stay set after the command has run, so a caller can
    tell "ran, and it was an install" from "ran, and it was a read".
    """

    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float = 0.0
    timed_out: bool = False
    error: str = ""
    requires_approval: bool = False
    approval_reason: str = ""


def run_command(
    command: str,
    *,
    workspace: str | os.PathLike[str],
    shell: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> CommandResult:
    """Run ``command`` in ``workspace`` and report the outcome as data.

    Never raises for a failing command — a non-zero exit, a timeout and a
    failure to even spawn the shell all come back as a :class:`CommandResult`.
    """
    if not command or not command.strip():
        return CommandResult(
            command=command or "",
            ok=False,
            exit_code=None,
            stdout="",
            stderr="",
            error="command must not be empty",
        )
    if timeout <= 0:
        return CommandResult(
            command=command,
            ok=False,
            exit_code=None,
            stdout="",
            stderr="",
            error=f"timeout must be > 0, got {timeout:g}",
        )

    argv = [*resolve_shell(shell), command]
    child_env = os.environ.copy()
    child_env.update({str(k): str(v) for k, v in (env or {}).items()})
    child_env.setdefault("PYTHONIOENCODING", "utf-8")

    popen_kwargs: dict[str, object] = {
        "cwd": str(workspace),
        "env": child_env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":  # pragma: no cover - platform specific
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    started = time.monotonic()
    try:
        process = subprocess.Popen(argv, **popen_kwargs)  # type: ignore[arg-type]
    except OSError as exc:
        return CommandResult(
            command=command,
            ok=False,
            exit_code=None,
            stdout="",
            stderr="",
            duration=time.monotonic() - started,
            error=f"cannot start shell ({argv[0]!r}): {exc}",
        )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except Exception:  # pragma: no cover - best effort drain
            stdout, stderr = "", ""
        elapsed = time.monotonic() - started
        return CommandResult(
            command=command,
            ok=False,
            exit_code=None,
            stdout=stdout or "",
            stderr=stderr or "",
            duration=elapsed,
            timed_out=True,
            error=(f"command exceeded timeout of {timeout:g}s and was killed after {elapsed:.2f}s"),
        )

    elapsed = time.monotonic() - started
    return CommandResult(
        command=command,
        ok=process.returncode == 0,
        exit_code=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        duration=elapsed,
    )


class BashTool(SuuClawTool):
    """Execute a shell command inside the workspace, with timeout control."""

    name = "bash"
    description = (
        "Run a shell command with the workspace root as the working directory and "
        "return its exit code, stdout, and stderr. The command is killed (process "
        "tree included) if it exceeds timeout_seconds. Prefer non-interactive, "
        "non-destructive commands; there is no stdin, so anything that prompts will "
        "fail rather than hang. Commands that install packages, download from the "
        "network, or start a long-running server are subject to an approval policy "
        "and may be refused, with the reason stated in the result."
    )
    args_schema = BashInput

    def run(self, command: str, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
        if not command or not command.strip():
            return "Error: command must not be empty."
        if timeout_seconds <= 0:
            return "Error: timeout_seconds must be > 0."

        # The gate runs *before* the shell does. Classifying afterwards would be
        # able to report the risk but never to prevent it, which is the whole
        # point of the feature.
        risk_reason = classify_command_risk(command)
        decision = self._approval_decision(command, risk_reason) if risk_reason else None
        if decision is not None and not decision.approved:
            return self._refusal(command, risk_reason or "", decision)

        result = run_command(
            command,
            workspace=self.state.workspace,
            shell=self.state.shell,
            env=self.state.env,
            timeout=timeout_seconds,
        )

        if decision is not None:
            # The flag survives the run on purpose: that an install happened is
            # the one part of the outcome an exit code cannot tell you.
            result.requires_approval = True
            result.approval_reason = f"{risk_reason} — {decision.reason}"

        approval_line = ""
        if result.requires_approval and result.approval_reason:
            approval_line = f"approval: required ({result.approval_reason})"

        if result.timed_out:
            body = "\n".join(
                [
                    f"Error: command exceeded timeout of {timeout_seconds:g}s and was killed "
                    f"after {result.duration:.2f}s.",
                    f"command: {command}",
                    "--- stdout (partial) ---",
                    result.stdout.rstrip() or "(empty)",
                    "--- stderr (partial) ---",
                    result.stderr.rstrip() or "(empty)",
                ]
            )
            truncate_note = "timeout output truncated"
        elif result.error and result.exit_code is None:
            return f"Error: {result.error}"
        else:
            stdout, stderr = result.stdout, result.stderr
            sections = [
                f"exit code: {result.exit_code}",
                f"duration: {result.duration:.2f}s",
                f"command: {command}",
            ]
            if stdout and stdout.strip():
                sections += ["--- stdout ---", stdout.rstrip()]
            if stderr and stderr.strip():
                sections += ["--- stderr ---", stderr.rstrip()]
            if not (stdout and stdout.strip()) and not (stderr and stderr.strip()):
                sections.append("(no output)")
            body = "\n".join(sections)
            truncate_note = "command output truncated"

        # Leading, not trailing: an approval notice printed under 200 lines of
        # stdout is a notice nobody reads, and the model reads this too.
        if approval_line:
            body = f"{approval_line}\n{body}"
        return self.state.truncate(body, note=truncate_note)

    # -- approval gate ----------------------------------------------------
    def _approval_mode(self) -> str:
        """The gate's policy, normalised at the point of use.

        :class:`~suuclaw.core.state.RuntimeState` already normalises on
        construction, but the mode is a plain attribute a caller may assign to
        afterwards; normalising here costs nothing and keeps an unknown value
        from slipping past every branch below into "treated as inline".
        """
        return normalize_approval_mode(self.state.approval_mode)

    def _approval_decision(self, command: str, risk_reason: str) -> ApprovalDecision:
        """Resolve a risky command to a verdict, according to the approval mode.

        ``deny`` and ``auto`` are decided here; ``inline`` is delegated to the
        state's handler. A *missing* handler is not a refusal (see the module
        docstring of :mod:`suuclaw.core.approval`), but a handler that raises
        is: when the approval path itself is broken, the risky command must not
        be the thing that still goes through.
        """
        mode = self._approval_mode()
        if mode == "deny":
            return ApprovalDecision(
                approved=False,
                reason=f"approval mode 'deny' refuses {risk_reason.lower()}",
            )
        if mode == "auto":
            return ApprovalDecision(
                approved=True,
                reason=f"approved by approval mode 'auto' ({risk_reason.lower()})",
            )

        handler = self.state.approval_handler
        if handler is None:
            return ApprovalDecision(
                approved=True,
                reason="no approval handler is configured; ran without confirmation",
            )
        try:
            return handler(ApprovalRequest(command=command, risk_reason=risk_reason))
        except Exception as exc:  # deliberate: fail closed on a broken handler
            return ApprovalDecision(
                approved=False,
                reason=f"the approval handler raised {type(exc).__name__}: {exc}",
            )

    def _refusal(self, command: str, risk_reason: str, decision: ApprovalDecision) -> str:
        """What the model is told when the gate stops a command.

        Refusing silently would be worse than not refusing at all: the model has
        to learn *which* command was blocked and why, or its next move is to
        send the same one again with the timeout changed.
        """
        return self.state.truncate(
            "\n".join(
                [
                    "Error: the approval gate refused this command and it did NOT run.",
                    f"risk: {risk_reason}",
                    f"approval mode: {self._approval_mode()}",
                    f"reason: {decision.reason or 'no reason given'}",
                    f"command: {command}",
                    "Either ask the user to approve it, or reach the goal without it.",
                ]
            ),
            note="refusal message truncated",
        )


def _kill_tree(process: subprocess.Popen[str]) -> None:
    """Kill ``process`` and everything it spawned."""
    if os.name == "nt":  # pragma: no cover - platform specific
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
    try:
        process.wait(timeout=5)
    except Exception:  # pragma: no cover - already gone
        pass
