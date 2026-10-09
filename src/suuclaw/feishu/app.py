"""Command-line entry point for the Feishu long-connection gateway."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ..core.paths import runtime_path
from ..providers.openai_provider import load_environment
from .config import FeishuGatewayConfig, GatewayConfigError
from .gateway import create_gateway


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="suuclaw-feishu",
        description="Run SuuClaw as a Feishu Bot over the official WebSocket channel.",
    )
    parser.add_argument("--workspace", type=Path, help="Agent sandbox (overrides env).")
    parser.add_argument("--model", help="Model id (overrides SUUCLAW_FEISHU_MODEL).")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate credentials/configuration and construct clients without connecting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_environment()
    try:
        config = FeishuGatewayConfig.from_env()
        if args.workspace is not None:
            workspace = args.workspace.expanduser().resolve()
            config = replace(
                config,
                workspace=workspace,
                session_root=runtime_path(workspace, "feishu", "conversations"),
            )
        if args.model:
            config = replace(config, model=args.model.strip())
        gateway = create_gateway(config)
    except (GatewayConfigError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        print("Feishu gateway configuration is valid.")
        print(f"workspace: {config.workspace}")
        print(f"conversation root: {config.session_root}")
        print(f"approval mode: {config.approval_mode}")
        print(f"max concurrency: {config.max_concurrency}")
        print(f"workflow progress: {'on' if config.workflow_progress else 'off'}")
        print(f"workflow heartbeat: {config.workflow_heartbeat_seconds:g}s")
        print(f"entry timeout: {config.entry_timeout:g}s")
        print(f"artifact delivery: {'on' if config.deliver_artifacts else 'off'}")
        print(
            "artifact limits: "
            f"{config.artifact_max_files} file(s), "
            f"{config.artifact_max_megabytes:g} MiB each"
        )
        print(f"WebSocket route: {config.ws_proxy}")
        print(f"WebSocket open timeout: {config.ws_open_timeout:g}s")
        return 0

    try:
        asyncio.run(gateway.serve())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Feishu gateway stopped by user")
        return 130
    except Exception as exc:  # noqa: BLE001 - process boundary
        logging.getLogger(__name__).exception("Feishu gateway stopped: %s", exc)
        return 1
    return 0
