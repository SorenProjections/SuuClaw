"""A loopback-only HTTP/SSE server with no additional runtime dependencies."""

from __future__ import annotations

import argparse
import ipaddress
import json
import mimetypes
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .service import DashboardService
from .store import TERMINAL_STATUSES, DashboardError

DEFAULT_PORT = 7788
MAX_BODY_BYTES = 64 * 1024
STATIC_ROOT = Path(__file__).parent / "static"


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: DashboardService) -> None:
        self.service = service
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, DashboardHandler)

    def server_close(self) -> None:
        self.service.shutdown()
        super().server_close()


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer
    server_version = "SuuClaw"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, format: str, *args: Any) -> None:
        # Requests include session IDs and artifact paths. Do not print them
        # by default; diagnostic failures are returned safely in JSON instead.
        pass

    def _headers(self, status: int, content_type: str, length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'",
        )
        if length is not None:
            self.send_header("Content-Length", str(length))

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, error: DashboardError) -> None:
        self._json(error.status, {"error": {"code": error.code, "message": str(error)}})

    def _validate_request(self) -> None:
        host = self.headers.get("Host", "")
        allowed_hosts = {"localhost", "127.0.0.1", "::1", str(self.server.server_address[0])}

        def allowed_authority(value: str) -> bool:
            try:
                parsed = urlsplit("//" + value)
                return bool(
                    parsed.hostname in allowed_hosts
                    and (parsed.port or 80) == self.server.server_port
                    and not parsed.username
                    and not parsed.password
                    and not parsed.path
                    and not parsed.query
                    and not parsed.fragment
                )
            except ValueError:
                return False

        if len(self.headers.get_all("Host", [])) != 1 or not allowed_authority(host):
            raise DashboardError("invalid_host", "Use the dashboard's local address.", 403)
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if (
                parsed.scheme != "http"
                or not allowed_authority(parsed.netloc)
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise DashboardError(
                    "invalid_origin", "Cross-origin requests are not allowed.", 403
                )
        if self.headers.get("Sec-Fetch-Site", "") == "cross-site":
            raise DashboardError("invalid_origin", "Cross-site requests are not allowed.", 403)

    def _body(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise DashboardError("invalid_body", "Chunked request bodies are not accepted.")
        if self.headers.get_content_type() != "application/json":
            raise DashboardError("invalid_content_type", "Send an application/json request.", 415)
        if len(self.headers.get_all("Content-Length", [])) != 1:
            raise DashboardError("invalid_body", "One Content-Length header is required.", 411)
        try:
            length = int(self.headers["Content-Length"])
        except (ValueError, TypeError) as exc:
            raise DashboardError("invalid_body", "Invalid Content-Length.") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise DashboardError("body_too_large", "JSON bodies are limited to 64 KiB.", 413)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise DashboardError("invalid_body", "Incomplete request body.")
        try:
            body = json.loads(raw)
        except (UnicodeError, ValueError) as exc:
            raise DashboardError("invalid_json", "Request body is not valid JSON.") from exc
        if not isinstance(body, dict):
            raise DashboardError("invalid_json", "JSON body must be an object.")
        return body

    def do_GET(self) -> None:
        try:
            self._validate_request()
            self._get()
        except DashboardError as exc:
            self.close_connection = True
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:  # noqa: BLE001 - internal paths/secrets must not reach HTTP errors
            self.close_connection = True
            self._error(
                DashboardError("server_error", "The local request could not be completed.", 500)
            )

    def do_POST(self) -> None:
        try:
            self._validate_request()
            self._post()
        except DashboardError as exc:
            self.close_connection = True
            self._error(exc)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception:  # noqa: BLE001 - report safe error without process internals
            self.close_connection = True
            self._error(
                DashboardError("server_error", "The local request could not be completed.", 500)
            )

    def _get(self) -> None:
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        parts = path.strip("/").split("/")
        service = self.server.service
        store = service.store
        if path == "/api/bootstrap":
            self._json(200, service.bootstrap())
        elif path == "/api/sessions":
            self._json(200, {"sessions": store.list_sessions()})
        elif len(parts) == 3 and parts[:2] == ["api", "sessions"]:
            self._json(200, store.get_session(parts[2]))
        elif len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "files":
            self._json(200, store.list_files(parts[2]))
        elif len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "file":
            raw_path = query.get("path", [""])[0]
            download = query.get("download", [""])[0] == "1"
            result = service.read_file(parts[2], raw_path, download=download)
            if download:
                name, data = result
                self._headers(200, "application/octet-stream", len(data))
                self.send_header(
                    "Content-Disposition", f"attachment; filename*=UTF-8''{quote(name)}"
                )
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(200, result)
        elif len(parts) == 3 and parts[:2] == ["api", "runs"]:
            run = store.get_run(parts[2])
            self._json(200, {**run, "events": store.read_events(parts[2])})
        elif len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "events":
            # Native EventSource sends Last-Event-ID on reconnection. It takes
            # precedence over the initial query cursor, avoiding duplicate replay.
            raw_after = self.headers.get("Last-Event-ID") or query.get("after", ["0"])[0]
            try:
                after = int(raw_after)
                if after < 0:
                    raise ValueError
            except ValueError as exc:
                raise DashboardError(
                    "invalid_cursor", "Event cursor must be a non-negative integer."
                ) from exc
            self._events(parts[2], after)
        elif path == "/" or path.startswith("/static/"):
            self._static(path)
        else:
            raise DashboardError("not_found", "Endpoint does not exist.", 404)

    def _post(self) -> None:
        parsed = urlsplit(self.path)
        parts = unquote(parsed.path).strip("/").split("/")
        body = self._body()
        service = self.server.service
        if parts == ["api", "sessions"]:
            title = body.get("title", "新会话")
            if not isinstance(title, str):
                raise DashboardError("invalid_title", "Session title must be text.")
            self._json(201, service.store.create_session(service.redactor.text(title)))
        elif len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "runs":
            self._json(202, service.start_run(parts[2], body.get("message")))
        elif len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "approvals":
            self._json(200, service.resolve_approval(parts[2], parts[4], body.get("approved")))
        else:
            raise DashboardError("not_found", "Endpoint does not exist.", 404)

    def _static(self, path: str) -> None:
        relative = "index.html" if path == "/" else path.removeprefix("/static/")
        parts = relative.replace("\\", "/").split("/")
        if (
            not relative
            or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
            or ":" in relative
            or "\x00" in relative
        ):
            raise DashboardError("not_found", "Asset does not exist.", 404)
        root = STATIC_ROOT.resolve()
        candidate = root.joinpath(*parts)
        if (
            candidate.resolve() != candidate
            or not candidate.is_file()
            or not candidate.resolve().is_relative_to(root)
        ):
            raise DashboardError("not_found", "Asset does not exist.", 404)
        body = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix == ".js":
            content_type = "text/javascript"
        if content_type.startswith("text/") or candidate.suffix == ".svg":
            content_type += "; charset=utf-8"
        self._headers(200, content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def _events(self, run_id: str, after: int) -> None:
        service = self.server.service
        service.store.get_run(run_id)
        self._headers(200, "text/event-stream; charset=utf-8")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        last_ping = time.monotonic()
        while True:
            records = service.store.read_events(run_id, after)
            for record in records:
                data = json.dumps(record, ensure_ascii=False)
                self.wfile.write(f"id: {record['seq']}\ndata: {data}\n\n".encode())
                after = record["seq"]
            if records:
                self.wfile.flush()
            run = service.store.get_run(run_id)
            if run["status"] in TERMINAL_STATUSES and after >= run["last_seq"]:
                return
            if service.stopped.is_set():
                return
            if time.monotonic() - last_ping >= 10:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                last_ping = time.monotonic()
            with service.changed:
                service.changed.wait(timeout=0.5)


def create_server(
    workspace: str | os.PathLike[str],
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    service: DashboardService | None = None,
    model: str | None = None,
) -> DashboardHTTPServer:
    """Create an unstarted server; caller owns serve_forever/shutdown/server_close."""
    try:
        is_loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError("The dashboard only binds to a loopback address (127.0.0.1 or ::1).")
    return DashboardHTTPServer((host, port), service or DashboardService(workspace, model=model))


def serve(
    workspace: str | os.PathLike[str],
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    model: str | None = None,
) -> None:
    server = create_server(workspace, host=host, port=port, model=model)
    display_host = f"[{host}]" if ":" in host else host
    print(f"SuuClaw · http://{display_host}:{server.server_port}", flush=True)
    print(f"Workspace: {server.service.store.workspace}", flush=True)
    print("Press Ctrl+C to stop. New browser sessions use isolated workspaces.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SuuClaw local workflow dashboard")
    parser.add_argument(
        "--workspace", "-w", default=".", help="Root for dashboard session storage."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Loopback bind address.")
    parser.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="Local HTTP port.")
    parser.add_argument("--model", default=None, help="Model ID; defaults to OPENAI_MODEL.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        serve(args.workspace, host=args.host, port=args.port, model=args.model)
    except (OSError, ValueError, DashboardError) as exc:
        parser.exit(1, f"SuuClaw dashboard: {exc}\n")


if __name__ == "__main__":
    main()
