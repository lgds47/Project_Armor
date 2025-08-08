"""Lightweight HTTP health server exposing /health and /ready.
Uses only Python's standard library to keep image slim and avoid extra deps.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import threading
from typing import Optional


class _HealthHandler(BaseHTTPRequestHandler):
    # Simple readiness flag; can be toggled by env var or externally via file
    READY_FLAG_ENV = "READINESS_READY"

    def _send(self, code: int, body: str = "OK"):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/health":
            self._send(200, "healthy")
        elif self.path == "/ready":
            ready = os.environ.get(self.READY_FLAG_ENV, "true").lower() in ("1", "true", "yes")
            if ready:
                self._send(200, "ready")
            else:
                self._send(503, "not ready")
        else:
            self._send(404, "not found")

    def log_message(self, format: str, *args):  # silence default logging
        return


def serve(port: int = 8080, host: str = "0.0.0.0") -> None:
    server = HTTPServer((host, port), _HealthHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def serve_in_thread(port: int = 8080, host: str = "0.0.0.0") -> threading.Thread:
    t = threading.Thread(target=serve, kwargs={"port": port, "host": host}, daemon=True)
    t.start()
    return t
