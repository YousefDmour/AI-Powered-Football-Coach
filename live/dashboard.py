"""Local dashboard serving the existing Python replay engine, without dependencies."""

import argparse
import json
import math
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from live.replay import Replay, load_events
from project_paths import EVENTS_FILE

ASSETS = Path(__file__).resolve().parent / "web"


class DashboardSession:
    """One shared local replay. Each request settles the previous clock interval."""

    def __init__(self, events, speed=60, clock=time.monotonic):
        self.events = events
        self.clock = clock
        self.lock = threading.RLock()
        self.initial_speed = speed
        self.reset()

    def reset(self):
        self.replay = Replay(self.events, speed=self.initial_speed)
        self.replay.advance(0)
        self.replay.paused = True
        self.alerts = []
        self.run_id = uuid.uuid4().hex
        self.updated_at = self.clock()

    def _advance(self):
        now = self.clock()
        alerts = self.replay.advance(max(0, now - self.updated_at))
        self.updated_at = now
        for alert in alerts:
            self.alerts.append({"id": len(self.alerts) + 1, **alert})

    def snapshot(self):
        with self.lock:
            self._advance()
            return {
                "source": "simulated_event_replay", "run_id": self.run_id,
                "paused": self.replay.paused, "completed": self.replay.finished,
                "speed": self.replay.speed, "total_events": len(self.events),
                "statistics": self.replay.state.snapshot(), "alerts": list(self.alerts),
            }

    def control(self, action, speed=None):
        # Validate before advancing or mutating the session.
        if action not in {"play", "pause", "reset", "speed"}:
            raise ValueError("Unknown control action")
        if action == "speed":
            if isinstance(speed, bool) or not isinstance(speed, (int, float)):
                raise ValueError("Speed must be a number")
            if not math.isfinite(speed) or not 1 <= speed <= 600:
                raise ValueError("Speed must be between 1 and 600")
        with self.lock:
            self._advance()
            if action == "reset":
                self.reset()
            elif action == "speed":
                self.replay.speed = speed
            else:
                self.replay.paused = action == "pause"
            return self.snapshot()


def handler_for(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, body, content_type, status=200, download=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="match-analysis.json"')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=200, download=False):
            self.respond(json.dumps(value, ensure_ascii=False).encode("utf-8"),
                         "application/json; charset=utf-8", status, download)

        def do_GET(self):
            if self.path in {"/api/state", "/api/export"}:
                self.send_json(session.snapshot(), download=self.path == "/api/export")
                return
            assets = {"/": ("index.html", "text/html"),
                      "/app.js": ("app.js", "text/javascript"),
                      "/style.css": ("style.css", "text/css")}
            if self.path not in assets:
                self.send_json({"error": "Not found"}, 404)
                return
            filename, content_type = assets[self.path]
            self.respond((ASSETS / filename).read_bytes(), content_type + "; charset=utf-8")

        def do_POST(self):
            if self.path != "/api/control":
                self.send_json({"error": "Not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                body = self.rfile.read(length)
                # Consume the bounded body before rejecting its origin. Closing
                # with unread request bytes can reset the socket on Windows.
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    self.send_json({"error": "Cross-origin control is not allowed"}, 403)
                    return
                payload = json.loads(body)
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                result = session.control(payload.get("action"), payload.get("speed"))
                self.send_json(result)
            except (ValueError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local simulated live football dashboard")
    parser.add_argument("--events", type=Path, default=EVENTS_FILE)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    session = DashboardSession(load_events(args.events))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(session))
    print(f"Dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    print("Recorded event replay. Starts paused. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
