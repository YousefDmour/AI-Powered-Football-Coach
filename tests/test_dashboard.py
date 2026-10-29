import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from live.dashboard import DashboardSession, handler_for
from tests.test_live import event


def test_dashboard_clock_pause_speed_reset_and_export_state():
    now = [0.0]
    session = DashboardSession([event(1, 0), event(2, 10), event(3, 30)], clock=lambda: now[0])
    first = session.snapshot()
    assert first["paused"]
    assert first["statistics"]["events_processed"] == 1
    now[0] = 100
    assert session.snapshot()["statistics"]["events_processed"] == 1
    session.control("speed", 1)
    session.control("play")
    now[0] = 109
    assert session.snapshot()["statistics"]["events_processed"] == 1
    now[0] = 110
    assert session.control("pause")["statistics"]["events_processed"] == 2
    now[0] = 200
    assert session.snapshot()["statistics"]["events_processed"] == 2
    session.control("speed", 60)
    session.control("play")
    now[0] = 201
    assert session.snapshot()["completed"]
    reset = session.control("reset")
    assert reset["run_id"] != first["run_id"]
    assert reset["paused"] and reset["alerts"] == []
    assert reset["statistics"]["events_processed"] == 1


@pytest.mark.parametrize("speed", [0, -1, 601, float("nan"), "60", True])
def test_bad_speed_does_not_change_session(speed):
    session = DashboardSession([event(1, 0)], clock=lambda: 0)
    before = session.snapshot()
    with pytest.raises(ValueError):
        session.control("speed", speed)
    assert session.snapshot() == before


def test_dashboard_alerts_persist_between_polls_and_clear_on_reset():
    now = [0]
    session = DashboardSession([event(i, i) for i in range(21)], clock=lambda: now[0])
    session.control("play")
    now[0] = 1
    result = session.snapshot()
    assert len(result["alerts"]) == 1
    assert result["alerts"][0]["attempted"] == 20
    assert session.snapshot()["alerts"] == result["alerts"]
    assert session.control("reset")["alerts"] == []


def test_http_assets_controls_export_and_errors():
    session = DashboardSession([event(1, 0), event(2, 600)], clock=lambda: 0)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        for path, expected in [("/", b'lang="ar"'), ("/app.js", b"renderAlerts"), ("/style.css", b"@media")]:
            connection.request("GET", path)
            response = connection.getresponse()
            assert response.status == 200
            assert expected in response.read()
        connection.request("POST", "/api/control", json.dumps({"action": "play"}))
        response = connection.getresponse()
        assert response.status == 200
        assert not json.loads(response.read())["paused"]
        connection.request("GET", "/api/export")
        response = connection.getresponse()
        assert "attachment" in response.getheader("Content-Disposition")
        assert json.loads(response.read())["statistics"]["events_processed"] == 1
        connection.request("POST", "/api/control", '{"action":"speed","speed":0}')
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.request("POST", "/api/control", '{"action":"pause"}', {"Origin": "https://example.com"})
        response = connection.getresponse()
        assert response.status == 403
        response.read()
        connection.request("GET", "/../project_paths.py")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
