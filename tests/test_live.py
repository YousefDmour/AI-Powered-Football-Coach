import json
import pytest

from data_layer.player_stats import load_player_stats
from live.engine import MatchState
from live.recommendations import LiveRecommendations
from live.replay import Replay, load_events
from project_paths import EVENTS_FILE


def event(number, seconds, kind="Pass", complete=False, period=1):
    result = {
        "id": str(number), "period": period,
        "timestamp": f"00:{seconds // 60:02}:{seconds % 60:02}.000",
        "minute": seconds // 60, "second": seconds % 60,
        "type": {"name": kind}, "team": {"name": "Test FC"},
        "player": {"id": 1, "name": "Player One"},
    }
    if not complete:
        result["pass"] = {"outcome": {"name": "Incomplete"}}
    return result


def test_replay_clock_pause_speed_and_no_future_events():
    source = [event(1, 0, complete=True), event(2, 10), event(3, 20)]
    replay = Replay(source, speed=2)
    replay.advance(0)
    assert replay.state.snapshot()["teams"]["Test FC"]["passes_attempted"] == 1
    replay.paused = True
    replay.advance(100)
    assert replay.position == 1
    replay.paused = False
    replay.advance(4)
    assert replay.position == 1
    replay.speed = 4
    replay.advance(0.5)
    assert replay.position == 2
    assert replay.state.snapshot()["teams"]["Test FC"]["passes_completed"] == 1
    replay.advance(2.5)
    assert replay.finished


def test_duplicate_out_of_order_and_snapshot_isolation():
    state = MatchState()
    first = event(1, 10)
    assert state.apply(first)
    assert not state.apply(first)
    with pytest.raises(ValueError):
        state.apply(event(2, 5))
    snapshot = state.snapshot()
    snapshot["teams"]["Test FC"]["shots"] = 100
    assert state.snapshot()["teams"]["Test FC"]["shots"] == 0


def test_windows_expire_across_periods_without_resetting_totals():
    state = MatchState(window_seconds=600)
    state.apply(event(1, 0))
    state.apply(event(2, 601, "Half End"))
    assert state.snapshot()["recent"]["Test FC"]["passes_attempted"] == 0
    state.apply(event(3, 0, complete=True, period=2))
    assert state.snapshot()["teams"]["Test FC"]["passes_attempted"] == 2
    assert state.snapshot()["recent"]["Test FC"]["passes_attempted"] == 1
    assert state.elapsed == 601


def test_alerts_require_evidence_and_respect_match_time_cooldown():
    state = MatchState()
    advisor = LiveRecommendations()
    for index in range(19):
        state.apply(event(index, index))
        assert advisor.evaluate(state.snapshot()) == []
    state.apply(event(19, 19))
    alerts = advisor.evaluate(state.snapshot())
    assert len(alerts) == 1
    assert "0/20" in alerts[0]["evidence"]
    assert alerts[0]["rule"] == "passing"
    assert advisor.evaluate(state.snapshot()) == []
    state.apply(event(20, 319, "Pressure"))
    assert len(advisor.evaluate(state.snapshot())) == 1
    state.apply(event(21, 1000, "Pressure"))
    assert advisor.evaluate(state.snapshot()) == []


def assert_matches_offline(state, frame):
    for team, rows in frame.groupby("team"):
        stats = state.snapshot()["teams"][team]
        for key in ("passes_attempted", "passes_completed", "shots", "shots_on_target", "goals"):
            assert stats[key] == int(rows[key].sum()), (team, key)
        assert stats["defensive_actions"] == int(rows["interceptions"].sum() + rows["tackles"].sum())
        assert stats["defensive_actions_successful"] == int(
            rows["interceptions_won"].sum() + rows["tackles_won"].sum())
    for player_id, stats in state.snapshot()["players"].items():
        row = frame.loc[frame["player_id"] == player_id].iloc[0]
        for key in ("passes_attempted", "passes_completed", "shots", "shots_on_target", "goals"):
            assert stats[key] == int(row[key]), (player_id, key)


def test_full_replay_matches_offline_team_and_player_totals():
    replay = Replay(load_events(EVENTS_FILE), speed=0)
    replay.advance(0)
    assert replay.finished
    assert_matches_offline(replay.state, load_player_stats(EVENTS_FILE))


def test_prefix_matches_only_received_events(tmp_path):
    source = load_events(EVENTS_FILE)
    prefix = source[:300]
    state = MatchState()
    for received in prefix:
        state.apply(received)
    path = tmp_path / "prefix.json"
    path.write_text(json.dumps(prefix), encoding="utf-8")
    assert_matches_offline(state, load_player_stats(path))
    assert state.events_processed == 300


def test_zero_shots_never_produces_shooting_alert():
    state = MatchState()
    state.apply(event(1, 0, "Starting XI"))
    assert state.snapshot()["teams"]["Test FC"]["shots_on_target_ratio"] is None
    assert LiveRecommendations().evaluate(state.snapshot()) == []


def test_replay_rejects_reversed_events_and_invalid_speed():
    with pytest.raises(ValueError):
        Replay([event(1, 10), event(2, 0)])
    with pytest.raises(ValueError):
        Replay([], speed=float("nan"))
