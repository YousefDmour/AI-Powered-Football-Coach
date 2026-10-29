"""Consume only received events; no access to the replay source is required."""

from collections import Counter, deque
from copy import deepcopy

from data_layer.player_stats import _successful_pass, _successful_duel


def event_clock(event):
    """Absolute displayed match seconds (including the provider's added time)."""
    return int(event.get("minute", 0)) * 60 + int(event.get("second", 0))


def event_delta(event):
    """Statistics supported by both the offline report and live analysis."""
    delta = Counter()
    kind = event.get("type", {}).get("name")
    if kind == "Pass":
        delta["passes_attempted"] = 1
        delta["passes_completed"] = int(_successful_pass(event))
    elif kind == "Shot":
        outcome = event.get("shot", {}).get("outcome", {}).get("name")
        delta["shots"] = 1
        delta["shots_on_target"] = int(outcome in {"Goal", "Saved", "Saved To Post"})
        delta["goals"] = int(outcome == "Goal")
    elif kind == "Interception":
        delta["defensive_actions"] = 1
        outcome = event.get("interception", {}).get("outcome", {}).get("name")
        delta["defensive_actions_successful"] = int(outcome not in {"Lost Out", "Lost"})
    elif kind == "Duel" and event.get("duel", {}).get("type", {}).get("name") == "Tackle":
        delta["defensive_actions"] = 1
        delta["defensive_actions_successful"] = int(_successful_duel(event))
    return delta


def metrics(counts):
    result = {key: counts[key] for key in (
        "passes_attempted", "passes_completed", "shots", "shots_on_target",
        "goals", "defensive_actions", "defensive_actions_successful",
    )}
    attempts = result["passes_attempted"]
    shots = result["shots"]
    result["pass_accuracy"] = result["passes_completed"] / attempts if attempts else None
    result["shots_on_target_ratio"] = result["shots_on_target"] / shots if shots else None
    return result


class MatchState:
    def __init__(self, window_seconds=600):
        if window_seconds <= 0:
            raise ValueError("Window must be positive")
        self.window_seconds = window_seconds
        self.teams = {}
        self.players = {}
        self.recent = {}
        self.history = deque()
        self.seen = set()
        self.events_processed = 0
        self.period = 0
        self.match_second = 0
        self.elapsed = 0.0
        self._period_time = 0.0
        self._offset = 0.0

    def apply(self, event):
        event_id = event.get("id")
        if event_id is not None and event_id in self.seen:
            return False
        period = int(event.get("period", 1))
        hours, minutes, seconds = event.get("timestamp", "00:00:00").split(":")
        period_time = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if period < self.period or (period == self.period and period_time < self._period_time):
            raise ValueError("Events must arrive in chronological period/timestamp order")
        if period > self.period:
            self._offset = self.elapsed
        self.period = period
        self._period_time = period_time
        self.elapsed = self._offset + period_time
        self.match_second = event_clock(event)
        while self.history and self.history[0][0] <= self.elapsed - self.window_seconds:
            _, team, delta = self.history.popleft()
            self.recent[team].subtract(delta)
        if event_id is not None:
            self.seen.add(event_id)
        self.events_processed += 1
        team = event.get("team", {}).get("name")
        if team:
            self.teams.setdefault(team, Counter())
            self.recent.setdefault(team, Counter())
            delta = event_delta(event)
            self.teams[team].update(delta)
            self.recent[team].update(delta)
            if delta:
                self.history.append((self.elapsed, team, delta))
            player = event.get("player", {})
            if player.get("id") is not None:
                record = self.players.setdefault(player["id"], {
                    "name": player.get("name"), "team": team, "counts": Counter(),
                })
                record["counts"].update(delta)
        return True

    def snapshot(self):
        return deepcopy({
            "period": self.period,
            "match_second": self.match_second,
            "elapsed": self.elapsed,
            "events_processed": self.events_processed,
            "window_seconds": self.window_seconds,
            "teams": {name: metrics(counts) for name, counts in self.teams.items()},
            "recent": {name: metrics(counts) for name, counts in self.recent.items()},
            "players": {key: {"name": row["name"], "team": row["team"],
                              **metrics(row["counts"])} for key, row in self.players.items()},
        })
