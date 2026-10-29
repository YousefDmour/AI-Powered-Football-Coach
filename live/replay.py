"""Replay source and controllable clock, separate from match analysis."""

import json
import math
from pathlib import Path

from live.engine import MatchState
from live.recommendations import LiveRecommendations


def load_events(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(events, list) or not events:
        raise ValueError("Expected a nonempty list of match events")
    return events


class Replay:
    def __init__(self, events, speed=60.0):
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Speed must be finite and nonnegative")
        self.events = events
        self.speed = speed
        self.paused = False
        self.position = 0
        self.playhead = 0.0
        self.state = MatchState()
        self.recommendations = LiveRecommendations()
        self.schedule = []
        previous_period, previous_time, offset, elapsed = 0, 0, 0.0, 0.0
        for event in events:
            period = int(event.get("period", 1))
            hours, minutes, seconds = event.get("timestamp", "00:00:00").split(":")
            timestamp = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            if period < previous_period or (period == previous_period and timestamp < previous_time):
                raise ValueError("Replay events must be ordered by period and timestamp")
            if period > previous_period:
                offset = elapsed
            elapsed = offset + timestamp
            self.schedule.append(elapsed)
            previous_period, previous_time = period, timestamp

    @property
    def finished(self):
        return self.position == len(self.events)

    def advance(self, wall_seconds):
        """Return new alerts while delivering events due on the replay clock.

        Speed zero runs all remaining events immediately. A paused replay never
        delivers events, including when using speed zero.
        """
        if wall_seconds < 0:
            raise ValueError("Elapsed time cannot be negative")
        if self.paused:
            return []
        self.playhead = (float("inf") if self.speed == 0
                         else self.playhead + wall_seconds * self.speed)
        alerts = []
        while not self.finished and self.schedule[self.position] <= self.playhead:
            event = self.events[self.position]
            self.position += 1
            if self.state.apply(event):
                alerts.extend(self.recommendations.evaluate(self.state.snapshot()))
        return alerts
