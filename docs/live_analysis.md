# Simulated live analysis

The first live milestone uses recorded event data as a simulated feed.

- `live/replay.py`: source loading, timestamp scheduling, speed and pause state.
- `live/engine.py`: cumulative player/team counts and a rolling event window.
- `live/recommendations.py`: evidence gates, explanatory alerts, and cooldowns.
- `live/cli.py`: terminal scoreboard, keyboard controls, and JSON export.
- `live/dashboard.py`: local HTTP server, synchronized replay controls, and JSON API.
- `live/web/`: Arabic dashboard with no frontend dependencies or build step.

`MatchState.apply(event)` is the boundary for a future event feed. It has no
reference to the replay source. `snapshot()` returns independent statistics so
consumers cannot mutate the state. Duplicate IDs do not count twice; timestamps
must be chronological within each period. Recent-window expiry uses a continuous
play clock rather than match minute numbers, which may overlap at added time.

Pass, shot, and defensive outcome definitions match the existing event-data
aggregation. Tests compare full replay and a received-event prefix to offline
counts, including player counts. Controls are tested against a deterministic
clock without waiting for a real match.

The video pipeline is not connected. No tactical model is trained by this
milestone. Live advice is currently team-level and uses configurable-in-code
heuristics; the original offline player report remains available via `main.py`.
