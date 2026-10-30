# AI-Powered Football Coach

Graduation-project prototype for match statistics and coaching support.
Reports and simulated live analysis use recorded event JSON. Video detection
and tracking are separate experiments. Connection to a real event feed and
video-based event recognition are future work.

## Project layout

```text
analysis/              Match and player analysis
data_layer/            Event loading and player statistics
recommendation/        Rule-based coaching recommendations
visualization/         Console player reports
vision/                Person detection and anonymous tracking
live/                  Incremental statistics, replay clock, live coaching alerts
tests/                 Automated regression tests
data/events/           Match-event JSON (15946.json)
data/videos/           Local source videos (match.mp4, sample.mp4)
models/                Local YOLO weights (yolov8n.pt, yolov8m.pt)
outputs/vision/        Generated images, tracking videos, CSVs, experiment logs
docs/                  Project history
project_paths.py       Shared absolute default asset paths
main.py                Report entry point
```

Videos, model weights, generated outputs, and the virtual environment remain
local and are ignored by Git. The existing named baseline experiment is retained
in `outputs/vision/`. Keep `data/events/15946.json` with the project.
The event JSON and local videos are separate inputs; synchronization is not
implemented.

## Run on Windows (PowerShell)

From the project directory, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m analysis.analysis
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

For a new environment, create it with `python -m venv .venv`, then install
`requirements.txt` for reports and tests, or `requirements-vision.txt` for the
additional video tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Optional: person detection and tracking
.\.venv\Scripts\python.exe -m pip install -r requirements-vision.txt
.\.venv\Scripts\python.exe -m vision.detect_players
.\.venv\Scripts\python.exe -m vision.track_players --tag baseline --max-frames 250
```

Vision commands expect `data/videos/sample.mp4` and `models/yolov8n.pt`.
Tracking accepts `--video` and `--model` to select other assets, for example
`--model models/yolov8m.pt`. Default paths work independently of the working
directory; explicitly supplied relative paths are relative to the caller.
YOLO labels people and tracker IDs are anonymous; these outputs do not yet
identify named players or recognize passes and shots.

For the new shirt-color and tracking experiment, run:

```powershell
.\.venv\Scripts\python.exe -B -m vision.team_tracking
```

This uses the existing `models/yolov8m.pt` quality configuration and writes a
colored 8-second video, tracking CSV, and summary to
`outputs/vision/team_tracking/quality/`. It distinguishes dark and white kits,
flags yellow-shirt officials as candidates, and leaves ambiguous detections
unknown. The profile is calibrated to the sample clip. See
[the experiment report](docs/team_tracking.md) for repeatable commands, the
comparison, annotation limits, and the CPU speed tradeoff.

For observed-only ball tracking on the same sample, with the cached team overlay:

```powershell
.\.venv\Scripts\python.exe -B -m vision.track_ball --players-csv outputs/vision/team_tracking/quality/tracks.csv
```

Outputs go to `outputs/vision/ball_tracking/quality/`. Missing or ambiguous ball
positions stay unknown and break the drawn trail. See
[the ball experiment report](docs/ball_tracking.md) for the evaluation, calibration
assumptions, commands and limitations. This does not count video passes or infer
possession.

A frozen follow-up test on original frames 300–539 found correct ball positions
on 9/18 visible samples (50%), so the project does not yet pass its prototype
gate for possession estimates. See [the validation report](docs/validation_300_539.md)
for the unchanged settings, reviewed labels, identity checks and repeat commands.

The nine misses now have a [diagnostic review](docs/ball_miss_review.md).
Open the self-contained `outputs/vision/ball_miss_review/index.html` in a browser
to compare original crops, current annotations and pre-filter detections, and
export review notes without changing the frozen annotations.

Dependency files list direct requirements without version pins. The existing
environment is preserved; installing on a new Python/platform combination may
require compatible PyTorch and OpenCV versions.

## Simulated live match analysis

### Local Arabic dashboard

```powershell
.\.venv\Scripts\python.exe -B main.py --dashboard
```

Open `http://127.0.0.1:8765` in a browser. If that port is occupied, use
`--port 8766`. No new dependencies are needed. The server binds only to this
computer. Stop it with Ctrl+C.

The Arabic dashboard starts paused and uses the same Python replay engine:

- Play/pause, reset, and speed selection from 1x to 600x.
- Cumulative team statistics, match clock, event progress, and player statistics.
- Evidence-backed recommendations separated into new and previously reviewed
  notes. Moving notes to history is local to the current browser tab and does
  not delete them from export.
- Download the current statistics and full alert history as JSON, including
  whether playback has completed.

One replay is shared by all tabs connected to the same server. Refreshing the
page preserves its state; restarting the server starts a new session. Playback
uses elapsed server time while running and catches up when requested again,
including after a browser tab has been inactive. The dashboard does not connect
to a live video or external event provider.

### Terminal playback

```powershell
# Roughly 2 match minutes per second, updating the terminal once a second
.\.venv\Scripts\python.exe -B main.py --live --speed 120

# Process the whole match immediately and export the final totals and alerts
.\.venv\Scripts\python.exe -B main.py --live --speed 0 --export outputs/live/final.json

# Tests, with temporary test files kept inside the workspace
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp=outputs/test-temp
```

In an interactive Windows terminal: **Space** pauses/resumes, **+** doubles
speed, **-** halves speed, and **q** stops playback. Ctrl+C also stops.
`--refresh` changes the terminal display interval in real seconds. Export on
stop contains only the events already delivered and marks the replay incomplete.
The module entry point `python -m live.cli` accepts the same options without
`--live`.

The scoreboard shows cumulative passes, completion rate, shots, shots on target,
and shot goals. Counts are also maintained per player, available in JSON export.
Own goals, shootout-specific scoring, possession estimates, and named-player video
identification are not implemented in this first version. Missing attempt counts
display as N/A rather than implying poor accuracy.

New live recommendations use the last 10 minutes of delivered play, with:

| Rule | Minimum attempts in window | Alert below |
| --- | --- | --- |
| Passing | 20 passes | 80% completed |
| Shooting | 5 shots | 35% on target |
| Defending | 10 tackles + interceptions | 50% successful |

Each alert includes its match time and supporting counts. The same team/rule
can alert at most once every 5 minutes of simulated play. These thresholds are
experimental coaching hypotheses, not trained tactical predictions. The existing
offline drill recommendations remain a separate baseline.

The replay reads the source file in advance for scheduling, but sends one due
event at a time to `MatchState`. Analysis and recommendations never receive future
events. Delivery follows period-local timestamps; the next period begins after
the previous period's last recorded event. Halftime waiting is skipped, while
the displayed clock uses the provider's match minute and second. Input must be
chronological; repeated event IDs are ignored by analysis.
