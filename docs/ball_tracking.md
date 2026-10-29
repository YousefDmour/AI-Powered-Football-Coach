# Ball detection and observed-only tracking

This is a bounded video experiment on the first 240 frames / 8 seconds of the
same sample used for team tracking. It does not generate possession, passes,
shots, or coaching recommendations from video.

## Run

```powershell
# Ball plus the already-computed team/player overlay
.\.venv\Scripts\python.exe -B -m vision.track_ball --players-csv outputs/vision/team_tracking/quality/tracks.csv

# Ball alone (no cached person exclusion or team overlay)
.\.venv\Scripts\python.exe -B -m vision.track_ball --output outputs/vision/ball_tracking/ball_only

# Evaluate the selected sample
.\.venv\Scripts\python.exe -B -m vision.evaluate_ball --tracks outputs/vision/ball_tracking/quality/ball_tracks.csv --annotations data/annotations/sample_ball.json --output outputs/vision/ball_tracking/quality/evaluation.json
```

Uses the existing local `models/yolov8m.pt` and optional vision dependencies.
No weights were downloaded and no detector was trained by this milestone.
Use `--video`, `--model`, `--max-frames` and `--output` for other bounded runs.
Existing files in the selected output directory are replaced. A cached player
CSV must have the matching source and sufficient processed frames in its
neighboring `summary.json`; it is not silently attached to a different video.

## Detection and association

- The local model's generic sports-ball class (32) produces candidates.
- Default search is a full 1280-pixel inference each frame. An optional
  `--search adaptive` uses a crop around the previous detection with periodic
  global scans; it is experimental and can stay on the wrong object.
- Candidate filters use bounding-box dimensions/aspect ratio, a green pitch
  neighborhood and a minimum white-pixel fraction. The latter is calibrated to
  this predominantly white ball, not universal football appearance.
- With cached player boxes, candidates inside or immediately next to a person
  are rejected. This deliberately also misses some real balls at players' feet.
- A distance gate associates consecutive detections. Two consecutive supported
  observations are required before displaying the ball. Ambiguous competing
  candidates, low confidence and missing detections produce no position.
- A strong independent global detection can restart association after the local
  track is lost or wrong, but requires confirmation on another frame.
- Abrupt camera-change heuristics reset state. This segment has no detected cut;
  camera-cut behavior is not validated against annotated cuts.

## Output contract

`ball_tracks.csv` contains exactly one row per processed source frame:

- `observed`: a measured candidate passed the current heuristics; coordinates
  and detector confidence are included. This is not guaranteed ground truth.
- `pending`: an unconfirmed observation; position is withheld.
- `unknown`: no reliable association; position is withheld.

Missing positions are empty CSV fields, not extrapolated or interpolated points.
The video trail contains only adjacent observed positions and is cleared on any
gap. Orange markers show accepted ball observations; no marker is drawn for
pending or unknown frames. Green neighborhood checks and motion gates do not
prove that a candidate is the ball.

Files include `ball_and_teams.mp4`, selected frame previews, `ball_tracks.csv`,
filtered `candidates.json`, `summary.json`, and the separately generated
`evaluation.json`. Timings include ball inference, video reading/writing and
cached-person overlay, **not person inference** or model construction. They
must not be described as end-to-end live system speed.

## Evaluation limits

`data/annotations/sample_ball.json` holds ten assistant-reviewed original source
frames: eight with approximate visible ball centers and two with uncertain
visibility. The uncertain frames are excluded; they are not scored as true
negatives. No confidently invisible frames are included, so a rate of false
visibility during true occlusion is not established.

The evaluator counts a correct observed center within 20 pixels of its reviewed
center, a wrong observed center outside that radius, and a missed visible frame
when no position is emitted. It reports uncertainty explicitly. These annotations
are a small curated calibration sample, not a held-out or independently
human-labeled benchmark. Review images are retained locally in
`outputs/vision/ball_tracking/review/`.

Initial adaptive runs revealed false tracks on footwear and poor recall.
The measured comparison on the same eight visible sample frames was:

| Measurement | Initial adaptive experiment | Revised full-frame experiment |
| --- | ---: | ---: |
| Correct centers within 20 pixels | 2 / 8 | 6 / 8 |
| Wrong centers on visible sample frames | 1 | 0 |
| Missed visible sample frames | 5 | 2 |
| Uncertain sample frames excluded | 2 | 2 |
| Measured ball-processing FPS with cached people | 3.13 | 1.88 |

The first frame is intentionally pending until another observation confirms the
candidate. Frame 60 was missed in the revised output. The revised run emitted
136 observed, 13 pending and 91 unknown rows across 240 frames; availability is
not accuracy. No claim of zero false detections across the entire video follows
from zero wrong centers on six accepted sample observations.

The filters/search policy were revised while reviewing this calibration sample;
the apparent improvement must be tested on independent clips. The final run's
results are recorded in `outputs/vision/ball_tracking/quality/evaluation.json`.
Exploratory `sample/` and `revised/` outputs retain older settings;
`interrupted_adaptive/` is incomplete and is excluded from comparison.
Passing all 30 software tests confirms state behavior and evaluation calculations;
it does not establish detection accuracy.

This version needs further detector improvement and independent validation
before estimating a possession candidate, counting passes or feeding video
statistics into the match dashboard. Pixel motion is not pitch speed/distance.
