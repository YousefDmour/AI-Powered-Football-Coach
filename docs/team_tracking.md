# Team colors and tracking experiment

This prototype processes the first **240 original frames (8 seconds at 30 FPS)**
of `data/videos/sample.mp4`, a continuous broadcast view. It is a separate
match from the event JSON used by the dashboard. No named identity or event
statistics are inferred from the video.

## Run

With the optional vision requirements installed:

```powershell
# Quality configuration; local yolov8m.pt is required
.\.venv\Scripts\python.exe -B -m vision.team_tracking

# Faster baseline for comparison on the same source interval
.\.venv\Scripts\python.exe -B -m vision.team_tracking --model models/yolov8n.pt --imgsz 960 --conf 0.3 --buffer 30 --output outputs/vision/team_tracking/baseline_v2

# Repeat the small visual evaluation
.\.venv\Scripts\python.exe -B -m vision.evaluate_tracking --tracks outputs/vision/team_tracking/quality/tracks.csv --annotations data/annotations/sample_teams.json --output outputs/vision/team_tracking/quality/evaluation.json
```

`--video`, `--max-frames`, `--model`, `--imgsz`, `--conf`, `--buffer`, and
`--output` allow bounded experiments. Use a different output directory to retain
earlier results; files in the selected directory are replaced on rerun.

Outputs include `teams.mp4`, `tracks.csv`, `summary.json`, and selected annotated
frames. CSV rows preserve original frame numbers and timestamps, raw shirt
classification, smoothed classification, detection confidence, and shirt pixel
support. Pixel support is not a calibrated class probability.

## Method and labels

- Detect people using the existing local YOLO weights; associate using BoT-SORT
  with camera-motion compensation from the packaged configuration.
- Use every frame, rather than skipping alternating frames. This changes the
  source interval relative to the old 250-frame/stride-2 baseline, so that old
  baseline is not used as the direct comparison here.
- Crop the central upper torso. Remove green turf pixels and require sufficient
  remaining evidence before applying HSV color thresholds.
- `team_a`: dark blue/maroon kit; `team_b`: white kit. This is a manually
  calibrated color profile for this clip, not automatic kit discovery.
- `official_candidate`: yellow shirt; `other_kit`: dominant red, which can
  include a goalkeeper. These are kit/role candidates, not validated role labels.
  Green goalkeepers and unusual kits can remain unknown.
- Apply a five-observation vote per tracker ID. First observations and ambiguous
  colors remain `unknown`; no player is forcibly assigned to one of the teams.
- Reset tracker/color memory on detected abrupt histogram changes. Segment
  prefixes prevent ID reuse across detected cuts. This detector is heuristic and
  did not trigger in the evaluated continuous segment; cut handling has not been
  validated on an annotated camera-cut benchmark.

## Recorded comparison

Both configurations used all source frames 0–239 and the same revised color
classifier. The initial exploratory `baseline/` and `tuned/` folders used an
earlier color classifier and are not included in the comparison.

| Measurement | Baseline (n / 960 / .3 / buffer 30) | Quality (m / 1280 / .2 / buffer 90) |
| --- | ---: | ---: |
| Distinct tracker IDs | 92 | 63 |
| Median observations per track | 14.5 | 28 |
| Annotated boxes matched at IoU >= .4 | 37 / 43 | 40 / 43 |
| Correct kit labels among matched boxes, including unknown errors | 27 / 37 | 27 / 40 |
| Correct team labels among matched boxes assigned A/B | 22 / 23 | 21 / 21 |
| Observed player ID changes / sampled player transitions | 1 / 13 | 0 / 13 |
| All observed ID changes, including officials | 2 / 16 | 2 / 17 |
| Measured processing FPS on this CPU | 10.36 | 1.78 |

These results show a tradeoff, not universal superiority: the quality setup
matched more reviewed boxes and preserved the sampled player identities, but
had more unknown kit labels and much higher compute cost. Fewer tracker IDs
alone do not establish better identity accuracy. Officials still changed IDs.

Timings include inference, classification and video writing, excluding model
construction; some experiments overlapped on the CPU. They are approximate
run timings, not a controlled performance benchmark. Neither configuration
processed this 30 FPS source in real time on the current CPU.

## Annotation and limits

`data/annotations/sample_teams.json` contains **43 approximate boxes** reviewed
visually by the assistant on source frames 0, 60, 120, 180 and 239. Five selected
outfield-player trajectories were given consistent anonymous labels where
visually clear; uncertain identities were omitted. Officials were evaluated
separately from players. Original-grid and annotation-review images are saved
locally in `outputs/vision/team_tracking/review/`.

This is a curated prototype/calibration sample, **not an independent human
ground-truth benchmark or held-out test set**. The report matches boxes one to
one by IoU. The sample does not annotate every visible person, so unmatched
predictions cannot be interpreted as false positives. Sparse identity checks
miss switches between selected frames and are not IDF1/HOTA scores. A coach or
independent annotator should review the annotations and a larger unseen clip
before any accuracy claim.

The annotated video is ready for inspection; team recognition, goalkeeper/referee
separation, tracking through occlusion and real-time speed remain incomplete.
Ball tracking is a separate follow-up experiment documented in
[ball_tracking.md](ball_tracking.md). Pass/shot event recognition and connection
of video-derived statistics to the dashboard remain unimplemented.
