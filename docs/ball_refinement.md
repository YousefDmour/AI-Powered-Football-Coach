# Small-ball refinement experiment

## Result and decision

Overlapping image tiles recovered additional small-ball observations. Keep this as an experimental runner; the current results do not meet the earlier 80% visible-frame recall gate and do not justify possession extraction or live tactical recommendations from video.

| Sampled window | Original baseline | Selected tiled_strict | Wrong observed centers | Negative labels |
| --- | --- | --- | --- | --- |
| Development 300-539 | 9/18 (50%) | 11/18 (61.1%) | 0 in both | 5, all from one correlated occlusion |
| Prospective 600-839 | 2/10 (20%) | 4/10 (40%) | 0 in both | 0; false visibility unmeasured |

A correct center is within 20 pixels of the annotation. Unknown and pending states count as misses on visible samples. Development contains 7 excluded uncertain samples; the prospective window contains 3. Only nine development annotations were user-confirmed, including corrected frame 330 at [1441, 511]. Remaining development labels and all prospective labels were assistant-reviewed. These small same-video samples are not independent human or independent-match validation.

The prospective window was reserved before development scoring, and tiled_strict was selected and its source/model/annotation hashes frozen before tiled predictions on that window. It recovered sampled frames 800 and 820. It still missed six visible samples. Do not select a different variant using these prospective scores.

## Implementation

`vision/ball_refinement.py` searches six overlapping 960x720 tiles of the 1920x1080 source with existing YOLOv8n weights, image size 960 and confidence .05. Coordinates are mapped back to the source; duplicate views are merged. The runner combines accepted tile candidates with same-frame cached full-image YOLOv8m candidates. The original color/size/person rejection filters and causal BallTrack confirmation remain in effect. No annotation coordinates enter detection, no future detections are used, and unknown positions stay empty.

`vision/refine_ball_video.py` produces per-frame CSVs for baseline, tiled_strict and experimental tiled_feet, raw candidate audit JSON, an overlay video and timings. Its default selected output is tiled_strict. The original production/baseline modules and original validation evidence remain unchanged; this is a separate experiment.

The near-foot exception was explored using confidence .30, two distinct tile views, lower-body proximity, and a recent confirmed strict track. The unguarded version degraded development recall to 9/18; the guarded version tied strict at 11/18, so the exception is not selected. Initial detector-only probe files used .35 and measure candidate availability, not tracked recall. Do not interpret them as final evaluation.

## Outputs and timing

- `outputs/vision/ball_refinement/development/comparison.json`: development comparison including the rejected unguarded feet variant.
- `outputs/vision/ball_refinement/development_guarded/comparison.json`: guarded development comparison. Its video was exploratory feet mode; use CSV tiled_strict for selected scores.
- `outputs/vision/ball_refinement/future/comparison.json`: prospective comparison; selected strategy is tiled_strict.
- `outputs/vision/ball_refinement/future/ball_refined.mp4`: selected strategy, 240 frames, source frames 600-839.
- `outputs/vision/ball_refinement/future/verification.json`: hash, frame-range, unknown-coordinate and full-video decode checks.
- `data/validation/ball_refinement_future_window.json`: prospective reservation and frozen hashes.

Prospective refinement took 117.420 s with baseline and person detections cached. Adding the original baseline ball stage (214.951 s) gives approximately 0.72 FPS for the serial ball stage, excluding person inference. This is not real time. Cache-only replays have no detector throughput estimate. Development and probe timings had concurrent CPU workloads and are not a controlled model speed comparison.

The project directory was renamed to `AI-Powered Football Coach` after inference. Historical JSON source paths retain the former directory name to preserve evidence; current relative artifact paths above resolve from the project root.

## Reproduce scoring and checks

Run from the current project root:

```powershell
.\.venv\Scripts\python.exe -B -m vision.report_ball_refinement --output outputs/vision/ball_refinement/future --annotations data/annotations/ball_refinement_future_600_839.json
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp=outputs/test-temp
```

The generic score command reports all variants; consult this document and the frozen manifest for the preselected strategy and provenance. Verification on 2026-11-05: 38 tests passed, all original and prospective frozen hashes matched, and all 240 video frames decoded. Non-observed CSV rows contain no coordinates.

## Next bounded milestone

Collect human-reviewed ball boxes on additional clips, including near-foot balls, motion blur, airborne balls, difficult backgrounds and confidently negative frames. Split by clip before training a ball-specific detector. Keep this experiment as a benchmark. Review the six remaining prospective misses for future development, but reserve another unseen clip for the next evaluation. Live integration needs both substantially stronger detection and a measured latency improvement.
