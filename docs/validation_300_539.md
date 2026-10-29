# Frozen same-video validation: original frames 300–539

## Outcome

**The current pipeline did not meet the predeclared prototype gate for starting
possession-candidate work.** Ball recall on reviewed visible frames was 9/18
(50%), below the fixed 80% threshold. No possession feature was added.

| Measure | Result |
| --- | ---: |
| Original frames processed | 300–539 inclusive (240 frames / 8 seconds) |
| Reviewed visible ball samples | 18 |
| Correct ball centers within 20 pixels | 9 |
| Missed visible ball samples | 9 |
| Wrong positions on visible samples | 0 |
| Reviewed not-visible samples | 5 |
| False visibility in those samples | 0 |
| Uncertain ball samples, excluded | 7 |
| Reviewed player boxes matched at IoU >= .4 | 25/25 |
| Kit labels correct, counting unknowns as errors | 20/25 |
| Correct among assigned A/B kit labels | 20/20 |
| Player ID changes / sampled identity transitions | 2/20 |
| Player processing FPS | 1.74 |
| Ball processing FPS with cached people | 1.65 |

The visible misses include the intentionally unconfirmed first frame, several
close-to-player samples, airborne samples, and a sample behind the goal net.
These are observed failure contexts, not proof of which internal stage caused
each miss. Raw detector errors and the conservative exclusion/filter policy both
need investigation on future training/development clips.

The separate stage speeds are not combined system throughput. The ball run
uses cached player detections; timings include video I/O and overlay and are
approximate CPU measurements.

## Protocol

The window was chosen and code/model/source hashes were saved in
`data/validation/heldout_300_539.json` before new-window predictions were
inspected. It does not overlap the previous calibration interval 0–239.
Detector confidence, shirt profiles, ball filters, motion/confirmation rules,
model weights and inference size were left unchanged.

The only runner additions enable selecting a starting source frame and validate
that cached player tracks cover the same original-frame interval. CSV frame
numbers and timestamps remain those of the source video rather than resetting
to zero. The report verifies frozen file hashes and output intervals.

Before inspecting ball predictions, original frames were reviewed at a
systematic ten-frame interval (300, 310, …, 530), plus 539. Five consecutive
frames 384–388 were added to inspect a short ball occlusion by players. This
gives 30 ball labels: 18 visible, 5 not visible and 7 uncertain. Player identity
and kit labels cover five trajectories at 300, 360, 420, 480 and 539 (25 boxes).
Review images and annotation overlays remain in the local `review/` directory.

These are assistant-produced, approximate visual labels, not independent human
ground truth. The five negative samples are correlated observations of one
short occlusion, so zero false visibility does not establish performance during
diverse occlusions. Identity checks are sparse and can miss intervening changes.
This is a frozen temporal holdout in the same source video, not an independent
match or a broadly representative benchmark. Some source images elsewhere in
the video had been inspected qualitatively in earlier work; ball thresholds
were calibrated on frames 0–239, not on this evaluation window.

## Predeclared decision rule

The code in `vision/validation.py`, hashed before scoring, requires all of:

- At least 10 visible ball samples and 5 not-visible samples.
- At least 80% correct centers on visible samples.
- At most 10% wrong centers on visible samples.
- At most 10% false visibility on not-visible samples.
- At least 10 sampled player identity transitions and at most 10% ID changes.

Only the visible-recall condition failed in this sample. Even passing this
prototype gate would not establish reliable real-world possession accuracy.
The fixed settings were **not retuned** to make this window pass. Once this
window is used for future development, it must not be called an untouched test
set again; retain a different clip for the next evaluation.

## Repeat

```powershell
.\.venv\Scripts\python.exe -B -m vision.team_tracking --start-frame 300 --max-frames 240 --output outputs/vision/validation_300_539/players
.\.venv\Scripts\python.exe -B -m vision.track_ball --start-frame 300 --max-frames 240 --players-csv outputs/vision/validation_300_539/players/tracks.csv --output outputs/vision/validation_300_539/ball
.\.venv\Scripts\python.exe -B -m vision.validate_segment
```

The report refuses scoring if frozen source/model/code or label hashes have
changed. It writes `outputs/vision/validation_300_539/report.json`, plus the
stage-specific `evaluation.json` files. The video is
`outputs/vision/validation_300_539/ball/ball_and_teams.mp4`.

The next development work should address missed ball detections near people
and during flight, using independently reviewed training/validation data and a
separate future test clip. Possession estimates remain deferred.
