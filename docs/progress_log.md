We reviewed the match event dataset and confirmed the Barcelona pass schema, then built and verified a pass-accuracy analysis showing 917 passes with 820 successful and an overall accuracy of 89.42%. The project now includes a working analysis script and a regression test to ensure the Barcelona pass metrics remain accurate as the project evolves. 

Added player and team coaching recommendations based on passing, shooting, dribbling, duels, and defensive metrics.
Added a simple console report in visualization/player_report.py showing each player's name, metrics, and recommendations.
Validated the recommendation system and report with the bundled match data; all recommendation tests passed.
## 2026-10-27

- أضفنا وحدة `vision/detect_players.py` لاستخراج إطار من الفيديو باستخدام OpenCV.
- أضفنا كشف اللاعبين باستخدام نموذج YOLOv8 مع فلترة الأشخاص، وإظهار درجة الثقة ومربعات الكشف.
- أضفنا رسم نتائج الكشف وحفظها في `vision/output.jpg`، مع توفير فيديو تجريبي `vision/sample.mp4` وملف النموذج `yolov8n.pt`.
- حدّثنا تقرير اللاعبين في `visualization/player_report.py` لعرض تسمية اللاعب باللغة الإنجليزية ضمن التقرير.
- أضفنا `vision/track_players.py` لتتبّع اللاعبين عبر الفيديو باستخدام YOLO وBoT-SORT، مع دعم ByteTrack وإعدادات مثل `track_buffer` و`conf` و`imgsz`.
- أضفنا حفظ بيانات التتبّع بصيغة CSV، وتلخيص عدد المعرفات وطول المسارات ومتوسط الاكتشافات لكل إطار، وتسجيل نتائج التجارب في `vision/experiments.csv`.
- شغّلنا تجربة التتبّع الأساسية على 250 إطارًا، وسجّلنا 106 معرفات للاعبين، منها 45 مسارًا بطول 25 إطارًا على الأقل و15 مسارًا بطول 100 إطار على الأقل.
- أنشأنا فيديو التتبّع `vision/tracked_baseline.mp4` وملف النتائج `vision/tracks_baseline.csv`، وأضفنا نموذج YOLOv8 متوسط الحجم `yolov8m.pt` للتجارب القادمة.

## 2026-10-29 - Repository cleanup

- Moved event JSON to data/events/, local videos to data/videos/, and YOLO weights to models/.
- Removed the byte-identical duplicate sample video and superseded untagged generated outputs.
- Retained the named baseline video, track CSV, and experiment log in outputs/vision/.
- Moved tests into tests/ and restored the commented-out pass-accuracy regression test.
- Removed Python/test caches, normalized indentation, simplified main.py, and corrected report formatting.
- Centralized default asset paths in project_paths.py and separated optional vision dependencies.
- Added README.md with layout, current limitations, and run commands.
- Validation: four tests passed; report ran from another working directory; detection loaded the moved model and video and detected 15 people in a sample frame at imgsz=640.
- Earlier entries above retain their historical file paths. Current locations are documented in README.md.

## 2026-10-29 - Simulated live analysis

- Added incremental team/player match statistics and a rolling 10-minute window.
- Added timestamped event replay with pause/resume, adjustable speed, terminal display, and JSON export.
- Added team coaching hypotheses with numeric evidence, minimum sample sizes, and a 5-minute per-rule/team cooldown.
- Kept the offline player report available and unchanged in purpose; video integration remains future work.
- Validation: 12 tests passed, including full-match/prefix equivalence with offline counts, no future-event delivery, duplicate handling, period transitions, clock controls, and evidence gates.
- End-to-end replay delivered all 3762 events: Barcelona 820/917 passes, 25 shots, 9 on target; Alaves 164/246 passes, 3 shots, 0 on target. Export saved locally to outputs/live/final.json.

## 2026-10-29 - Local Arabic dashboard

- Added an Arabic RTL browser dashboard served by Python's standard library, with no extra dependencies.
- Connected live replay controls, scoreboard, cumulative team/player statistics, and evidence-based coaching notes to the existing engine.
- Added new/reviewed note views, reset, speed selection, and JSON download. One server session is shared across tabs.
- Added structured recommendation evidence for Arabic presentation while preserving console output.
- Validation: all 21 tests passed, including HTTP assets/API, export, invalid controls, clock behavior, alert persistence, and reset.
- Browser QA verified play/pause, 600x speed, reset, note review, and full replay completion (Barcelona 3-0; 820/917 passes).
- Frontend JavaScript syntax check passed. Preview saved locally to outputs/dashboard/preview.png.
- Start with: .\.venv\Scripts\python.exe -B main.py --dashboard

## 2026-10-29 - Shirt-team classification and tracking evaluation

- Added calibrated torso HSV profiles, grass rejection, unknown labels, kit-role candidates and per-track temporal voting.
- Added bounded full-frame tracking runs with original source timestamps, annotated video, CSV, summaries and camera-cut reset heuristic.
- Compared YOLOv8n/960/buffer30 with YOLOv8m/1280/buffer90 on frames 0-239; retained both runs.
- Added 43 assistant-reviewed source boxes and a one-to-one IoU evaluator with separate player/official identity diagnostics.
- Sample results: matched boxes 37 to 40; observed player ID changes 1/13 to 0/13; all-person ID changes remained 2. Quality processing was approximately 1.78 FPS on CPU, not real time.
- The small curated annotation is not an independent human benchmark. Unknowns, role confusion, occlusion and latency remain limitations.
- Generated video and frame previews are in outputs/vision/team_tracking/quality/. Full method and measurements are in docs/team_tracking.md.
- Final validation: all 25 tests passed; output MP4 decoded through frame 239. Fixed a Windows HTTP socket-reset edge case in dashboard origin rejection by consuming the bounded request body before replying.

## 2026-10-29 - Observed-only ball tracking

- Added generic sports-ball detection, pitch/white-ball/size filters, cached-person exclusion, motion association and consecutive-observation confirmation.
- Unknown/pending frames contain no position; trails are cleared on gaps and camera-cut resets. No interpolated ball locations or possession statistics are produced.
- Added ten assistant-reviewed calibration frames (eight visible centers, two uncertain), a center-based evaluator and review overlays.
- Initial adaptive search: 2/8 visible centers correct, 1 wrong, 5 missed. Revised full-frame search: 6/8 correct, 0 wrong, 2 missed, within a 20-pixel tolerance. The sample is not independent or held out.
- Processed 240 frames, producing 136 observed, 13 pending and 91 unknown rows. Measured 1.88 FPS with cached-person overlay; this excludes person inference and is not end-to-end real-time speed.
- Validation: all 30 tests passed; output MP4 decoded through frame 239; unknown rows verified to have no coordinates.
- Final artifacts: outputs/vision/ball_tracking/quality/ball_and_teams.mp4, ball_tracks.csv, candidates.json, summary.json and evaluation.json. Method and commands: docs/ball_tracking.md.

## 2026-10-29 - Frozen new-window validation

- Added start-frame support to person/ball runners with original source timestamps and source-window cache validation.
- Froze code, model and source hashes before reviewing predictions on frames 300-539; no detector/color/tracking thresholds were retuned on this window.
- Added 30 assistant-reviewed ball labels (18 visible, 5 not visible from one short correlated occlusion, 7 uncertain) and 25 boxes across five player trajectories.
- Results: 9/18 correct visible ball centers within 20 pixels, 9 misses, 0 wrong visible centers; 0 false visibility in 5 occluded samples. Player boxes matched 25/25 with 2 ID changes across 20 sparse transitions.
- The predeclared 80% ball-recall gate failed (50% measured), so possession-candidate functionality was deferred. These are same-video assistant-reviewed results, not independent human/independent-match validation.
- Added a repeatable report command verifying frozen hashes and output windows. Report: outputs/vision/validation_300_539/report.json; explanation: docs/validation_300_539.md.
- Validation: all 32 tests passed; both 240-frame output videos decoded through their last frame; source frame/time offsets and empty unknown positions verified. Separate CPU timings: people 1.74 FPS, ball with cached people 1.65 FPS.

## 2026-10-30 - Ball miss diagnostic review

- Added a separate frozen-settings diagnostic command capturing pre-filter detections and first rejection reasons on the nine missed samples.
- Replayed all 240 cached tracker frames; states/positions matched original output and all nine rerun survivor sets matched cached candidates.
- Provisional findings: six samples had no raw candidate within 20 pixels of the current annotation at confidence >= .05; one was rejected near a person; one awaited confirmation; one fell below tracker confidence .06.
- Generated an offline, self-contained Arabic review page with source/overlay crops, full source images, raw candidate details, local review notes and JSON export. Original annotations/scores remain unchanged and await user review.
- Verification: 34 tests passed; JavaScript syntax checked separately. Page: outputs/vision/ball_miss_review/index.html; audit: diagnosis.json; method: docs/ball_miss_review.md.

## 2026-10-30 - User-confirmed ball annotation correction

- User confirmed eight original miss-review labels and explicitly approved the proposed frame-330 center [1441, 511], replacing [1439, 591].
- Saved a separate version-two annotation file with per-frame human-review provenance; only nine of the 30 labels are user-confirmed.
- Recomputed evaluation and diagnostic matching against unchanged predictions: 9/18 correct visible centers and nine misses, with unchanged failure categories.
- Frame 330 remains a miss because no output position was emitted; its corrected ground truth does not improve the system's score.
- Verified every original frozen model/source/code/annotation hash is unchanged. Original results/review page remain version-one evidence.
- Approved labels: data/annotations/heldout_ball_300_539_v2_user_reviewed.json. Updated results: outputs/vision/ball_miss_review/user_confirmed_v2/.
