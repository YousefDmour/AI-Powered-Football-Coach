# First ball-labeling batch

## Open the local review page

From the project directory:

```powershell
.\.venv\Scripts\python.exe -B -m vision.ball_dataset serve --port 8781
```

Open http://127.0.0.1:8781/. The server binds to loopback and serves only the batch directory; it does not upload images or annotations. Stop it with Ctrl+C in the terminal.

## Review workflow

1. Draw a tight rectangle around the visible ball by dragging over the image. Zoom x2 or x4 and scroll to inspect small balls. Drawing a new box replaces the old one.
2. Use the confidently-not-visible button only when no ball can be seen. Use uncertain when you cannot judge its position or visibility; do not guess a box.
3. Move with Previous/Next, N/P, the image-number field, or First unreviewed. U marks uncertain. Clear restores the pending state.
4. The browser saves progress locally for this batch and browser origin. Download JSON regularly and before clearing browser data. Import JSON restores a backup after confirmation. Saving in the browser does not write training labels on disk.
5. Send the downloaded ball-labels-<batch-id>.json for import. No automatic labels have been supplied: every image begins pending.

The downloaded JSON records the batch ID, per-image status, and box coordinates in original image pixels. Visible labels require a valid box. Confident negative images export an empty YOLO label file. Uncertain and pending images are excluded, not treated as negatives.

## Dataset and provenance

`outputs/ball_dataset/batch_001/manifest.json` records source SHA256, source frame indices, times, dimensions and source group. The first batch contains 200 images from data/videos/match.mp4 (50 FPS, 12,836 frames). Sampling considers half-second candidates, rejects low thumbnail change relative to the last retained candidate, then spreads the selected 200 over the clip. This heuristic reduces adjacent similarity; manual review still needs to identify unsuitable or redundant pictures. Quality is not guaranteed by sampling.

No images are extracted from sample.mp4. However, visual inspection shows both local sources contain Barcelona/Bayern footage, so different file hashes do not establish independence. Treat both as one source family. If this batch is used for training, previous sample.mp4 scores become development evidence, and a new match must supply independent evaluation. Source-group separation is required before training. This first batch comes from one video and is insufficient for independent validation. Additional clips/matches, plus a visual check for repeated footage across files, are needed before making train/validation/test splits. No data.yaml is created yet.

The batch workflow uses local media. The separate [experiment guide](ball_experiments.md) covers later SoccerNet sampling, match-level splits and evaluation.

## Import human-reviewed JSON

```powershell
.\.venv\Scripts\python.exe -B -m vision.ball_dataset import-review --review "C:\Users\user\Downloads\ball-labels-BATCH_ID.json" --output outputs/ball_dataset/reviewed_001
```

Replace the filename with the actual download. Import requires a matching batch ID, rejects unknown image IDs, invalid/empty/out-of-bounds boxes and conflicting negative labels, and refuses to overwrite a nonempty export. The export includes images/, labels/, the submitted review and an audit manifest. It is staging data rather than a training configuration. Original batch files are preserved.

To prepare another source, use a new empty output directory:

```powershell
.\.venv\Scripts\python.exe -B -m vision.ball_dataset prepare --video data/videos/another_clip.mp4 --output outputs/ball_dataset/batch_002 --count 200
```

Use the same --batch path for serving and importing that batch. The preparer refuses to overwrite an existing batch and reports if there are too few sufficiently different candidates. Generated images and reviews remain under the Git-ignored outputs directory; downloaded JSON backups should be kept separately.

## Checks

On 2026-11-05, all 47 project tests passed and the labeler JavaScript passed Node syntax validation. A simulated DOM smoke check exercised drawing, zoom coordinate mapping, navigation, statuses, local persistence, export and foreign-batch rejection. All 200 selected images were re-extracted by sequential decoding and their dimensions verified; see verification.json. Import tests cover coordinate conversion, positive/negative output pairing, exclusion of uncertain/unreviewed images, foreign batch rejection, invalid boxes and overwrite protection.
