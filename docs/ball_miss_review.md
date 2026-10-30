# Diagnostic review of the nine missed ball samples

## User-confirmed annotation revision (2026-10-30)

The user confirmed the visible ball and annotation on frames 300, 360, 370,
380, 390, 460, 470 and 539. Frame 330 was marked incorrect, then its proposed
replacement center **[1441, 511]** was explicitly confirmed. Its old center was
[1439, 591].

The approved version is
`data/annotations/heldout_ball_300_539_v2_user_reviewed.json`; chat review evidence
is recorded separately in `data/reviews/ball_frames_user_2026-10-30_confirmed.json`.
Only those nine frames are user-confirmed. The other 21 samples retain their
original assistant-only annotation provenance.

Recomputed evaluation and diagnosis are in
`outputs/vision/ball_miss_review/user_confirmed_v2/`. The score remains **9/18**:
the system emitted no position on frame 330, so correcting its reference center
does not change the missed-detection count. Diagnosis still identifies six
samples without a nearby initial detection, one person-filter rejection, one
confirmation delay and one tracker-confidence rejection.

The original frozen annotations, file hashes, evaluation, and HTML review page
remain unchanged as version-one evidence. The old page therefore still displays
the rejected frame-330 mark; use the approved version-two JSON for subsequent
development and scoring. No model/filter settings were changed by this revision.

## Original diagnostic procedure

No detector thresholds, tracker rules or frozen annotations were changed.
The diagnostic command verifies the frozen file hashes, reruns the original
model at the original size/confidence on the nine samples, and records all
post-NMS sports-ball candidates above confidence .05 before application filters.

It also replays **all 240 saved candidate frames**, preserving the tracker
history. Replay states and coordinates matched the original output, and the
rerun filter survivors matched the cached candidates on all nine samples.

## Provisional diagnosis

| Source frame | Finding relative to the current annotation |
| --- | --- |
| 300 | Correct nearby candidate passed filters; first observation remained pending confirmation. |
| 330 | No raw candidate within 20 pixels of the annotation at confidence >= .05. |
| 360 | No raw candidate within 20 pixels of the annotation at confidence >= .05. |
| 370 | Nearby candidate (confidence approximately .371) was rejected by person exclusion. |
| 380 | No raw candidate within 20 pixels of the annotation at confidence >= .05. |
| 390 | No raw candidate within 20 pixels of the annotation at confidence >= .05. |
| 460 | No raw candidate within 20 pixels of the annotation at confidence >= .05. |
| 470 | No raw candidate within 20 pixels of the annotation at confidence >= .05. |
| 539 | Nearby candidate passed image/person filters, but confidence .0507 was below the tracker's .06 minimum. |

This does not prove absence of weaker detector responses. Candidate-to-label
matching depends on the current approximate visual annotation and may associate
a nearby shoe with a ball label; user review is needed before treating the
diagnosis as confirmed. In this sample the six raw-detection misses dominate;
simply loosening person exclusion cannot recover all nine cases.

## Review together

The self-contained page is:
`outputs/vision/ball_miss_review/index.html`.

Open it in a browser. Every case includes a source crop without annotations,
an enlarged overlay, the full original frame, raw-candidate/filter details,
and a review selection with optional notes. Images are embedded, so the file
can be reviewed offline after the temporary preview server stops.

- Cyan circle: current ball annotation, awaiting human review.
- Green box: candidate passed application filters (not necessarily tracking).
- Red box: rejected candidate.
- Thin white box: cached person detection.

Choose whether the label is correct, needs correction, or is visually uncertain.
Notes are stored in the browser when available and can be exported with the
JSON download button. They do **not** update the frozen annotations or scores.
Review exports should be retained separately; any agreed annotation corrections
must create a versioned annotation set rather than overwrite the frozen result.

Repeat diagnostic generation from the project directory:

```powershell
.\.venv\Scripts\python.exe -B -m vision.diagnose_ball_misses
```

The full audit is saved in `diagnosis.json`; review images are retained in
`images/`. Software verification: 34 tests passed, including filter-explanation
equivalence and replay with intermediate frames preserved.
