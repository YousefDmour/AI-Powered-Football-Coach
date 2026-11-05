# Ball detection experiments

The local labeling batch has 200 images from one Barcelona–Bayern source family. The reviewed export has 197 examples: 142 with ball boxes and 55 without a visible ball. Three ambiguous images were excluded. Keep the local match and its shorter sample in the same source group when splitting footage.

## Results

A correct center is within 20 pixels of the reference on a 1920×1080 frame. Counts are frame counts; successive frames in a clip are correlated. Static-image comparisons use the highest-confidence ball detection at confidence .05. Video comparisons use the same causal tracker and person, appearance, and confirmation rules for both detectors.

| Comparison | Original or reference | Candidate | Scope |
| --- | --- | --- | --- |
| Single-match pilot on original SoccerNet subset | YOLOv8n 16/59; YOLOv8m 20/59 | Single-match YOLOv8n 13/59 | Two matches, positive frames |
| Multi-match model on newly reserved game | Single-match 34/76; YOLOv8m 35/76 | Multi-match 41/76 | One game, four clips |
| Multi-match model on original subset | Single-match 13/59; YOLOv8m 20/59 | Multi-match 19/59 | Historical comparison |
| Shared tracker on first two continuous clips | YOLOv8m 38/129, 9/100 absent-frame marks | Multi-match 35/129, 0/100 marks | Two reviewed out-of-view episodes |
| Grass relaxation on two development clips | Strict 35/129, 0/100 marks | No grass condition 46/129, 5/100 marks | Relaxed variants failed the development gate |
| Confidence .30 exception on two reserved clips | Strict 40/158, 11/11 marks | Exception 40/158, 11/11 marks | The repeated marks landed on a white cone |
| Distractor training on frozen continuous clips | Multi-match 75/287, 11/111 marks | Distractor model 73/287, 4/111 marks | The four remaining marks were at another location |
| Distractor training on new sampled images | Multi-match 43/79 | Distractor model 41/79 | Unused clips from previously evaluated test games |
| Different-venue absence control | Multi-match 21/74, 0/51 marks, 17-frame longest correct run | Distractor model 15/74, 0/51 marks, 10-frame longest run | One new clip, no confirmed white cones |

On the new sparse game, median inference including detector preprocessing and postprocessing was 9.9 ms for the multi-match YOLOv8n and 26.2 ms for the full-image YOLOv8m on the RTX 4060 Ti. The first continuous comparison measured 34.3 ms for combined player and ball detection with the multi-match model and 50.8 ms with YOLOv8m for both. These timings measure vision stages; live event extraction and recommendations have their own runtime and accuracy requirements.

The multi-match model trained on 427 images from three matches and validated on 71 images from a separate SoccerNet game. The distractor model adds 26 reviewed negative crops from the training games while retaining the same initial weights and training settings. Its results are mixed, so it stays experimental. The size gate rejected 15 correct raw detections in the different-venue control. Reviewing their box sizes is the next experiment. A different-venue test with verified white cones is still pending.

## Local inputs

- Set up the vision dependencies and place the existing YOLOv8n and YOLOv8m weights in `models/`. Training and these timing comparisons require CUDA-enabled PyTorch. The recorded Windows environment used Ultralytics 8.4.190 and PyTorch 2.14.1+cu130. The runners accept a CUDA-enabled environment or a project-local package overlay in `outputs/ball_training/cuda_packages`.
- Restore the reviewed local YOLO export to `outputs/ball_dataset/reviewed_001`. The [labeling guide](ball_dataset.md) describes preparing and importing a new batch. The recorded run used the original reviewed export, so a new labeling run can give different results.
- External sequences come from [SoccerNet Tracking](https://www.soccer-net.org/tasks/tracking), revision `3777f456d908570e10b19e7354744640e77c351e` of the [official archive](https://huggingface.co/datasets/SoccerNet/SN-Tracking-2023). Preparation commands fetch bounded ZIP ranges and verify member checksums. They write local images and labels to `outputs/`.

## Run from the repository root

Use empty destination directories for a new run. Keep existing local results as an archive. The following PowerShell commands show the order for the recorded workflow.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -B -m experiments.ball.prepare_soccernet_holdout_001
& $python -B -m experiments.ball.run_ball_pilot train
& $python -B -m experiments.ball.run_ball_pilot evaluate
& $python -B -m experiments.ball.prepare_multigame_002
& $python -B -m experiments.ball.run_ball_multigame_002 train
& $python -B -m experiments.ball.run_ball_multigame_002 evaluate

& $python -B -m experiments.ball.prepare_continuous 001
& $python -B -m experiments.ball.restore_negative_review continuous_001
& $python -B -m experiments.ball.run_continuous_001
& $python -B -m experiments.ball.prepare_continuous 002
& $python -B -m experiments.ball.restore_negative_review continuous_002
& $python -B -m experiments.ball.run_grass_filter_001 develop
& $python -B -m experiments.ball.run_grass_filter_001 verify

& $python -B -m experiments.ball.prepare_hard_negatives_001
& $python -B -m experiments.ball.run_hard_negatives_001 train
& $python -B -m experiments.ball.run_hard_negatives_001 evaluate
& $python -B -m experiments.ball.replay_hard_negatives_001

& $python -B -m experiments.ball.prepare_cone_holdout_001
& $python -B -m experiments.ball.prepare_cone_holdout_001 --window SNMOT-128 465 150
& $python -B -m experiments.ball.restore_negative_review cone_holdout_001
& $python -B -m experiments.ball.run_absence_control
```

The crop review coordinates and reviewed negative intervals are in `experiments/ball/reviews/`. The temporal review restore checks its exact source manifest. The optional `mine_hard_negatives_001` command regenerates the candidate preview sheets for manual inspection. It does not change the saved review decisions.
