import argparse
import csv
import time
from pathlib import Path

import cv2
import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT


def build_tracker_config(tracker: str, track_buffer: int | None, out_dir: Path) -> str:
    """Return a tracker yaml path. If track_buffer is given, write a modified copy
    of ultralytics' packaged config; otherwise use the packaged config name."""
    if track_buffer is None:
        return f"{tracker}.yaml"
    base = ROOT / "cfg" / "trackers" / f"{tracker}.yaml"
    with base.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["track_buffer"] = track_buffer
    custom = out_dir / f"{tracker}_buffer{track_buffer}.yaml"
    with custom.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(custom)


def track_video(
    video_path,
    out_dir="vision",
    tag="baseline",
    model_name="yolov8n.pt",
    tracker="botsort",         
    track_buffer=None,          
    imgsz=960,
    conf=0.3,
    vid_stride=2,
    max_frames=250,
    save_video=True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"tracked_{tag}.mp4"
    out_csv = out_dir / f"tracks_{tag}.csv"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if save_video:
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps / vid_stride, (w, h)
        )

    tracker_cfg = build_tracker_config(tracker, track_buffer, out_dir)
    model = YOLO(model_name)
    rows = []
    processed = 0
    start = time.time()

    results = model.track(
        source=str(video_path),
        tracker=tracker_cfg,
        classes=[0],
        conf=conf,
        imgsz=imgsz,
        vid_stride=vid_stride,
        persist=True,
        stream=True,
        verbose=False,
    )
    for frame_idx, r in enumerate(results):
        if frame_idx >= max_frames:
            break
        processed += 1
        if frame_idx % 10 == 0:
            print(f"[{tag}] processed {frame_idx} frames...", flush=True)

        frame = r.orig_img.copy()
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                rows.append(
                    {"frame": frame_idx, "track_id": tid,
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                )
                if writer is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, str(tid), (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if writer is not None:
            writer.write(frame)
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2"])
    df.to_csv(out_csv, index=False)

    summary = summarize(df, processed)
    summary.update(
        {
            "tag": tag, "model": model_name, "tracker": tracker,
            "track_buffer": track_buffer, "imgsz": imgsz, "conf": conf,
            "vid_stride": vid_stride, "frames": processed,
            "seconds": round(time.time() - start, 1),
        }
    )
    log_experiment(out_dir / "experiments.csv", summary)
    return df, summary


def summarize(df: pd.DataFrame, n_frames: int) -> dict:
    if df.empty:
        return {"unique_ids": 0, "ids_ge_25": 0, "ids_ge_100": 0,
                "median_len": 0, "max_len": 0, "det_per_frame": 0}
    lengths = df.groupby("track_id").size()
    per_frame = df.groupby("frame").size().reindex(range(n_frames), fill_value=0)
    return {
        "unique_ids": int(df["track_id"].nunique()),
        "ids_ge_25": int((lengths >= 25).sum()),
        "ids_ge_100": int((lengths >= 100).sum()),
        "median_len": float(lengths.median()),
        "max_len": int(lengths.max()),
        "det_per_frame": round(float(per_frame.mean()), 1),
    }


def log_experiment(path: Path, summary: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(summary)


def parse_args():
    p = argparse.ArgumentParser(description="Track players in a football clip")
    p.add_argument("--video", default="vision/sample.mp4")
    p.add_argument("--tag", default="baseline", help="name for this experiment")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--tracker", default="botsort", choices=["botsort", "bytetrack"])
    p.add_argument("--track-buffer", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=250)
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    df, summary = track_video(
        a.video, tag=a.tag, model_name=a.model, tracker=a.tracker,
        track_buffer=a.track_buffer, imgsz=a.imgsz, conf=a.conf,
        vid_stride=a.stride, max_frames=a.max_frames, save_video=not a.no_video,
    )
    print(summary)