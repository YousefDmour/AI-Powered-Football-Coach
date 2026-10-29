"""Bounded, reproducible person tracking and shirt-team experiment on local video."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from project_paths import DEFAULT_MODEL, SAMPLE_VIDEO, VISION_OUTPUT_DIR
from vision.team_colors import PALETTE, TeamSmoother, classify_shirt, shirt_features
from vision.track_players import build_tracker_config, summarize

QUALITY_MODEL = DEFAULT_MODEL.with_name("yolov8m.pt")


def run(video=SAMPLE_VIDEO, output=VISION_OUTPUT_DIR / "team_tracking" / "quality",
        model=QUALITY_MODEL, max_frames=240, imgsz=1280, conf=.2, buffer=90, start_frame=0):
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    if start_frame < 0:
        raise ValueError("start_frame cannot be negative")
    video, model, output = Path(video), Path(model), Path(output)
    if not video.is_file() or not model.is_file():
        raise FileNotFoundError("The local source video and YOLO model must exist")
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Video FPS is unavailable")
    width, height = int(cap.get(3)), int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(str(output / "teams.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        cap.release()
        raise OSError("Video writer could not be opened")
    rows, source_frame, cuts = [], start_frame, []
    smoother = TeamSmoother()
    previous_histogram = None
    selected = {start_frame+offset for offset in (0,60,120,180,max_frames-1)}
    started = time.perf_counter()
    try:
        tracker = build_tracker_config("botsort", buffer, output)
        detector = YOLO(str(model))
        started = time.perf_counter()
        while source_frame < start_frame+max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # Abrupt shot changes invalidate association. Gradual pans are retained.
            thumb = cv2.cvtColor(cv2.resize(frame, (160,90)), cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([thumb], [0,1], None, [24,16], [0,180,0,256])
            cv2.normalize(histogram, histogram)
            cut = previous_histogram is not None and cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA) > .65
            if cut:
                cuts.append(source_frame)
                smoother.reset()
                for existing in getattr(detector.predictor, "trackers", []):
                    existing.reset()
            previous_histogram = histogram
            result = detector.track(frame, persist=True, tracker=tracker, classes=[0],
                                    conf=conf, imgsz=imgsz, verbose=False)[0]
            annotated = frame.copy()
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().tolist()
                ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None]*len(boxes)
                for bbox, confidence, track_id in zip(boxes, confidences, ids):
                    features = shirt_features(frame, bbox)
                    raw, support = classify_shirt(features)
                    # Segment key prevents tracker-ID reuse after camera cuts.
                    segment = len(cuts)
                    key = f"{segment}:{track_id}" if track_id is not None else None
                    label = smoother.update(key, raw) if key is not None else "unknown"
                    x1,y1,x2,y2 = map(int,bbox)
                    rows.append({"source_frame": source_frame, "time_seconds": source_frame/fps,
                                 "segment": segment, "track_id": key, "detector_id": track_id,
                                 "x1": x1,"y1":y1,"x2":x2,"y2":y2,
                                 "detection_confidence":confidence,"raw_team":raw,"team":label,
                                 "shirt_support":support, **(features or {})})
                    color = PALETTE[label]
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                    caption = f"{key or '?'} {label}"
                    cv2.putText(annotated,caption,(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
            cv2.rectangle(annotated,(0,0),(width,44),(25,30,25),-1)
            cv2.putText(annotated, f"A: dark blue/red | B: white | yellow: official candidate | gray: unknown | frame {source_frame}",
                        (15,28),0,.65,(235,240,230),1,cv2.LINE_AA)
            writer.write(annotated)
            if source_frame in selected:
                cv2.imwrite(str(output/f"frame_{source_frame:04}.jpg"),annotated)
            source_frame += 1
            if (source_frame-start_frame) % 30 == 0:
                print(f"{output.name}: {source_frame-start_frame}/{max_frames} frames",flush=True)
    finally:
        cap.release()
        writer.release()
    elapsed = time.perf_counter()-started
    dataframe = pd.DataFrame(rows, columns=["source_frame","time_seconds","segment","track_id","detector_id",
        "x1","y1","x2","y2","detection_confidence","raw_team","team","shirt_support","blue","white","yellow","red"])
    dataframe.to_csv(output/"tracks.csv",index=False)
    tracked = dataframe.dropna(subset=["track_id"]).rename(columns={"source_frame":"frame"})
    tracked = tracked.copy()
    tracked["frame"] -= start_frame
    processed_frames = source_frame-start_frame
    summary = summarize(tracked, processed_frames)
    assigned = dataframe["team"].isin(["team_a","team_b"])
    summary.update({"source": str(video.resolve()),"source_fps":fps,"source_start_frame":start_frame,
        "processed_frames":processed_frames,"source_seconds":processed_frames/fps,"wall_seconds":round(elapsed,3),
        "processing_fps":round(processed_frames/elapsed,2),"model":str(model.resolve()),"imgsz":imgsz,
        "conf":conf,"track_buffer":buffer,"stride":1,"scene_cuts":cuts,
        "team_assigned_detections":int(assigned.sum()),"all_detections":len(dataframe),
        "label_counts":dataframe["team"].value_counts().to_dict(),
        "notes":"Color-profile heuristic for this clip; anonymous IDs; no ball or event recognition. Track lengths are diagnostics, not ID accuracy."})
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
    return dataframe,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video",type=Path,default=SAMPLE_VIDEO)
    parser.add_argument("--output",type=Path,default=VISION_OUTPUT_DIR/"team_tracking"/"quality")
    parser.add_argument("--model",type=Path,default=QUALITY_MODEL)
    parser.add_argument("--max-frames",type=int,default=240)
    parser.add_argument("--start-frame",type=int,default=0)
    parser.add_argument("--imgsz",type=int,default=1280)
    parser.add_argument("--conf",type=float,default=.2)
    parser.add_argument("--buffer",type=int,default=90)
    args=parser.parse_args()
    run(args.video,args.output,args.model,args.max_frames,args.imgsz,args.conf,args.buffer,args.start_frame)


if __name__ == "__main__":
    main()
