import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_paths import DEFAULT_MODEL, SAMPLE_VIDEO, VISION_OUTPUT_DIR


def extract_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def detect_players(frame, model_name=str(DEFAULT_MODEL), conf=0.3, imgsz=1280):
    model = YOLO(model_name)
    result = model(frame, conf=conf, imgsz=imgsz, classes=[0], verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    return [{"bbox": b.tolist(), "confidence": float(s)} for b, s in zip(boxes, scores)]


def draw_detections(frame, detections, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if not cv2.imwrite(str(out_path), frame):
        raise OSError(f"Could not save detections to {out_path}")


if __name__ == "__main__":
    video = SAMPLE_VIDEO
    frame = extract_frame(video, frame_index=100)
    detections = detect_players(frame)
    print(f"Detected {len(detections)} people")
    draw_detections(frame, detections, VISION_OUTPUT_DIR / "output.jpg")
