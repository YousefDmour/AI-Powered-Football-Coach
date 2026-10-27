from pathlib import Path
import cv2
from ultralytics import YOLO


def extract_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def detect_players(frame, model_name="yolov8n.pt", conf=0.3, imgsz=1280):
    model = YOLO(model_name)
    result = model(frame, conf=conf, imgsz=imgsz, classes=[0], verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    return [{"bbox": b.tolist(), "confidence": float(s)} for b, s in zip(boxes, scores)]


def draw_detections(frame, detections, out_path):
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(out_path), frame)


if __name__ == "__main__":
    video = Path("vision/sample.mp4")
    frame = extract_frame(video, frame_index=100)
    detections = detect_players(frame)
    print(f"Detected {len(detections)} people")
    draw_detections(frame, detections, "vision/output.jpg")