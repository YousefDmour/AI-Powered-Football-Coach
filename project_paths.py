"""Default asset paths, resolved independently of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EVENTS_FILE = PROJECT_ROOT / "data" / "events" / "15946.json"
SAMPLE_VIDEO = PROJECT_ROOT / "data" / "videos" / "sample.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8n.pt"
VISION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vision"
