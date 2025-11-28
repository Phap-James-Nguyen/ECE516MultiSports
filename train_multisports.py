from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("dataset") / "multisports.yaml"

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=5,
        imgsz=640,
    )

if __name__ == "__main__":
    main()
