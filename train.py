from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="data.yaml",
    epochs=80,
    imgsz=512,
    batch=6,
    device="cpu"
)