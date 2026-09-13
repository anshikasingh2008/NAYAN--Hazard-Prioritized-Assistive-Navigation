from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=15,
    imgsz=416,
    batch=8,
    name="nayan_hazard_detector",
    patience=10,
)

print("Training complete. Best weights saved under runs/detect/nayan_hazard_detector/weights/best.pt")

metrics = model.val()
print(metrics)
