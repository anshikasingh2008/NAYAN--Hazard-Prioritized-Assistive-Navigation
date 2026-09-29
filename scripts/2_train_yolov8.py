from ultralytics import YOLO

# NAYAN YOLOv8n model training
# Dataset is prepared with a reproducible 80/20 train-validation split.
# Training configuration used for the NAYAN model training experiment.

MODEL_PATH = "yolov8n.pt"
DATA_YAML = "data.yaml"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16

model = YOLO(MODEL_PATH)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0,
    name="nayan_hazard_detector",
    patience=10
)

print("Training complete.")
print("Best weights are saved under the training run's weights directory.")

# Validate the trained model
metrics = model.val(data=DATA_YAML)

print("Validation results:")
print(metrics)
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
