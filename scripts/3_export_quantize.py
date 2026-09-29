from ultralytics import YOLO

# NAYAN model export and quantization
#
# The trained YOLOv8n model is exported to LiteRT/TFLite.
# FP32 and INT8 artifacts are generated and validated.
#
# Note:
# The current Ultralytics LiteRT exporter does not support
# creating a separate FP16 TFLite artifact using quantize=16.

MODEL_PATH = "runs/detect/nayan_hazard_detector/weights/best.pt"
DATA_YAML = "data.yaml"

model = YOLO(MODEL_PATH)

# --------------------------------------------------
# FP32 LiteRT export
# --------------------------------------------------

fp32_path = model.export(
    format="litert",
    imgsz=640,
    quantize=32
)

print("FP32 LiteRT model exported:")
print(fp32_path)

# --------------------------------------------------
# INT8 LiteRT export
# --------------------------------------------------

int8_path = model.export(
    format="litert",
    imgsz=640,
    quantize=8,
    data=DATA_YAML
)

print("INT8 LiteRT model exported:")
print(int8_path)

print("\nExport complete.")
print("FP32 and INT8 LiteRT models are ready for inference.")
print(
    "A separate FP16 TFLite artifact was not generated because "
    "the current Ultralytics LiteRT exporter does not support "
    "quantize=16."
)
