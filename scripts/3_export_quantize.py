from ultralytics import YOLO

model = YOLO("runs/detect/nayan_hazard_detector/weights/best.pt")

model.export(format="tflite", int8=False)

print("Exported quantized model to runs/detect/nayan_hazard_detector/weights/best_saved_model/")
print("Use the .tflite file inside that folder for on-device inference (Step 5).")
