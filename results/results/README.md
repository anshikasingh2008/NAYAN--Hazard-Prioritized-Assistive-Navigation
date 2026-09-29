# NAYAN Model Training and Quantization Results

## Training Configuration

- Model: YOLOv8n
- Dataset: NAYAN custom 11-class dataset
- Dataset size: 500 images
- Train/validation split: 80/20
- Split seed: 42
- Epochs: 50
- Batch size: 16
- Image size: 640 × 640
- Training device: NVIDIA Tesla T4 GPU

## Validation Results

The model was evaluated on the held-out 20% validation set.

- mAP50: 0.3779
- mAP50-95: 0.2488

## Exported Models

### FP32 LiteRT

- File: `nayan_yolov8n_fp32.tflite`
- Size: 11.70 MB
- mAP50: 0.3743
- mAP50-95: 0.2367

### INT8 LiteRT

- File: `nayan_yolov8n_int8.tflite`
- Size: 3.18 MB
- mAP50: 0.3592
- mAP50-95: 0.1982

## FP16 Note

A separate FP16 `.tflite` artifact was not included.

The Ultralytics LiteRT exporter version used for this experiment does not support `quantize=16` for creating a separate FP16 TFLite file. An alternative ONNX-to-TFLite conversion was tested, but its resulting detections were invalid, so that artifact was excluded rather than reported as a valid model.

Therefore, no FP16 file or FP16 accuracy values are claimed in this repository.

## Files

- `nayan_yolov8n_best.pt` — trained PyTorch model
- `nayan_yolov8n_fp32.tflite` — FP32 LiteRT model
- `nayan_yolov8n_int8.tflite` — INT8 LiteRT model
- `model_comparison.csv` — model comparison data
- `NAYAN_Model_Training_Quantization.ipynb` — Colab training and export workflow
