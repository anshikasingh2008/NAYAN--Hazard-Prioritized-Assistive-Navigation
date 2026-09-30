import cv2
import time
import pyttsx3
from ultralytics import YOLO
from importlib import import_module
from pathlib import Path

hazard_priority = import_module("4_hazard_priority")

model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "models"
    / "nayan_yolov8n_best.pt"
)
if not model_path.is_file():
    raise SystemExit(
        f"ERROR: Trained YOLO weights not found: {model_path}\n"
        "Retrieve nayan_yolov8n_best.pt from the model-training-quantization branch before starting the live demo."
    )

model = YOLO(str(model_path))


CONFIDENCE_THRESHOLD = 0.4
NO_DETECTION_FRAME_LIMIT = 15
ALERT_COOLDOWN_SECONDS = 2.5
FPS = 30

previous_centers = {}
previous_bboxes = {}
no_detection_streak = 0
last_alert_time = 0


def speak(text):
    print(f"[ALERT] {text}")
    engine = pyttsx3.init()
    engine.setProperty('rate', 165)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("NAYAN live demo running. Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # Walking corridor: central 40% of frame
    corridor_left = int(frame_w * 0.30)
    corridor_right = int(frame_w * 0.70)
    cv2.rectangle(
        frame,
        (corridor_left, 0),
        (corridor_right, frame_h),
        (255, 255, 0),
        2
    )
    cv2.putText(
        frame,
        "WALKING CORRIDOR",
        (corridor_left + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=CONFIDENCE_THRESHOLD,
    )[0]

    detections = []
    track_ids = results.boxes.id
    for box_index, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        track_id = int(track_ids[box_index]) if track_ids is not None else None
        detections.append({
            "class_name": class_name,
            "bbox": [x1, y1, w, h],
            "confidence": confidence,
            "track_id": track_id,
        })
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        track_label = f" ID:{track_id}" if track_id is not None else ""
        cv2.putText(frame, f"{class_name} {confidence:.2f}{track_label}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    active_track_ids = {
        det["track_id"] for det in detections if det["track_id"] is not None
    }
    for previous_track_id in tuple(previous_centers):
        if previous_track_id not in active_track_ids:
            previous_centers.pop(previous_track_id, None)
            previous_bboxes.pop(previous_track_id, None)

    top_hazard = hazard_priority.get_top_hazard(
        detections,
        frame_w,
        frame_h,
        previous_centers,
        previous_bboxes,
        CONFIDENCE_THRESHOLD,
        FPS
    )

    now = time.time()
    if top_hazard is None:
        no_detection_streak += 1
        if no_detection_streak >= NO_DETECTION_FRAME_LIMIT and (now - last_alert_time) > ALERT_COOLDOWN_SECONDS:
            speak("Camera view unclear, please check camera position")
            last_alert_time = now
    else:
        no_detection_streak = 0
        if (now - last_alert_time) > ALERT_COOLDOWN_SECONDS:
            speak(f"{top_hazard['class_name']} ahead")
            last_alert_time = now

        ttc_value = top_hazard["ttc_seconds"]
        if ttc_value is None:
            ttc_text = "TTC: N/A"
        else:
            ttc_text = f"TTC: {ttc_value:.2f}s"

        info_text = (
            f"Hazard: {top_hazard['score']:.2f} | "
            f"Path: {top_hazard['path_relevance']:.2f} | "
            f"{ttc_text}"
        )
        cv2.putText(
            frame,
            info_text,
            (10, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

    # Update previous object information for the next frame
    for det in detections:
        track_id = det["track_id"]
        if track_id is None:
            continue

        class_name = det["class_name"]
        x, y, w, h = det["bbox"]
        current_center = (x + w / 2, y + h / 2)
        previous_centers[track_id] = current_center
        previous_bboxes[track_id] = det["bbox"]

    cv2.imshow("NAYAN - Live Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
