import cv2
import time
import pyttsx3
from ultralytics import YOLO
from importlib import import_module

hazard_priority = import_module("4_hazard_priority")

model = YOLO("runs/detect/nayan_hazard_detector-2/weights/best.pt")


CONFIDENCE_THRESHOLD = 0.4
NO_DETECTION_FRAME_LIMIT = 15
ALERT_COOLDOWN_SECONDS = 2.5

previous_centers = {}
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
    results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x1, y2 - y1
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        detections.append({
            "class_name": class_name,
            "bbox": [x1, y1, w, h],
            "confidence": confidence,
        })
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    top_hazard = hazard_priority.get_top_hazard(
        detections, frame_w, frame_h, previous_centers, CONFIDENCE_THRESHOLD
    )

    now = time.time()
    if top_hazard is None:
        no_detection_streak += 1
        if no_detection_streak >= NO_DETECTION_FRAME_LIMIT and (now - last_alert_time) > ALERT_COOLDOWN_SECONDS:
            speak("Camera view unclear, please check camera position")
            last_alert_time = now
    else:
        no_detection_streak = 0
        previous_centers[top_hazard["class_name"]] = top_hazard["center"]
        if (now - last_alert_time) > ALERT_COOLDOWN_SECONDS:
            speak(f"{top_hazard['class_name']} ahead")
            last_alert_time = now

    cv2.imshow("NAYAN - Live Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
