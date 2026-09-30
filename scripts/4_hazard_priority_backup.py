RISK_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 1.0,
    "bus": 1.0,
    "truck": 1.0,
    "bicycle": 0.8,
    "person": 0.5,
    "stop sign": 0.3,
    "fire hydrant": 0.3,
    "bench": 0.2,
    "chair": 0.2,
    "traffic light": 0.2,
}

DEFAULT_WEIGHT = 0.3


def proximity_score(bbox, frame_width, frame_height):
    x, y, w, h = bbox
    box_area = w * h
    frame_area = frame_width * frame_height
    return min(box_area / frame_area * 4, 1.0)


def motion_score(current_center, previous_center, frame_width):
    if previous_center is None:
        return 0.0
    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    distance = (dx ** 2 + dy ** 2) ** 0.5
    return min(distance / (frame_width * 0.1), 1.0)


def score_detection(class_name, bbox, frame_width, frame_height, previous_center=None):
    x, y, w, h = bbox
    current_center = (x + w / 2, y + h / 2)

    p_score = proximity_score(bbox, frame_width, frame_height)
    m_score = motion_score(current_center, previous_center, frame_width)
    risk_weight = RISK_WEIGHTS.get(class_name, DEFAULT_WEIGHT)

    final_score = (0.5 * p_score) + (0.3 * m_score) + (0.2 * risk_weight)

    return {
        "class_name": class_name,
        "score": final_score,
        "proximity": p_score,
        "motion": m_score,
        "risk_weight": risk_weight,
        "center": current_center,
    }


def get_top_hazard(detections, frame_width, frame_height, previous_centers=None, confidence_threshold=0.4):
    if not detections:
        return None

    previous_centers = previous_centers or {}
    scored = []
    for det in detections:
        if det["confidence"] < confidence_threshold:
            continue
        prev_center = previous_centers.get(det["class_name"])
        result = score_detection(det["class_name"], det["bbox"], frame_width, frame_height, prev_center)
        scored.append(result)

    if not scored:
        return None

    return max(scored, key=lambda d: d["score"])
