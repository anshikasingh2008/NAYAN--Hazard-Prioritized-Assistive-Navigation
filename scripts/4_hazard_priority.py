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


# ---------------------------------------------------------
# 1. PROXIMITY SCORE
# ---------------------------------------------------------

def proximity_score(bbox, frame_width, frame_height):
    x, y, w, h = bbox

    box_area = w * h
    frame_area = frame_width * frame_height

    return min((box_area / frame_area) * 4, 1.0)


# ---------------------------------------------------------
# 2. MOTION SCORE
# ---------------------------------------------------------

def motion_score(current_center, previous_center, frame_width):
    if previous_center is None:
        return 0.0

    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]

    distance = (dx ** 2 + dy ** 2) ** 0.5

    return min(distance / (frame_width * 0.1), 1.0)


# ---------------------------------------------------------
# 3. PATH RELEVANCE SCORE
# ---------------------------------------------------------

def path_relevance_score(bbox, frame_width):
    """
    Measures how strongly an object's bounding-box center
    falls inside the central walking corridor.

    Central 40% of the frame is considered the walking path.

    Returns:
        0.0 = outside / edge of corridor
        1.0 = exactly at the center of the corridor
    """

    x, y, w, h = bbox

    # Bounding-box center
    center_x = x + (w / 2)

    # Central 40% of frame
    corridor_left = frame_width * 0.30
    corridor_right = frame_width * 0.70

    # Object is outside walking corridor
    if center_x < corridor_left or center_x > corridor_right:
        return 0.0

    # Center of the walking corridor
    corridor_center = frame_width * 0.50

    # Half-width of the corridor
    corridor_half_width = frame_width * 0.20

    # Distance from corridor center
    distance = abs(center_x - corridor_center)

    # Higher when closer to corridor center
    score = 1.0 - (distance / corridor_half_width)

    return max(0.0, min(score, 1.0))


# ---------------------------------------------------------
# 4. TIME-TO-COLLISION SCORE
# ---------------------------------------------------------

def ttc_score(
    bbox,
    previous_bbox,
    frame_width,
    frame_height,
    fps=30,
    danger_area_ratio=0.15
):
    """
    Estimates approximate Time-To-Collision (TTC) using
    bounding-box area growth.

    Bigger bounding box over time means the object is
    apparently getting closer.

    Returns:
        ttc_score: 0 to 1
        ttc_seconds: estimated time until danger size
    """

    # No previous bounding box means we cannot estimate growth
    if previous_bbox is None:
        return 0.0, None

    x, y, w, h = bbox
    px, py, pw, ph = previous_bbox

    current_area = w * h
    previous_area = pw * ph

    # Change in bounding-box area per frame
    area_growth = current_area - previous_area

    # Object is not getting visually larger
    if area_growth <= 0:
        return 0.0, None

    frame_area = frame_width * frame_height

    # Define the danger size
    danger_area = frame_area * danger_area_ratio

    # Already at or above danger size
    if current_area >= danger_area:
        return 1.0, 0.0

    # Remaining area before danger size
    remaining_area = danger_area - current_area

    # Estimated frames until danger size
    ttc_frames = remaining_area / area_growth

    # Convert frames to seconds
    ttc_seconds = ttc_frames / fps

    # Convert TTC into urgency score
    if ttc_seconds <= 0.5:
        score = 1.0

    elif ttc_seconds <= 1.0:
        score = 0.9

    elif ttc_seconds <= 2.0:
        score = 0.7

    elif ttc_seconds <= 3.0:
        score = 0.4

    elif ttc_seconds <= 5.0:
        score = 0.2

    else:
        score = 0.0

    return score, ttc_seconds


# ---------------------------------------------------------
# 5. FINAL SCORE
# ---------------------------------------------------------

def score_detection(
    class_name,
    bbox,
    frame_width,
    frame_height,
    previous_center=None,
    previous_bbox=None,
    fps=30
):
    """
    Combines:

    - Proximity
    - Motion
    - Object risk
    - Path relevance
    - Time-to-Collision
    """

    x, y, w, h = bbox

    # Current bounding-box center
    current_center = (
        x + w / 2,
        y + h / 2
    )

    # -----------------------------
    # Existing features
    # -----------------------------

    p_score = proximity_score(
        bbox,
        frame_width,
        frame_height
    )

    m_score = motion_score(
        current_center,
        previous_center,
        frame_width
    )

    risk_weight = RISK_WEIGHTS.get(
        class_name,
        DEFAULT_WEIGHT
    )

    # -----------------------------
    # NEW: Path relevance
    # -----------------------------

    path_score = path_relevance_score(
        bbox,
        frame_width
    )

    # -----------------------------
    # NEW: TTC
    # -----------------------------

    ttc, ttc_seconds = ttc_score(
        bbox,
        previous_bbox,
        frame_width,
        frame_height,
        fps
    )

    # -----------------------------
    # Combined score
    # -----------------------------

    base_score = (
        0.30 * p_score +
        0.15 * m_score +
        0.20 * risk_weight +
        0.15 * path_score +
        0.20 * ttc
    )

    # Objects directly in the walking corridor
    # receive additional priority.
    path_multiplier = 1.0 + (0.5 * path_score)

    final_score = base_score * path_multiplier

    # Keep final score between 0 and 1
    final_score = min(final_score, 1.0)

    return {
        "class_name": class_name,
        "score": final_score,

        "proximity": p_score,
        "motion": m_score,
        "risk_weight": risk_weight,

        "path_relevance": path_score,

        "ttc_score": ttc,
        "ttc_seconds": ttc_seconds,

        "center": current_center,
    }


# ---------------------------------------------------------
# 6. FIND TOP HAZARD
# ---------------------------------------------------------

def get_top_hazard(
    detections,
    frame_width,
    frame_height,
    previous_centers=None,
    previous_bboxes=None,
    confidence_threshold=0.4,
    fps=30
):
    if not detections:
        return None

    previous_centers = previous_centers or {}
    previous_bboxes = previous_bboxes or {}

    scored = []

    for det in detections:

        if det["confidence"] < confidence_threshold:
            continue

        class_name = det["class_name"]
        track_id = det.get("track_id")
        history_key = track_id if "track_id" in det else class_name

        prev_center = previous_centers.get(
            history_key
        )

        prev_bbox = previous_bboxes.get(
            history_key
        )

        result = score_detection(
            class_name,
            det["bbox"],
            frame_width,
            frame_height,
            prev_center,
            prev_bbox,
            fps
        )
        result["track_id"] = track_id

        scored.append(result)

    if not scored:
        return None

    return max(
        scored,
        key=lambda d: d["score"]
    )