# utils.py
import cv2
import numpy as np
from typing import Dict, Any

def brighten_image(img, brightness=20):
    """Slightly brighten image to help detections in dark footage."""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, int(brightness))
        v[v > 255] = 255
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    except Exception:
        return img

def draw_detections_on_frame(frame, res: Dict[str, Any]) -> np.ndarray:
    """
    Draw bounding boxes and text based on the output 'res' produced by FusionEngine.
    Expects res["per_model"][<key>]["boxes"] list of {"xyxy": [x1,y1,x2,y2], "conf":.., "label":..}
    and res["fused"] / res["score"] for overlay text.
    Returns annotated BGR frame.
    """
    h, w = frame.shape[:2]
    # draw boxes
    color_map = {
        "rdd": (0, 0, 255),
        "potholes": (0, 0, 255),
        "roughroad": (0,165,255),
        "signs": (0, 255, 0),
        "lanes": (255, 200, 0),
    }
    for key, pm in res.get("per_model", {}).items():
        if not pm.get("present"):
            continue
        boxes = pm.get("boxes", [])
        for box in boxes:
            xy = box.get("xyxy", [0,0,0,0])
            x1, y1, x2, y2 = map(int, xy)
            color = color_map.get(key, (180,180,180))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = box.get("label", "")[:18]
            conf = box.get("conf", 0.0)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, max(12, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

    # overlay fused metrics
    fused = res.get("fused", {})
    score = res.get("score", {}).get("RDS", None)
    lines = []
    if score is not None:
        lines.append(f"RDS: {score:.1f}")
    # keep a few fused numbers
    lines.append(f"Potholes: {fused.get('pothole_count', 0)}")
    lines.append(f"Rough: {fused.get('roughroad_count', 0)}")
    lines.append(f"Signs: {fused.get('sign_count', 0)}")
    lines.append(f"Lanes: {fused.get('lane_count', 0)}")

    y = 22
    for line in lines:
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2, cv2.LINE_AA)
        y += 24

    return frame

def save_annotated_image(frame, out_path: str):
    """Write annotated BGR frame to disk (jpeg)."""
    cv2.imwrite(out_path, frame)
