# inference.py
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from typing import Optional, Tuple, Dict

TASKS = {}   # cancellation map shared with app.py


# --------------------------------------------
# Helper: draw bounding boxes on frame
# --------------------------------------------
def draw_boxes(frame, results, color):
    """
    Draw YOLO result boxes on frame with given color.
    """
    if results is None:
        return frame

    try:
        boxes = results.boxes
    except:
        return frame

    if boxes is None or boxes.data is None:
        return frame

    for b in boxes:
        xyxy = b.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame


# --------------------------------------------
# Fusion Engine
# --------------------------------------------
class FusionEngine:
    def __init__(self):
        self.model_potholes = YOLO("https://pub-97b8b19d33494b5382b0b908fe2458b9.r2.dev/models/potholes.pt")
        self.model_potholes = YOLO("https://pub-97b8b19d33494b5382b0b908fe2458b9.r2.dev/models/roughroad.pt")
        self.model_potholes = YOLO("https://pub-97b8b19d33494b5382b0b908fe2458b9.r2.dev/models/cityseg.pt")
        self.model_potholes = YOLO("https://pub-97b8b19d33494b5382b0b908fe2458b9.r2.dev/models/lanes.pt")
        self.model_potholes = YOLO("https://pub-97b8b19d33494b5382b0b908fe2458b9.r2.dev/models/signs.pt")

    # =====================================================================
    # A) IMAGE ANALYSIS  (NEW — Working Image Detection)
    # =====================================================================
    def analyze_image(self, image_path: str):
        """
        Returns detection counts + annotated image path.
        """

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Could not read image.")

        # Run models
        res_p = self.model_potholes(img)[0]
        res_r = self.model_rough(img)[0]
        res_l = self.model_lanes(img)[0]
        res_s = self.model_signs(img)[0]

        # counts
        potholes = len(res_p.boxes) if res_p.boxes is not None else 0
        rough = len(res_r.boxes) if res_r.boxes is not None else 0
        lanes = len(res_l.boxes) if res_l.boxes is not None else 0
        signs = len(res_s.boxes) if res_s.boxes is not None else 0

        # Annotated image
        annotated = img.copy()
        annotated = draw_boxes(annotated, res_p, (0,0,255))
        annotated = draw_boxes(annotated, res_r, (0,255,0))
        annotated = draw_boxes(annotated, res_l, (255,0,0))
        annotated = draw_boxes(annotated, res_s, (0,255,255))

        out_path = image_path.replace(".jpg", "_annotated.jpg")
        cv2.imwrite(out_path, annotated)

        return {
            "potholes": potholes,
            "rough": rough,
            "lanes": lanes,
            "signs": signs,
            "annotated_path": out_path
        }

    # =====================================================================
    # B) VIDEO ANALYSIS — with annotated snapshots + cancellation support
    # =====================================================================
    def analyze_video(
        self,
        video_path: str,
        skip_frames: int = 2,
        imgsz: int = 640,
        conf: float = 0.35,
        max_frames: Optional[int] = None,
        save_annotated: Optional[str] = None,
        task_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_annotated:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(save_annotated, fourcc, fps, (W, H))

        frame_id = 0
        stats = []

        while True:

            # cancellation check
            if task_id and TASKS.get(task_id) is True:
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            if frame_id % skip_frames != 0:
                continue

            if max_frames and len(stats) >= max_frames:
                break

            # Safe YOLO inference
            res_p = self.model_potholes(frame, imgsz=imgsz, conf=conf)[0]
            res_r = self.model_rough(frame, imgsz=imgsz, conf=conf)[0]
            res_l = self.model_lanes(frame, imgsz=imgsz, conf=conf)[0]
            res_s = self.model_signs(frame, imgsz=imgsz, conf=conf)[0]

            potholes = len(res_p.boxes) if res_p.boxes is not None else 0
            rough = len(res_r.boxes) if res_r.boxes is not None else 0
            lanes = len(res_l.boxes) if res_l.boxes is not None else 0
            signs = len(res_s.boxes) if res_s.boxes is not None else 0

            stats.append({
                "potholes": potholes,
                "rough": rough,
                "lanes": lanes,
                "signs": signs,
            })

            # Write annotated frame
            if writer:
                annotated = frame.copy()
                annotated = draw_boxes(annotated, res_p, (0,0,255))
                annotated = draw_boxes(annotated, res_r, (0,255,0))
                annotated = draw_boxes(annotated, res_l, (255,0,0))
                annotated = draw_boxes(annotated, res_s, (0,255,255))
                writer.write(annotated)

        cap.release()
        if writer:
            writer.release()

        # avoid error when empty
        if len(stats) == 0:
            df = pd.DataFrame([{"potholes": 0, "rough": 0, "lanes": 0, "signs": 0}])
        else:
            df = pd.DataFrame(stats)

        # Averages
        pothole_avg = float(df["potholes"].mean())
        rough_avg = float(df["rough"].mean())
        lane_avg = float(df["lanes"].mean())
        sign_avg = float(df["signs"].mean())

        # Final formula (unchanged)
        raw = 1 / (1 + pothole_avg * 3 + rough_avg * 2)
        RDS = (raw ** 3) * 100
        RDS += lane_avg * 20
        RDS += sign_avg * 10
        RDS = float(max(0, min(100, RDS)))

        summary = {
            "avg_potholes": pothole_avg,
            "avg_rough": rough_avg,
            "avg_lanes": lane_avg,
            "avg_signs": sign_avg,
            "avg_RDS": RDS
        }

        return df, summary
