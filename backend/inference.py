# inference.py
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from typing import Optional, Tuple, Dict
import os
import requests

TASKS = {}

# ---------------------------
# Helper: draw bounding boxes
# ---------------------------
def draw_boxes(frame, results, color):
    if results is None:
        return frame

    boxes = getattr(results, "boxes", None)
    if boxes is None or boxes.data is None:
        return frame

    for b in boxes:
        xyxy = b.xyxy.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame


# ---------------------------
# Fusion Engine
# ---------------------------
class FusionEngine:
    def __init__(self):

        print("📁 Preparing model directory...")
        os.makedirs("weights", exist_ok=True)

        # GitHub Release URLs — SAFE ✔️
        self.URLS = {
            "cityseg":   "https://github.com/kartikchill/road-infra-models/releases/download/models/cityseg.pt",
            "lanes":     "https://github.com/kartikchill/road-infra-models/releases/download/models/lanes.pt",
            "potholes":  "https://github.com/kartikchill/road-infra-models/releases/download/models/potholes.pt",
            "roughroad": "https://github.com/kartikchill/road-infra-models/releases/download/models/roughroad.pt",
            "signs":     "https://github.com/kartikchill/road-infra-models/releases/download/models/signs.pt",
        }

        print("⬇️  Downloading YOLO models locally...")
        self.paths = {}

        for name, url in self.URLS.items():
            local_path = f"weights/{name}.pt"
            self.paths[name] = local_path

            if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000:
                print(f"⬇️  Downloading {name}.pt ...")
                self.download(url, local_path)
            else:
                print(f"✔️ {name}.pt already exists locally")

        print("🤖 Loading YOLO models from local disk...")

        self.model_cityseg   = YOLO(self.paths["cityseg"])
        self.model_lanes     = YOLO(self.paths["lanes"])
        self.model_potholes  = YOLO(self.paths["potholes"])
        self.model_roughroad = YOLO(self.paths["roughroad"])
        self.model_signs     = YOLO(self.paths["signs"])

        print("🔥 All models loaded successfully!")

    # -------------------------
    # Safe Downloader
    # -------------------------
    def download(self, url, path):
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError(f"Download failed: {url}")

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"✅ Saved → {path}")

    # =====================================================================
    # A) IMAGE ANALYSIS
    # =====================================================================
    def analyze_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Could not read image.")

        # Run models
        res_p = self.model_potholes(img)[0]
        res_r = self.model_roughroad(img)[0]   # FIXED
        res_l = self.model_lanes(img)[0]
        res_s = self.model_signs(img)[0]

        # Counts
        potholes = len(res_p.boxes or [])
        rough    = len(res_r.boxes or [])
        lanes    = len(res_l.boxes or [])
        signs    = len(res_s.boxes or [])

        # Annotate
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
    # B) VIDEO ANALYSIS
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
            writer = cv2.VideoWriter(
                save_annotated,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (W, H)
            )

        stats = []
        frame_id = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_id += 1
            if frame_id % skip_frames != 0:
                continue

            if max_frames and len(stats) >= max_frames:
                break

            # Run models
            res_p = self.model_potholes(frame, conf=conf)[0]
            res_r = self.model_roughroad(frame, conf=conf)[0]  # FIXED
            res_l = self.model_lanes(frame, conf=conf)[0]
            res_s = self.model_signs(frame, conf=conf)[0]

            potholes = len(res_p.boxes or [])
            rough    = len(res_r.boxes or [])
            lanes    = len(res_l.boxes or [])
            signs    = len(res_s.boxes or [])

            stats.append({
                "potholes": potholes,
                "rough": rough,
                "lanes": lanes,
                "signs": signs,
            })

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

        df = pd.DataFrame(stats) if len(stats) > 0 else pd.DataFrame([{
            "potholes":0,"rough":0,"lanes":0,"signs":0
        }])

        # Compute RDS
        pothole_avg = float(df["potholes"].mean())
        rough_avg   = float(df["rough"].mean())
        lane_avg    = float(df["lanes"].mean())
        sign_avg    = float(df["signs"].mean())

        raw = 1 / (1 + pothole_avg * 3 + rough_avg * 2)
        RDS = (raw ** 3) * 100 + lane_avg * 20 + sign_avg * 10
        RDS = float(max(0, min(100, RDS)))

        summary = {
            "avg_potholes": pothole_avg,
            "avg_rough": rough_avg,
            "avg_lanes": lane_avg,
            "avg_signs": sign_avg,
            "avg_RDS": RDS
        }

        return df, summary