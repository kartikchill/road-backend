# app.py
import uuid
import os
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from inference import FusionEngine  # updated inference.py

# ----------------------------
# Config
# ----------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Road Infra Analysis API")
engine = FusionEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static route for frontend
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ----------------------------
# Helpers
# ----------------------------
def safe_float(x):
    try:
        return float(x)
    except:
        return None


# ----------------------------
# VIDEO ANALYSIS
# ----------------------------
@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    skip_frames: int = Query(2, ge=1),
    imgsz: int = Query(640, ge=1),
    conf: float = Query(0.35, ge=0.0, le=1.0),
    max_frames: Optional[int] = Query(None, ge=1),
    save_annotated: bool = Query(False),
):
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}.mp4"
    input_path = os.path.join(OUTPUT_DIR, input_filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    out_video_rel = f"{file_id}_annotated.mp4" if save_annotated else None
    out_video_path = os.path.join(OUTPUT_DIR, out_video_rel) if save_annotated else None

    try:
        df, summary = engine.analyze_video(
            video_path=input_path,
            skip_frames=skip_frames,
            imgsz=imgsz,
            conf=conf,
            max_frames=max_frames,
            save_annotated=out_video_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    normalized_summary = {k: safe_float(v) for k, v in summary.items()}

    return {
        "summary": normalized_summary,
        "output_video": f"/outputs/{out_video_rel}" if out_video_rel else None,
        "input_path": f"/outputs/{input_filename}",
        "frames": len(df) if df is not None else 0,
    }


# ----------------------------
# SNAPSHOT API
# ----------------------------
@app.post("/snapshot")
async def snapshot(video_path: str = Query(...)):
    import cv2
    import uuid

    if video_path.startswith("/outputs/"):
        local_path = os.path.join(OUTPUT_DIR, video_path.split("/outputs/", 1)[1])
    else:
        local_path = video_path

    if not os.path.isfile(local_path):
        raise HTTPException(status_code=400, detail=f"Video not found: {local_path}")

    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Video has no frames.")

    frame_points = [int(total * 0.2), int(total * 0.5), int(total * 0.8)]
    out_paths = []

    for f in frame_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        name = f"sample_{uuid.uuid4().hex}.jpg"
        path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(path, frame)
        out_paths.append(f"/outputs/{name}")

    cap.release()
    return {"samples": out_paths}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)