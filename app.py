CURRENT_TASK_CANCELLED = False
# app.py
import uuid
import os
import shutil
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from inference import FusionEngine  # your updated inference.py

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

# Mount a static route so front end can fetch e.g. /outputs/<file>
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ----------------------------
# Helpers
# ----------------------------
def safe_float(x):
    """Ensure values (possibly numpy types) are converted to Python floats."""
    try:
        return float(x)
    except Exception:
        return None


# ----------------------------
# API Endpoints
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
    
    """
    Upload a video file and run the FusionEngine.analyze_video.
    Returns: summary, output_video (relative path) and frames processed.
    """

    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_filename = f"{file_id}.mp4"
    input_path = os.path.join(OUTPUT_DIR, input_filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")

    # Prepare annotated output filename (if requested)
    out_video_rel = f"{file_id}_annotated.mp4" if save_annotated else None
    out_video_path = os.path.join(OUTPUT_DIR, out_video_rel) if save_annotated else None

    # Run analysis (this may take time)
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
        # give a readable error to frontend
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    # Normalize summary numeric fields to Python floats for JSON serialization
    normalized_summary = {}
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            normalized_summary[k] = safe_float(v)
        else:
            normalized_summary[k] = v

    response = {
        "summary": normalized_summary,
        "output_video": f"/outputs/{out_video_rel}" if out_video_rel else None,
        "input_path": f"/outputs/{input_filename}",
        "frames": int(len(df)) if df is not None else 0,
    }
    return response

@app.post("/snapshot")
async def snapshot(video_path: str = Query(...)):
    """
    Create 3 sample snapshots at 20%, 50%, 80% of the video and return paths (relative to /outputs).
    Expects `video_path` to be a path that the server can open, e.g. '/outputs/<file>.mp4' or a full path.
    """
    import cv2
    import uuid

    # Accept both '/outputs/<name>' and local path
    if video_path.startswith("/outputs/"):
        local_path = os.path.join(OUTPUT_DIR, video_path.split("/outputs/", 1)[1])
    else:
        local_path = video_path

    if not os.path.isfile(local_path):
        raise HTTPException(status_code=400, detail=f"Video not found on server: {local_path}")

    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video for snapshots.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Video contains no frames.")

    chosen = [max(0, int(total * 0.2)), max(0, int(total * 0.5)), max(0, int(total * 0.8))]

    out_paths = []
    for fnum in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        img_name = f"sample_{uuid.uuid4().hex}.jpg"
        img_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(img_path, frame)
        out_paths.append(f"/outputs/{img_name}")

    cap.release()

    return {"samples": out_paths}


# Simple health endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}


# ----------------------------
# Run server if invoked directly
# ----------------------------
if __name__ == "__main__":
    # When you run `python app.py` this will start uvicorn programmatically
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)

#cancel button
@app.post("/cancel")
def cancel_processing():
    global CURRENT_TASK_CANCELLED
    CURRENT_TASK_CANCELLED = True
    return {"status": "cancelled"}
