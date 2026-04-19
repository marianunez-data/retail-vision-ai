"""
RetailVision AI — FastAPI endpoint for gap detection.

Routes:
    POST /detect     → Upload image, get gap detections
    GET  /health     → Health check
    GET  /model-info → Model metadata
"""

import yaml
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# --- Config ---
PROJECT_ROOT = Path(__file__).parent.parent.parent

with open(PROJECT_ROOT / "configs" / "base_config.yaml") as f:
    config = yaml.safe_load(f)

CONF_THRESHOLD = config["inference"]["confidence_threshold"]
IOU_THRESHOLD = config["inference"]["iou_threshold"]

# --- App ---
app = FastAPI(
    title="RetailVision AI",
    description="Retail shelf gap detection API powered by YOLOv11",
    version="0.1.0",
)

# --- Load model once at startup ---
MODEL = None


@app.on_event("startup")
def load_model():
    global MODEL
    model_path = PROJECT_ROOT / "models" / "yolo11n_baseline_v1" / "weights" / "best.pt"
    MODEL = YOLO(str(model_path))
    print(f"Model loaded from: {model_path}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": MODEL is not None}


@app.get("/model-info")
def model_info():
    return {
        "model": "YOLOv11n",
        "parameters": "2.6M",
        "task": "gap detection",
        "classes": ["gap"],
        "input_size": 640,
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "test_metrics": {
            "precision": 0.917,
            "recall": 0.900,
            "mAP50": 0.874,
        },
    }


@app.post("/detect")
async def detect_gaps(
    file: UploadFile = File(...),
    confidence: float = None,
    iou: float = None,
):
    """Upload a shelf image → get gap detections as JSON."""
    conf = confidence or CONF_THRESHOLD
    iou_val = iou or IOU_THRESHOLD

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        results = MODEL(tmp_path, conf=conf, iou=iou_val, verbose=False)
        boxes = results[0].boxes

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": "gap",
                "confidence": round(float(box.conf), 3),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                },
            })

        total_gaps = len(detections)
        avg_conf = (
            round(sum(d["confidence"] for d in detections) / total_gaps, 3)
            if total_gaps > 0
            else 0
        )

        return JSONResponse(content={
            "filename": file.filename,
            "threshold_used": conf,
            "detections": detections,
            "summary": {
                "total_gaps": total_gaps,
                "average_confidence": avg_conf,
                "alert_level": (
                    "CRITICAL" if total_gaps >= 5
                    else "WARNING" if total_gaps >= 2
                    else "OK"
                ),
            },
        })

    finally:
        Path(tmp_path).unlink(missing_ok=True)
