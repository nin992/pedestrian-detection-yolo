"""
app.py
A small web API that runs the trained pedestrian detector on uploaded photos.

Endpoints:
    GET  /               -> check the API is running
    POST /detect         -> upload a photo, get JSON: people count, boxes, confidences
    POST /detect/image   -> upload a photo, get the photo back with boxes drawn on it

Run (from the project folder):
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs in a browser to try it.

Use a different model file (e.g. the INT8 OpenVINO version):
    MODEL_PATH=models/best_int8_openvino_model uvicorn app:app
"""
import io
import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

app = FastAPI(title="Pedestrian Detection API")
model = YOLO(MODEL_PATH, task="detect")     # loaded once, when the server starts


def read_image(data: bytes) -> Image.Image:
    """Turn uploaded bytes into an RGB image, or raise a clear 400 error."""
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")


def detect(img: Image.Image, conf: float):
    """Run the model on one image. Returns the raw result and a JSON-friendly summary."""
    start = time.perf_counter()
    result = model.predict(img, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - start) * 1000

    detections = []
    for box, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        detections.append({
            "box": [round(v) for v in box],          # x_min, y_min, x_max, y_max in pixels
            "confidence": round(score, 3),
        })

    summary = {
        "people_count": len(detections),
        "detections": detections,
        "image_size": {"width": img.width, "height": img.height},
        "confidence_threshold": conf,
        "inference_ms": round(ms, 1),
        "model": MODEL_PATH,
    }
    return result, summary


@app.get("/")
def home():
    return {"status": "ok", "model": MODEL_PATH, "try_it": "/docs"}


@app.post("/detect")
async def detect_json(file: UploadFile = File(...), conf: float = 0.25):
    img = read_image(await file.read())
    _, summary = detect(img, conf)
    return summary


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...), conf: float = 0.25):
    img = read_image(await file.read())
    result, summary = detect(img, conf)

    drawn = Image.fromarray(result.plot()[..., ::-1])   # plot() gives BGR; flip to RGB
    buffer = io.BytesIO()
    drawn.save(buffer, format="JPEG")
    return Response(content=buffer.getvalue(), media_type="image/jpeg",
                    headers={"X-People-Count": str(summary["people_count"])})
