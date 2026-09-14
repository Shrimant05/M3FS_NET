from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from .model import DiseaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("CLASSIFIER_MODEL", ROOT / "PVDS_deploy.pt"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s-worldv2.pt")
LEAF_CONFIDENCE = float(os.getenv("LEAF_CONFIDENCE", "0.15"))
OTHER_OBJECT_CONFIDENCE = float(os.getenv("OTHER_OBJECT_CONFIDENCE", "0.35"))

app = FastAPI(title="PVDS Plant Disease API", version="1.0.0")
classifier = DiseaseModel(MODEL_PATH)
yolo = YOLO(YOLO_MODEL)
yolo.set_classes(["leaf", "plant leaf", "person", "car", "truck", "animal", "building", "phone", "laptop"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_PATH.name, "detector": YOLO_MODEL}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    try:
        picture = Image.open(BytesIO(await image.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    result = yolo.predict(picture, conf=LEAF_CONFIDENCE, verbose=False)[0]
    detections = []
    leaf_found = False
    invalid_objects = []
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        detections.append({"label": label, "confidence": round(confidence, 4)})
        if "leaf" in label.lower() and confidence >= LEAF_CONFIDENCE:
            leaf_found = True
        elif confidence >= OTHER_OBJECT_CONFIDENCE:
            invalid_objects.append(label)
    if invalid_objects:
        raise HTTPException(status_code=422, detail={"message": "Non-leaf object detected.", "objects": sorted(set(invalid_objects)), "detections": detections})
    if not leaf_found:
        raise HTTPException(status_code=422, detail={"message": "No plant leaf detected. Upload a clear leaf image.", "detections": detections})

    label, confidence = classifier.predict(picture)
    plant, disease = label.split("___", 1)
    return {"plant": plant, "disease": disease, "class": label, "confidence": round(confidence, 4), "detections": detections}
