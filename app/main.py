from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from .model import DiseaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("CLASSIFIER_MODEL", ROOT / "PVDS_deploy.pt"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8s-worldv2.pt")
LEAF_CONFIDENCE = float(os.getenv("LEAF_CONFIDENCE", "0.05"))
OTHER_OBJECT_CONFIDENCE = float(os.getenv("OTHER_OBJECT_CONFIDENCE", "0.35"))
PLANT_LABELS = ("leaf", "plant", "foliage", "vegetation", "crop", "flower")

app = FastAPI(title="PVDS Plant Disease API", version="1.0.0")
classifier = DiseaseModel(MODEL_PATH)
yolo = YOLO(YOLO_MODEL)
yolo.set_classes([
    "leaf", "plant leaf", "foliage", "vegetation", "crop", "flower", "plant",
    "person", "car", "truck", "animal", "building", "phone", "laptop",
])


@app.get("/", response_class=HTMLResponse)
def demo() -> str:
        return """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>PVDS Plant Disease Demo</title>
            <style>
                :root { color-scheme: light; font-family: Georgia, serif; background: #f3f1e9; color: #18352d; }
                body { max-width: 760px; margin: 0 auto; padding: 48px 20px; }
                main { background: #fffdf7; border: 1px solid #d8d3c4; padding: 32px; box-shadow: 8px 8px 0 #dce6d9; }
                h1 { margin-top: 0; font-size: clamp(2rem, 7vw, 4.5rem); line-height: .95; }
                p { font: 1rem/1.5 system-ui, sans-serif; }
                input { display: block; width: 100%; margin: 24px 0; font: 1rem system-ui; }
                button { background: #18352d; color: white; border: 0; padding: 13px 20px; font: 600 1rem system-ui; cursor: pointer; }
                button:disabled { opacity: .5; cursor: wait; }
                #preview { display: none; max-width: 100%; max-height: 360px; margin: 24px 0; border: 1px solid #d8d3c4; }
                #result { white-space: pre-wrap; margin-top: 24px; font: .95rem/1.5 ui-monospace, monospace; }
            </style>
        </head>
        <body>
            <main>
                <p>PVDS / LOCAL MODEL DEMO</p>
                <h1>Check a plant leaf.</h1>
                <p>Upload a clear leaf image. The detector first checks that the image contains a leaf and rejects unrelated objects before classification.</p>
                <form id="form">
                    <input id="image" name="image" type="file" accept="image/*" required>
                    <img id="preview" alt="Selected leaf preview">
                    <button id="submit" type="submit">Classify image</button>
                </form>
                <div id="result" aria-live="polite"></div>
            </main>
            <script>
                const form = document.querySelector('#form');
                const input = document.querySelector('#image');
                const preview = document.querySelector('#preview');
                const result = document.querySelector('#result');
                input.addEventListener('change', () => {
                    const file = input.files[0];
                    if (file) { preview.src = URL.createObjectURL(file); preview.style.display = 'block'; }
                });
                form.addEventListener('submit', async (event) => {
                    event.preventDefault();
                    const button = document.querySelector('#submit');
                    button.disabled = true; result.textContent = 'Analyzing...';
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: new FormData(form) });
                        const data = await response.json();
                        result.textContent = response.ok
                            ? `Plant: ${data.plant}\nDisease: ${data.disease}\nConfidence: ${(data.confidence * 100).toFixed(1)}%\n\nDetections:\n${JSON.stringify(data.detections, null, 2)}`
                            : (data.detail?.message || data.detail || 'Prediction failed.');
                    } catch (error) { result.textContent = 'Could not reach the local API.'; }
                    finally { button.disabled = false; }
                });
            </script>
        </body>
        </html>
        """


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
        normalized_label = label.lower()
        if any(plant_label in normalized_label for plant_label in PLANT_LABELS) and confidence >= LEAF_CONFIDENCE:
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
