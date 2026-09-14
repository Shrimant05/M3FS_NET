# PVDS plant disease API

This service validates an uploaded image with YOLO-World, rejects images containing detected non-leaf objects, and classifies a detected plant leaf with `PVDS_deploy.pt`.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API. The first start downloads the YOLO-World weights. Keep `PVDS_deploy.pt` in the project root.

For the browser demo, open `http://localhost:8000/`, choose a plant-leaf image, and click **Classify image**. The page shows the plant, disease, confidence, and detector results.

## Deploy on Vercel

1. Push this folder, including `PVDS_deploy.pt`, to a private GitHub repository.
2. In Vercel, select **Add New -> Project** and import the repository.
3. Leave the framework preset as **Other**. Vercel detects `api/index.py` and `vercel.json`.
4. Deploy with the default build settings.
5. Test the deployment at `https://your-project.vercel.app/docs`.

The endpoints are `/health` and `/predict`. Send `multipart/form-data` to `/predict` using the field name `image`.

Set these optional Vercel environment variables when needed:

- `YOLO_MODEL`: defaults to `yolov8s-worldv2.pt`.
- `CLASSIFIER_MODEL`: defaults to `/var/task/PVDS_deploy.pt`.
- `LEAF_CONFIDENCE`: defaults to `0.15`.
- `OTHER_OBJECT_CONFIDENCE`: defaults to `0.35`.

Important: PyTorch plus Ultralytics is a large dependency for a serverless function. This project is configured for the Hobby plan's 2048 MB memory limit. If Vercel rejects the deployment because the function exceeds its size limit, or if cold starts/timeouts are too slow, keep the frontend/API on Vercel and move inference to a container service such as Render, Railway, or Modal.

`YOLO_MODEL` may be changed to a custom Ultralytics leaf detector. For a production-grade leaf gate, use a detector trained with a `leaf` class; the open-vocabulary model is a useful baseline but cannot guarantee leaf detection on every crop.
