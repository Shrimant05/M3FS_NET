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

## Deploy on Render

1. Push this folder, including `PVDS_deploy.pt`, to a private GitHub repository.
2. In Render, create a **Blueprint** from the repository. Render will use `render.yaml`.
3. Use at least a Starter instance because PyTorch and YOLO-World need more memory than the free tier reliably provides.
4. Send `multipart/form-data` to `/predict` using the field name `image`.

`YOLO_MODEL` may be changed to a custom Ultralytics leaf detector. For a production-grade leaf gate, use a detector trained with a `leaf` class; the open-vocabulary model is a useful baseline but cannot guarantee leaf detection on every crop.
