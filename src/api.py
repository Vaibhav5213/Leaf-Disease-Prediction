import sys
import os
import uuid
from fastapi import FastAPI, File, UploadFile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from predict import predict_disease

app = FastAPI(title="Disease Prediction API")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    segmented_img, results = predict_disease(image_bytes)
    return {
    "inference_id": f"req_{uuid.uuid4().hex[:8]}",
    "filename": file.filename,
    "predictions": results,
    "status": "success"
}
