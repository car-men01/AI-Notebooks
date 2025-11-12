# FastAPI app

from fastapi import FastAPI, File, UploadFile
from app.model import load_model
from app.predict import predict_image
from app.schema import PredictionResponse
import torch
from io import BytesIO

app = FastAPI(title="Plant Classifier API")

model = None
params = None


@app.on_event("startup")
async def startup_event():
    global model, params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, params = load_model(
        "model/efficientnet_b0_plants.pt",
        "model/preprocessing_params.joblib",
        device
    )


@app.get("/")
async def root():
    return {"message": "Plant Classifier API is running"}


@app.get("/classes")
async def get_classes():
    """Get list of all plant classes the model can recognize"""
    if params is None:
        return {"error": "Model not loaded"}

    return {
        "total_classes": len(params['classes']),
        "classes": sorted(params['classes'])
    }


@app.post("/predict")
async def classify_plant(file: UploadFile = File(...)):
    """Upload an image to get plant classification"""
    if model is None or params is None:
        return {"error": "Model not loaded"}

    # Read file contents
    contents = await file.read()
    from io import BytesIO
    image_bytes = BytesIO(contents)

    # Package model, params, and device together
    package = {
        'model': model,
        'params': params,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # Call predict_image with correct arguments
    result = predict_image(image_bytes, package)
    return result

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "classes_available": len(params['classes']) if params else 0
    }