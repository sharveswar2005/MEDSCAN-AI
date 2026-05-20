from fastapi import APIRouter, File, UploadFile
import base64

from backend.app.services.ml_service import ml_service
from backend.app.schemas.predict import PredictionResponse, PredictionWithHeatmapResponse

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = ml_service.predict(image_bytes)
    return result

@router.post("/predict-with-heatmap", response_model=PredictionWithHeatmapResponse)
async def predict_with_heatmap(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # Run prediction
    result = ml_service.predict(image_bytes)
    
    # Generate heatmap
    heatmap_bytes = ml_service.generate_gradcam(image_bytes)
    heatmap_base64 = base64.b64encode(heatmap_bytes).decode("utf-8")
    
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "heatmap": heatmap_base64
    }
