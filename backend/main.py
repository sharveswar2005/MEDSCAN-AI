from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.inference import predict_image
from backend.gradcam import generate_gradcam
import base64

app = FastAPI(title="MedScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "MedScan AI Backend Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    result = predict_image(image_bytes)

    return result


@app.post("/predict-with-heatmap")
async def predict_with_heatmap(file: UploadFile = File(...)):

    image_bytes = await file.read()

    # Run prediction
    result = predict_image(image_bytes)

    # Generate heatmap
    heatmap_bytes = generate_gradcam(image_bytes)

    heatmap_base64 = base64.b64encode(heatmap_bytes).decode("utf-8")

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "heatmap": heatmap_base64
    }
