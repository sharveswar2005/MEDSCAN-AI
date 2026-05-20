from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float

class PredictionWithHeatmapResponse(PredictionResponse):
    heatmap: str
