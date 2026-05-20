from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.api.endpoints import predict

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for MedScan AI Chest X-ray Classification",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Can be moved to config later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Route
@app.get("/health")
def health_check():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}

# Include routers
app.include_router(predict.router, prefix=settings.API_V1_STR, tags=["Predictions"])
