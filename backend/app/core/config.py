import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MedScan AI"
    API_V1_STR: str = "/api/v1"
    
    # ML Model Configuration
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_WEIGHTS_PATH: str = os.path.join(BASE_DIR, "ml", "weights", "pneumonia_resnet18.pth")
    
    # Environment
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
