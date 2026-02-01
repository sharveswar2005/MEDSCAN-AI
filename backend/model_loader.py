import torch
import os
from training.model import get_model

# Get absolute path safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pneumonia_resnet18.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model()

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()
