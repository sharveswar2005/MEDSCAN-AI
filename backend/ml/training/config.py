import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001

DATA_DIR = "../data/chest_xray"
MODEL_SAVE_PATH = "../models/pneumonia_resnet18.pth"
