import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataset import get_data_loaders
from model import get_model
from config import *


def evaluate():

    print("Using device:", DEVICE)

    _, _, test_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)

    model = get_model()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("Confusion Matrix:\n")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    plt.xticks([0, 1], ["NORMAL", "PNEUMONIA"])
    plt.yticks([0, 1], ["NORMAL", "PNEUMONIA"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.show()


if __name__ == "__main__":
    evaluate()
