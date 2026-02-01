import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_data_loaders
from model import get_model
from config import *
from utils import calculate_accuracy, save_model


def train():

    print("Using device:", DEVICE)

    train_loader, val_loader, _, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)

    model = get_model()
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # ---- Training ----
        model.train()
        running_loss = 0
        running_acc = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels)

        train_loss = running_loss / len(train_loader)
        train_acc = running_acc / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_acc = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                val_acc += calculate_accuracy(outputs, labels)

        val_acc = val_acc / len(val_loader)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, MODEL_SAVE_PATH)
            print("Best model saved!")

    print("\nTraining Completed")


if __name__ == "__main__":
    train()
