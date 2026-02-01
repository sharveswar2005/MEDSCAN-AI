import torch


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def save_model(model, path):
    torch.save(model.state_dict(), path)
