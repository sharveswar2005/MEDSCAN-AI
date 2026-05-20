import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):

    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
   
    for param in model.parameters():
        param.requires_grad = False
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
