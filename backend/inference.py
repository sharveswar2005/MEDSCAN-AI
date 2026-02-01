import torch
from torchvision import transforms
from PIL import Image
import io

from backend.model_loader import model, DEVICE

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)

        probs = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probs, 1)

    class_names = ["NORMAL", "PNEUMONIA"]

    return {
        "prediction": class_names[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }
