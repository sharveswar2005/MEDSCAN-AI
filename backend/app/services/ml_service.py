import torch
from torchvision import transforms
from PIL import Image
import io
import cv2
import numpy as np

from backend.ml.training.model import get_model
from backend.app.core.config import settings

class MLService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model()
        self.model.load_state_dict(
            torch.load(settings.MODEL_WEIGHTS_PATH, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.class_names = ["NORMAL", "PNEUMONIA"]

        # Setup GradCAM
        self.target_layer = self.model.layer4[-1].conv2
        self.activations = None

        def forward_hook(module, input, output):
            output.requires_grad_(True)
            self.activations = output
            self.activations.retain_grad()

        self.target_layer.register_forward_hook(forward_hook)

    def predict(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return {
            "prediction": self.class_names[predicted.item()],
            "confidence": round(confidence.item() * 100, 2)
        }

    def generate_gradcam(self, image_bytes: bytes) -> bytes:
        self.model.eval()
        self.activations = None

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad_(True)

        self.model.zero_grad()
        output = self.model(input_tensor)
        pred_class = torch.argmax(output, dim=1)
        score = output[0, pred_class]
        score.backward()

        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("GradCAM failed: gradients not captured")

        gradients = self.activations.grad
        pooled_gradients = torch.mean(gradients, dim=(0, 2, 3))
        activation_map = self.activations.squeeze(0)

        for i in range(activation_map.shape[0]):
            activation_map[i] *= pooled_gradients[i]

        heatmap = torch.mean(activation_map, dim=0)
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original_img = cv2.cvtColor(
            np.array(image.resize((224, 224))),
            cv2.COLOR_RGB2BGR
        )
        superimposed_img = cv2.addWeighted(
            original_img, 0.6, heatmap, 0.4, 0
        )
        _, buffer = cv2.imencode(".jpg", superimposed_img)
        return buffer.tobytes()

ml_service = MLService()
