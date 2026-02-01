import torch
import cv2
import numpy as np
from PIL import Image
import io

from backend.model_loader import model, DEVICE
from backend.inference import transform


# Target last conv layer
target_layer = model.layer4[-1].conv2

activations = None


def forward_hook(module, input, output):
    global activations

    # Force grad tracking
    output.requires_grad_(True)

    activations = output
    activations.retain_grad()


# Register hook ONCE
target_layer.register_forward_hook(forward_hook)


def generate_gradcam(image_bytes):

    global activations

    model.eval()
    activations = None

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    model.zero_grad()

    output = model(input_tensor)

    pred_class = torch.argmax(output, dim=1)

    score = output[0, pred_class]
    score.backward()

    if activations is None or activations.grad is None:
        raise RuntimeError("GradCAM failed: gradients not captured")

    gradients = activations.grad

    pooled_gradients = torch.mean(gradients, dim=(0, 2, 3))

    activation_map = activations.squeeze(0)

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
