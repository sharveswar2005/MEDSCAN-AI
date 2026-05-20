# System Architecture

## Overview

The MedScan AI platform follows a modern microservices-oriented approach where the frontend, backend, and machine learning components are decoupled yet seamlessly integrated.

## Backend Architecture

The backend is built using **FastAPI** following the Controller-Service-Repository pattern (though the repository layer is omitted currently as we don't have a database). 

- **API Routes (`backend/app/api/`)**: Define the endpoints and handle HTTP requests/responses. They remain thin and delegate logic to services.
- **Services (`backend/app/services/`)**: Contain the core business logic. The `MLService` acts as a Singleton to ensure the PyTorch model is loaded into memory only once during the application's lifecycle, avoiding expensive reload times per request.
- **Core (`backend/app/core/`)**: Houses application-wide configurations and environment variables using Pydantic Settings.
- **Schemas (`backend/app/schemas/`)**: Defines strict Pydantic data models for validating request payloads and formatting responses.

## Machine Learning Pipeline

- **Training**: Training is handled offline using scripts located in `scripts/train.py`. The model is a PyTorch ResNet-18 fine-tuned on the Chest X-ray dataset.
- **Inference**: The model's state dictionary is saved in `backend/ml/weights/` and loaded dynamically into the `MLService`.
- **Explainability**: We use Grad-CAM (Gradient-weighted Class Activation Mapping) implemented natively in PyTorch and OpenCV. By hooking into the final convolutional layer of ResNet-18, we can extract gradients and feature maps to highlight regions of the X-ray that led to the prediction.

## Frontend Architecture

The frontend is built with **React** and **Vite**.
- **Components**: Reusable UI elements (Buttons, Upload areas).
- **Pages**: Top-level views connecting multiple components.
- **Services**: Abstracted Axios/Fetch calls to the backend API.
