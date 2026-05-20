# API Documentation

The MedScan AI backend provides a RESTful API powered by FastAPI.

## Base URL
`http://localhost:8000/api/v1`

## Endpoints

### 1. Health Check
- **Endpoint**: `/health`
- **Method**: `GET`
- **Description**: Returns the operational status of the backend API.
- **Response**:
  ```json
  {
    "status": "healthy",
    "environment": "development"
  }
  ```

### 2. Predict Image
- **Endpoint**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Payload**: `file` (UploadFile)
- **Description**: Analyzes an uploaded Chest X-ray and returns the prediction (Normal/Pneumonia) and confidence score.
- **Response**:
  ```json
  {
    "prediction": "PNEUMONIA",
    "confidence": 98.45
  }
  ```

### 3. Predict with Heatmap (GradCAM)
- **Endpoint**: `/predict-with-heatmap`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Payload**: `file` (UploadFile)
- **Description**: Performs the prediction and also returns a base64 encoded string of the GradCAM heatmap overlaid on the original X-ray.
- **Response**:
  ```json
  {
    "prediction": "PNEUMONIA",
    "confidence": 98.45,
    "heatmap": "iVBORw0KGgoAAAANSUhEUgAA..."
  }
  ```
