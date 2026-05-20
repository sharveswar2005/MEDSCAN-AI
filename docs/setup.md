# Local Setup Guide

This guide will walk you through setting up the MedScan AI project on your local machine without Docker.

## Backend Setup

1. **Navigate to the Backend Directory**:
   ```bash
   cd backend
   ```
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the FastAPI Server**:
   ```bash
   uvicorn app.main:app --reload
   ```

The backend will be available at `http://localhost:8000`.

## Frontend Setup

1. **Navigate to the Frontend Directory**:
   ```bash
   cd frontend
   ```
2. **Install Node Modules**:
   ```bash
   npm install
   ```
3. **Run the Development Server**:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`.
