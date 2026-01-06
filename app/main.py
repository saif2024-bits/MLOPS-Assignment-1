"""
FastAPI application for Heart Disease Prediction Model
Provides REST API endpoints for making predictions
"""

import logging
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.model_pipeline import HeartDiseasePredictor

# Import monitoring utilities
try:
    from app.monitoring import (
        PrometheusMiddleware,
        metrics_endpoint,
        set_health_status,
        set_model_load_time,
        track_batch_prediction,
        track_prediction,
    )

    MONITORING_ENABLED = True
except ImportError:
    MONITORING_ENABLED = False
    logging.warning("Prometheus monitoring not available - install prometheus-client")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/api.log", mode="a")],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus monitoring middleware
if MONITORING_ENABLED:
    app.add_middleware(PrometheusMiddleware)
    logger.info("Prometheus monitoring enabled")

# Global predictor instance
predictor = None


# Pydantic models for request/response validation
class PatientData(BaseModel):
    """Patient data for heart disease prediction"""

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: int = Field(
        ..., ge=80, le=200, description="Resting blood pressure (mm Hg)"
    )
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ...,
        ge=0,
        le=1,
        description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    )
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)"
    )
    oldpeak: float = Field(
        ..., ge=0, le=10, description="ST depression induced by exercise"
    )
    slope: int = Field(
        ..., ge=1, le=3, description="Slope of peak exercise ST segment (1-3)"
    )
    ca: int = Field(
        ...,
        ge=0,
        le=3,
        description="Number of major vessels colored by fluoroscopy (0-3)",
    )
    thal: int = Field(..., ge=3, le=7, description="Thalassemia (3, 6, 7)")

    class Config:
        schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""

    patients: List[PatientData] = Field(..., min_items=1, max_items=100)


class PredictionResponse(BaseModel):
    """Prediction response"""

    prediction: int = Field(
        ..., description="Prediction (0 = No Heart Disease, 1 = Heart Disease)"
    )
    diagnosis: str = Field(..., description="Human-readable diagnosis")
    confidence: Optional[float] = Field(None, description="Confidence score")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Class probabilities"
    )
    model_used: str = Field(..., description="Model name used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_performance: Optional[Dict[str, float]] = Field(
        None, description="Model performance metrics"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""

    predictions: List[PredictionResponse]
    total: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_name: Optional[str]
    timestamp: str


class ModelInfo(BaseModel):
    """Model information response"""

    model_name: str
    model_type: str
    features_required: List[str]
    feature_count: int
    performance: Optional[Dict[str, float]]


# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global predictor
    try:
        logger.info("Starting Heart Disease Prediction API...")
        start_time = time.time()

        # Initialize predictor
        model_dir = os.environ.get("MODEL_DIR", "models")
        model_name = os.environ.get("MODEL_NAME", "xgboost")

        predictor = HeartDiseasePredictor(model_dir=model_dir)
        predictor.load_models(model_name=model_name)

        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully: {model_name} (took {load_time:.2f}s)")
        logger.info("API is ready to serve predictions")

        # Track model load time in metrics
        if MONITORING_ENABLED:
            set_model_load_time(load_time)
            set_health_status(True)

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        if MONITORING_ENABLED:
            set_health_status(False)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Heart Disease Prediction API...")


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor and predictor.model else "unhealthy",
        "model_loaded": predictor.model is not None if predictor else False,
        "model_name": predictor.model_name if predictor and predictor.model else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information"""
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = predictor.get_feature_names()

    return {
        "model_name": predictor.model_name,
        "model_type": (
            "XGBoost" if "xgboost" in predictor.model_name.lower() else "ML Classifier"
        ),
        "features_required": features,
        "feature_count": len(features),
        "performance": predictor.model_metadata if predictor.model_metadata else None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """
    Make a single prediction

    Predicts whether a patient has heart disease based on their medical data.

    Returns:
    - prediction: 0 (No Heart Disease) or 1 (Heart Disease)
    - diagnosis: Human-readable diagnosis
    - confidence: Confidence score (0-1)
    - probabilities: Probability for each class
    - model_used: Name of the model
    - timestamp: When the prediction was made
    """
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to dict
        patient_dict = patient.dict()

        # Make prediction
        result = predictor.predict(patient_dict)

        # Track prediction in metrics
        if MONITORING_ENABLED:
            track_prediction(
                prediction=result["prediction"], confidence=result.get("confidence")
            )

        logger.info(
            f"Prediction made: {result['diagnosis']} (confidence: {result.get('confidence', 'N/A')})"
        )

        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions

    Predicts heart disease for multiple patients in a single request.
    Maximum 100 patients per request.
    """
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        patients_data = [p.dict() for p in request.patients]
        df = pd.DataFrame(patients_data)

        # Make batch predictions
        results = predictor.predict_batch(df)

        # Track batch prediction in metrics
        if MONITORING_ENABLED:
            track_batch_prediction(len(results))
            for result in results:
                track_prediction(
                    prediction=result["prediction"], confidence=result.get("confidence")
                )

        logger.info(f"Batch prediction completed for {len(results)} patients")

        return {
            "predictions": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/features", tags=["Model"])
async def get_features():
    """Get list of required features and their descriptions"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_info = predictor.get_feature_info()

    return {
        "features": feature_info,
        "count": len(feature_info),
        "timestamp": datetime.now().isoformat(),
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions"""
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint

    Returns metrics in Prometheus format for monitoring and alerting.
    Includes:
    - Request counts and durations
    - Prediction counts and confidence scores
    - Model load time
    - Health status
    - Active requests
    - Error counts
    """
    if MONITORING_ENABLED:
        return metrics_endpoint()
    else:
        raise HTTPException(
            status_code=503,
            detail="Monitoring not enabled. Install prometheus-client to enable metrics.",
        )


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
