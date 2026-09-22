"""EduPredict FastAPI application."""

from contextlib import asynccontextmanager
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.dependencies import get_model
from api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.monitoring.metrics import metrics


def performance_band(score: float) -> str:
    """Return a project-defined display band."""

    if score < 50:
        return "Needs Attention"

    if score < 75:
        return "Developing"

    return "Strong"


def create_features(request: PredictionRequest) -> pd.DataFrame:
    """Convert API input into the features expected by the trained model."""

    payload = request.model_dump()

    # Feature engineering must match the training pipeline.
    payload["Study_Attendance_Interaction"] = (
        payload["Hours_Studied"] * payload["Attendance"]
    )

    payload["Study_Sleep_Interaction"] = (
        payload["Hours_Studied"] * payload["Sleep_Hours"]
    )

    return pd.DataFrame([payload])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the API starts."""

    get_model()

    yield


app = FastAPI(
    title="EduPredict API",
    version="1.0.0",
    description="Student exam-score prediction API.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""

    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    """Return information about the production model."""

    return {
        "model_path": "models/edupredict_final_pipeline.joblib",
        "target": "Exam_Score",
        "model_type": "LinearRegression",
        "feature_engineering": [
            "Study_Attendance_Interaction",
            "Study_Sleep_Interaction",
        ],
        "status": "loaded",
    }


@app.get("/metrics")
def api_metrics():
    """Return API runtime metrics."""

    return metrics.snapshot()


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):
    """Predict a student's exam score."""

    start = perf_counter()

    try:
        model = get_model()

        # Build the exact feature representation expected by
        # the trained production pipeline.
        X = create_features(request)

        score = float(
            model.predict(X)[0]
        )

        metrics.observe(
            perf_counter() - start,
            prediction=True,
        )

        return {
            "predicted_score": round(score, 2),
            "performance_band": performance_band(score),
            "model_version": "EduPredict_Exam_Score_Model:1@champion",
        }

    except Exception as exc:

        metrics.observe(
            perf_counter() - start,
            error=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
