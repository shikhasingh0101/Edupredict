"""EduPredict FastAPI application."""

from contextlib import asynccontextmanager
from time import perf_counter
from fastapi import FastAPI, HTTPException
from api.dependencies import get_model
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.monitoring.metrics import metrics


def performance_band(score: float) -> str:
    # Project-defined display band; not a validated educational diagnosis.
    if score < 50:
        return "Needs Attention"
    if score < 75:
        return "Developing"
    return "Strong"


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    return {
        "model_path": "models/model.joblib",
        "target": "Exam_Score",
        "status": "loaded",
    }


@app.get("/metrics")
def api_metrics():
    return metrics.snapshot()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start = perf_counter()
    try:
        model = get_model()
        payload = request.model_dump()
        import pandas as pd
        X = pd.DataFrame([payload])
        score = float(model.predict(X)[0])
        metrics.observe(perf_counter() - start, prediction=True)
        return {
            "predicted_score": round(score, 2),
            "performance_band": performance_band(score),
            "model_version": "local",
        }
    except Exception as exc:
        metrics.observe(perf_counter() - start, error=True)
        raise HTTPException(status_code=500, detail=str(exc))
