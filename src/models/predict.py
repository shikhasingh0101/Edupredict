"""Model loading utilities for the EduPredict API."""

from functools import lru_cache
from pathlib import Path

import joblib


MODEL_PATH = Path("models/edupredict_final_pipeline.joblib")


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained EduPredict pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)
