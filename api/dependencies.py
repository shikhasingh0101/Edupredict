"""FastAPI dependencies."""

from functools import lru_cache
from src.models.predict import load_model


@lru_cache(maxsize=1)
def get_model():
    return load_model()
