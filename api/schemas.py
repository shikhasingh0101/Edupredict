"""FastAPI request/response schemas."""

from typing import Any
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    Hours_Studied: int = Field(..., ge=0)
    Attendance: int = Field(..., ge=0, le=100)
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: int = Field(..., ge=0)
    Previous_Scores: int = Field(..., ge=0)
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: int = Field(..., ge=0)
    Family_Income: str
    Teacher_Quality: str | None = None
    School_Type: str
    Peer_Influence: str
    Physical_Activity: int = Field(..., ge=0)
    Learning_Disabilities: str
    Parental_Education_Level: str | None = None
    Distance_from_Home: str | None = None
    Gender: str


class PredictionResponse(BaseModel):
    predicted_score: float
    performance_band: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
