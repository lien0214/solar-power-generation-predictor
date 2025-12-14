"""Pydantic schemas package."""

from .prediction import (
    Location,
    DayPrediction,
    MonthPrediction,
    DayPredictionResponse,
    MonthPredictionResponse,
    YearPredictionResponse,
    ErrorResponse
)

__all__ = [
    "Location",
    "DayPrediction",
    "MonthPrediction",
    "DayPredictionResponse",
    "MonthPredictionResponse",
    "YearPredictionResponse",
    "ErrorResponse",
]
