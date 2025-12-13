"""
DTOs package for Solar Power Prediction API
"""

from .responses import (
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
    "ErrorResponse"
]
