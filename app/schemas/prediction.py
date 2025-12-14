"""
Pydantic schemas (DTOs) for Solar Power Prediction API.

These models define the data contracts for all API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List


# --- Base Models ---

class Location(BaseModel):
    """Geographic location coordinates."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


# --- Prediction Models ---

class DayPrediction(BaseModel):
    """Single day solar generation prediction."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    kwh: float = Field(..., ge=0, description="Predicted energy generation in kWh")


class MonthPrediction(BaseModel):
    """Single month solar generation prediction."""
    date: str = Field(..., description="Month in YYYY-MM format")
    kwh: float = Field(..., ge=0, description="Predicted energy generation in kWh")


# --- API Response Models ---

class DayPredictionResponse(BaseModel):
    """Response for day range prediction endpoint."""
    location: Location
    startDate: str = Field(..., description="Start date (YYYY-MM-DD)")
    endDate: str = Field(..., description="End date (YYYY-MM-DD)")
    pmp: float = Field(..., gt=0, description="Panel Maximum Power (W)")
    predictions: List[DayPrediction]

    class Config:
        json_schema_extra = {
            "example": {
                "location": {"lat": 23.530236, "lon": 119.588339},
                "startDate": "2025-01-01",
                "endDate": "2025-01-31",
                "pmp": 1000.0,
                "predictions": [
                    {"date": "2025-01-01", "kwh": 5.2},
                    {"date": "2025-01-02", "kwh": 5.5}
                ]
            }
        }


class MonthPredictionResponse(BaseModel):
    """Response for month range prediction endpoint."""
    location: Location
    startDate: str = Field(..., description="Start month (YYYY-MM)")
    endDate: str = Field(..., description="End month (YYYY-MM)")
    pmp: float = Field(..., gt=0, description="Panel Maximum Power (W)")
    predictions: List[MonthPrediction]

    class Config:
        json_schema_extra = {
            "example": {
                "location": {"lat": 23.530236, "lon": 119.588339},
                "startDate": "2025-01",
                "endDate": "2025-12",
                "pmp": 1000.0,
                "predictions": [
                    {"date": "2025-01", "kwh": 150.5},
                    {"date": "2025-02", "kwh": 145.2}
                ]
            }
        }


class YearPredictionResponse(BaseModel):
    """Response for year prediction endpoint."""
    location: Location
    year: int = Field(..., ge=2000, le=2100, description="Year")
    pmp: float = Field(..., gt=0, description="Panel Maximum Power (W)")
    kwh: float = Field(..., ge=0, description="Total predicted energy (kWh)")

    class Config:
        json_schema_extra = {
            "example": {
                "location": {"lat": 23.530236, "lon": 119.588339},
                "year": 2025,
                "pmp": 1000.0,
                "kwh": 1850.5
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error message")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid date format. Use YYYY-MM-DD"
            }
        }
