"""
Data Transfer Objects (DTOs) for Solar Power Prediction API

These Pydantic models define the structure and validation rules
for all API requests and responses according to the API contract.
"""

from pydantic import BaseModel, Field
from typing import List


# --- Response Models ---

class Location(BaseModel):
    """Location coordinates"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class DayPrediction(BaseModel):
    """Single day prediction"""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    kwh: float = Field(..., description="Predicted energy generation in kWh")


class MonthPrediction(BaseModel):
    """Single month prediction"""
    date: str = Field(..., description="Month in YYYY-MM format")
    kwh: float = Field(..., description="Predicted energy generation in kWh")


class DayPredictionResponse(BaseModel):
    """Response for day range prediction endpoint"""
    location: Location
    startDate: str = Field(..., description="Start date in YYYY-MM-DD format")
    endDate: str = Field(..., description="End date in YYYY-MM-DD format")
    pmp: float = Field(..., description="Panel Maximum Power in Watts")
    predictions: List[DayPrediction]


class MonthPredictionResponse(BaseModel):
    """Response for month range prediction endpoint"""
    location: Location
    month: int = Field(..., ge=1, le=12, description="Month number (1-12)")
    year: int = Field(..., description="Year")
    pmp: float = Field(..., description="Panel Maximum Power in Watts")
    predictions: List[MonthPrediction]


class YearPredictionResponse(BaseModel):
    """Response for year prediction endpoint"""
    location: Location
    year: int = Field(..., description="Year")
    pmp: float = Field(..., description="Panel Maximum Power in Watts")
    kwh: float = Field(..., description="Total predicted energy generation in kWh for the year")


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str = Field(..., description="Error message")
