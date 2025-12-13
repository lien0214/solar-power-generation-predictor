"""
Solar Power Prediction API
FastAPI application with properly defined DTOs according to API contract.
"""

from fastapi import FastAPI, Query, HTTPException
from datetime import datetime, timedelta
from typing import Optional

from dto import (
    Location,
    DayPrediction,
    MonthPrediction,
    DayPredictionResponse,
    MonthPredictionResponse,
    YearPredictionResponse,
    ErrorResponse
)

# Create FastAPI app instance
app = FastAPI(
    title="Solar Power Generation Predictor",
    description="API for predicting solar power generation based on location and time range",
    version="1.0.0",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)


@app.get(
    "/predict/day",
    response_model=DayPredictionResponse,
    summary="Predict solar generation for a day range",
    description="Predict solar generation for a specific day or date range at a given location"
)
def predict_day(
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    startDate: str = Query(..., description="Start date (YYYY-MM-DD)", alias="startDate"),
    endDate: str = Query(..., description="End date (YYYY-MM-DD)", alias="endDate"),
    pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)")
) -> DayPredictionResponse:
    """
    Predict solar generation for a specific day or date range at a given location.
    
    Example: /predict/day?lon=119.588339&lat=23.530236&startDate=2025-01-01&endDate=2025-01-31&pmp=1000
    """
    try:
        # Validate date format
        start = datetime.strptime(startDate, "%Y-%m-%d")
        end = datetime.strptime(endDate, "%Y-%m-%d")
        
        if end < start:
            raise HTTPException(status_code=400, detail="endDate must be after or equal to startDate")
        
        # Generate mock predictions for each day in the range
        predictions = []
        current = start
        while current <= end:
            predictions.append(
                DayPrediction(
                    date=current.strftime("%Y-%m-%d"),
                    kwh=round(5.0 + (hash(current.strftime("%Y-%m-%d")) % 100) / 100, 2)  # Mock data
                )
            )
            current += timedelta(days=1)
        
        return DayPredictionResponse(
            location=Location(lat=lat, lon=lon),
            startDate=startDate,
            endDate=endDate,
            pmp=pmp,
            predictions=predictions
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}")


@app.get(
    "/predict/month",
    response_model=MonthPredictionResponse,
    summary="Predict solar generation for a month range",
    description="Predict solar generation for a specific month range at a given location"
)
def predict_month(
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    startDate: str = Query(..., description="Start month (YYYY-MM)", alias="startDate"),
    endDate: str = Query(..., description="End month (YYYY-MM)", alias="endDate"),
    pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)")
) -> MonthPredictionResponse:
    """
    Predict solar generation for a specific month range at a given location.
    
    Example: /predict/month?lon=119.588339&lat=23.530236&startDate=2025-01&endDate=2025-05&pmp=1000
    """
    try:
        # Validate date format
        start = datetime.strptime(startDate, "%Y-%m")
        end = datetime.strptime(endDate, "%Y-%m")
        
        if end < start:
            raise HTTPException(status_code=400, detail="endDate must be after or equal to startDate")
        
        # Generate mock predictions for each month in the range
        predictions = []
        current = start
        while current <= end:
            predictions.append(
                MonthPrediction(
                    date=current.strftime("%Y-%m"),
                    kwh=round(100.0 + (hash(current.strftime("%Y-%m")) % 200), 2)  # Mock data
                )
            )
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return MonthPredictionResponse(
            location=Location(lat=lat, lon=lon),
            month=start.month,
            year=start.year,
            pmp=pmp,
            predictions=predictions
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM: {str(e)}")


@app.get(
    "/predict/year",
    response_model=YearPredictionResponse,
    summary="Predict solar generation for a year",
    description="Predict solar generation for a specific year at a given location"
)
def predict_year(
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    year: int = Query(..., ge=2000, le=2100, description="Year (YYYY)"),
    pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)")
) -> YearPredictionResponse:
    """
    Predict solar generation for a specific year at a given location.
    
    Example: /predict/year?lon=119.588339&lat=23.530236&year=2025&pmp=1000
    """
    # Generate mock prediction for the entire year
    total_kwh = round(1200.0 + (hash(f"{year}{lat}{lon}") % 500), 2)  # Mock data
    
    return YearPredictionResponse(
        location=Location(lat=lat, lon=lon),
        year=year,
        pmp=pmp,
        kwh=total_kwh
    )


@app.on_event("startup")
async def startup_event():
    """
    This runs ONCE when the server starts
    Perfect for loading models, connecting to databases, etc.
    
    In later stages, we'll use this to:
    - Load configuration
    - Fetch weather data
    - Train or load ML models
    - Connect to Redis cache
    """
    print("🚀 FastAPI server starting up...")
    print("📝 This is where we'll load models and fetch data later")
    print("✅ Startup complete!")


@app.on_event("shutdown")
async def shutdown_event():
    """
    This runs when the server shuts down
    Perfect for cleanup: closing connections, saving state, etc.
    """
    print("👋 FastAPI server shutting down...")