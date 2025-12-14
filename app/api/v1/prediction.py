"""
API v1 Prediction Endpoints.
HTTP layer that handles requests and delegates to the prediction service.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from ...schemas import (
    DayPredictionResponse,
    MonthPredictionResponse,
    YearPredictionResponse,
    Location
)
from ...services import get_prediction_service

router = APIRouter()
prediction_service = get_prediction_service()


@router.get(
    "/day",
    response_model=DayPredictionResponse,
    summary="Predict solar generation for a day range",
    description="Predict solar generation for a specific day or date range at a given location"
)
async def predict_day(
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    startDate: str = Query(..., description="Start date (YYYY-MM-DD)", alias="startDate"),
    endDate: str = Query(..., description="End date (YYYY-MM-DD)", alias="endDate"),
    pmp: Optional[float] = Query(1000, gt=0, description="Panel Maximum Power (W)")
) -> DayPredictionResponse:
    """
    Predict solar generation for a date range.
    
    Example: /predict/day?lon=119.588339&lat=23.530236&startDate=2025-01-01&endDate=2025-01-31&pmp=1000
    """
    try:
        predictions = await prediction_service.predict_day_range(
            lon=lon,
            lat=lat,
            start_date=startDate,
            end_date=endDate,
            pmp=pmp
        )
        
        return DayPredictionResponse(
            location=Location(lat=lat, lon=lon),
            startDate=startDate,
            endDate=endDate,
            pmp=pmp,
            predictions=predictions
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get(
    "/month",
    response_model=MonthPredictionResponse,
    summary="Predict solar generation for a month range",
    description="Predict solar generation for a specific month range at a given location"
)
async def predict_month(
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    startDate: str = Query(..., description="Start month (YYYY-MM)", alias="startDate"),
    endDate: str = Query(..., description="End month (YYYY-MM)", alias="endDate"),
    pmp: Optional[float] = Query(1000, gt=0, description="Panel Maximum Power (W)")
) -> MonthPredictionResponse:
    """
    Predict solar generation for a month range.
    
    Example: /predict/month?lon=119.588339&lat=23.530236&startDate=2025-01&endDate=2025-12&pmp=1000
    """
    try:
        predictions = await prediction_service.predict_month_range(
            lon=lon,
            lat=lat,
            start_date=startDate,
            end_date=endDate,
            pmp=pmp
        )
        
        return MonthPredictionResponse(
            location=Location(lat=lat, lon=lon),
            startDate=startDate,
            endDate=endDate,
            pmp=pmp,
            predictions=predictions
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get(
    "/year",
    response_model=YearPredictionResponse,
    summary="Predict solar generation for a year",
    description="Predict total solar generation for a specific year at a given location"
)
async def predict_year(
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    year: int = Query(..., ge=2000, le=2100, description="Year"),
    pmp: Optional[float] = Query(1000, gt=0, description="Panel Maximum Power (W)")
) -> YearPredictionResponse:
    """
    Predict total solar generation for a year.
    
    Example: /predict/year?lon=119.588339&lat=23.530236&year=2025&pmp=1000
    """
    try:
        total_kwh = await prediction_service.predict_year(
            lon=lon,
            lat=lat,
            year=year,
            pmp=pmp
        )
        
        return YearPredictionResponse(
            location=Location(lat=lat, lon=lon),
            year=year,
            pmp=pmp,
            kwh=total_kwh
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
