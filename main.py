"""
Solar Power Prediction API
FastAPI application with properly defined DTOs according to API contract.
"""

import os
import logging
from pathlib import Path
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
from model import train_weather_model, train_solar_model, load_weather_model, load_solar_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model storage
weather_model_bundle = None
solar_model_bundle = None

# Configuration from environment
STARTUP_MODE = os.getenv("STARTUP_MODE", "load_models")  # "train_now" or "load_models"
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
WEATHER_DATA_DIR = os.getenv("WEATHER_DATA_DIR", "../code/grid-weather")
WEATHER_HIST_FILE = os.getenv("WEATHER_HIST_FILE", "../code/data/23.530236_119.588339.csv")
WEATHER_PRED_FILE = os.getenv("WEATHER_PRED_FILE", "../code/data/weather-pred.csv")
SOLAR_DATA_DIR = os.getenv("SOLAR_DATA_DIR", "../code/data")

# Solar dataset files (matching reference code)
SOLAR_FILES = {
    "CT安集01": os.path.join(SOLAR_DATA_DIR, "旭東URE發電量v7-3 2.xlsx - CT安集01-data.csv"),
    "CT安集02": os.path.join(SOLAR_DATA_DIR, "旭東URE發電量v7-3 2.xlsx - CT安集02-data.csv"),
    "元晶": os.path.join(SOLAR_DATA_DIR, "旭東URE發電量v7-3 2.xlsx - 元晶-data.csv"),
    "EEC_land": os.path.join(SOLAR_DATA_DIR, "旭東URE發電量v7-3 2.xlsx - EEC-data.csv"),
}

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
    This runs ONCE when the server starts.
    Trains or loads ML models based on STARTUP_MODE environment variable.
    """
    global weather_model_bundle, solar_model_bundle
    
    logger.info("=" * 70)
    logger.info("🚀 FastAPI Server Starting Up")
    logger.info(f"Startup Mode: {STARTUP_MODE}")
    logger.info(f"Model Directory: {MODEL_DIR}")
    logger.info("=" * 70)
    
    try:
        if STARTUP_MODE == "train_now":
            logger.info("Training models from scratch...")
            
            # Train weather model
            logger.info("📊 Training weather forecasting model...")
            weather_result = train_weather_model(
                csv_path=WEATHER_HIST_FILE,
                output_dir=MODEL_DIR,
                win=30,  # 30-day window (exact match to reference)
                mode="multi"  # Multi-output model
            )
            logger.info(f"✅ Weather model trained: {weather_result['bundle_path']}")
            
            # Load the trained weather model
            weather_model_bundle = load_weather_model(weather_result['bundle_path'])
            
            # Train solar model
            logger.info("☀️ Training solar generation model...")
            solar_result = train_solar_model(
                solar_files=SOLAR_FILES,
                weather_hist_file=WEATHER_HIST_FILE,
                weather_pred_file=WEATHER_PRED_FILE,
                output_dir=MODEL_DIR,
                test_months=6,
                valid_months=1
            )
            logger.info(f"✅ Solar model trained: {solar_result['bundle_path']}")
            
            # Load the trained solar model
            solar_model_bundle = load_solar_model(solar_result['bundle_path'])
            
        elif STARTUP_MODE == "load_models":
            logger.info("Loading pre-trained models...")
            
            # Load weather model
            weather_bundle_path = Path(MODEL_DIR) / "weather_model_bundle.pkl"
            if weather_bundle_path.exists():
                weather_model_bundle = load_weather_model(str(weather_bundle_path))
                logger.info(f"✅ Weather model loaded from {weather_bundle_path}")
            else:
                logger.warning(f"⚠️ Weather model not found at {weather_bundle_path}")
            
            # Load solar model
            solar_bundle_path = Path(MODEL_DIR) / "solar_model_bundle.pkl"
            if solar_bundle_path.exists():
                solar_model_bundle = load_solar_model(str(solar_bundle_path))
                logger.info(f"✅ Solar model loaded from {solar_bundle_path}")
            else:
                logger.warning(f"⚠️ Solar model not found at {solar_bundle_path}")
        
        else:
            logger.warning(f"Unknown STARTUP_MODE: {STARTUP_MODE}. Skipping model setup.")
        
        logger.info("=" * 70)
        logger.info("✅ Startup Complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """
    This runs when the server shuts down
    Perfect for cleanup: closing connections, saving state, etc.
    """
    print("👋 FastAPI server shutting down...")