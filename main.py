"""
Solar Power Prediction API
FastAPI application with properly defined DTOs according to API contract.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
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
from model import (
    train_weather_model,
    load_weather_model,
    train_solar_model_merged,
    load_solar_model_merged,
    train_solar_model_seperated,
    load_solar_model_seperated
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model storage
# We now store a dictionary of models to support multiple strategies.
weather_model_bundle = None
solar_models = {}
last_valid_date = None

# Constants
MISSING_VALUE = -999

# Configuration from environment
STARTUP_MODE = os.getenv("STARTUP_MODE", "load_models")  # "train_now" or "load_models"
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
WEATHER_DATA_DIR = os.getenv("WEATHER_DATA_DIR", "../code/grid-weather")
WEATHER_HIST_FILE = os.getenv("WEATHER_HIST_FILE", "../code/data/23.530236_119.588339.csv")
WEATHER_PRED_FILE = os.getenv("WEATHER_PRED_FILE", "../code/data/weather-pred.csv")
SOLAR_DATA_DIR = os.getenv("SOLAR_DATA_DIR", "app/data")

# SOLAR_FILES will be dynamically discovered from SOLAR_DATA_DIR during training.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This runs ONCE when the server starts and once when it shuts down.
    Trains or loads ML models based on STARTUP_MODE environment variable.
    """
    global weather_model_bundle, solar_models, last_valid_date
    
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

            # --- Dynamic Solar Data Discovery ---
            solar_data_path = Path(SOLAR_DATA_DIR)
            if not solar_data_path.is_dir():
                logger.warning(f"Solar data directory not found: {solar_data_path}. Skipping solar model training.")
                solar_files = {}
            else:
                solar_files = {p.stem: str(p) for p in solar_data_path.glob("*.csv")}

            if not solar_files:
                logger.warning(f"No solar data CSV files found in {solar_data_path}. Skipping solar model training.")
            else:
                logger.info(f"Found {len(solar_files)} solar data files for training: {list(solar_files.keys())}")
                # Train solar model (Merged)
                logger.info("☀️ Training solar generation model (Merged)...")
                solar_result = train_solar_model_merged(solar_files=solar_files, weather_hist_file=WEATHER_HIST_FILE, weather_pred_file=WEATHER_PRED_FILE, output_dir=MODEL_DIR, test_months=6, valid_months=1)
                logger.info(f"✅ Solar model (Merged) trained: {solar_result['bundle_path']}")
                solar_models["merged"] = load_solar_model_merged(solar_result['bundle_path'])

                # Train solar model (Seperated)
                logger.info("☀️ Training solar generation model (Seperated)...")
                solar_result_sep = train_solar_model_seperated(solar_files=solar_files, weather_hist_file=WEATHER_HIST_FILE, weather_pred_file=WEATHER_PRED_FILE, output_dir=MODEL_DIR, test_months=6, valid_months=1)
                logger.info(f"✅ Solar model (Seperated) trained: {solar_result_sep['bundle_path']}")
                solar_models["seperated"] = load_solar_model_seperated(solar_result_sep['bundle_path'])

        elif STARTUP_MODE == "load_models":
            logger.info("Loading pre-trained models...")
            
            # Load weather model
            weather_bundle_path = Path(MODEL_DIR) / "weather_model_bundle.pkl"
            if weather_bundle_path.exists():
                weather_model_bundle = load_weather_model(str(weather_bundle_path))
                logger.info(f"✅ Weather model loaded from {weather_bundle_path}")
            else:
                logger.warning(f"⚠️ Weather model not found at {weather_bundle_path}")
            
            # Load solar model (Merged)
            solar_bundle_path = Path(MODEL_DIR) / "solar_model_bundle.pkl"
            if solar_bundle_path.exists():
                solar_models["merged"] = load_solar_model_merged(str(solar_bundle_path))
                logger.info(f"✅ Solar model (Merged) loaded from {solar_bundle_path}")
            else:
                logger.warning(f"⚠️ Solar model (Merged) not found at {solar_bundle_path}")

            # Load solar model (Seperated)
            solar_sep_path = Path(MODEL_DIR) / "solar_model_bundle_seperated.pkl" # Assuming naming convention
            if solar_sep_path.exists():
                solar_models["seperated"] = load_solar_model_seperated(str(solar_sep_path))
                logger.info(f"✅ Solar model (Seperated) loaded from {solar_sep_path}")
            else:
                logger.warning(f"⚠️ Solar model (Seperated) not found at {solar_sep_path}")
        
        else:
            logger.warning(f"Unknown STARTUP_MODE: {STARTUP_MODE}. Skipping model setup.")
        
        # Initialize the last valid date from the data source
        _update_last_valid_date()
        logger.info(f"📅 Last valid weather date: {last_valid_date.date() if last_valid_date else 'Unknown'}")

        logger.info("=" * 70)
        logger.info("✅ Startup Complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}", exc_info=True)
        raise

    yield

    # Shutdown logic
    print("👋 FastAPI server shutting down...")

# Create FastAPI app instance
app = FastAPI(
    title="Solar Power Generation Predictor",
    description="API for predicting solar power generation based on location and time range",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)

def _update_last_valid_date():
    """
    Simulates checking the external API return value for the last valid date.
    It reads the historical file and finds the last row that does not contain
    the MISSING_VALUE (-999).
    """
    global last_valid_date
    try:
        if not os.path.exists(WEATHER_HIST_FILE):
            logger.warning(f"Weather history file not found at {WEATHER_HIST_FILE}. Using fallback.")
            last_valid_date = datetime.now() - timedelta(days=1)
            return

        df = pd.read_csv(WEATHER_HIST_FILE)
        # Construct Date column
        df['Date'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
        
        # Check for missing values (-999) in columns other than Date
        cols_to_check = [c for c in df.columns if c != 'Date']
        valid_mask = (df[cols_to_check] != MISSING_VALUE).all(axis=1)
        
        if valid_mask.any():
            last_valid_idx = df.index[valid_mask].max()
            last_valid_date = df.loc[last_valid_idx, 'Date'].to_pydatetime()
        else:
            last_valid_date = datetime.now() - timedelta(days=1)
            
    except Exception as e:
        logger.error(f"Failed to update last valid weather date: {e}")
        last_valid_date = datetime.now() - timedelta(days=1)

def _get_last_valid_weather_date() -> datetime:
    """
    Returns the cached last valid weather date.
    """
    global last_valid_date
    if last_valid_date is None:
        _update_last_valid_date()
    return last_valid_date

def _fetch_weather_from_api(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches historical weather data from an external API for the dates in input_df.
    """
    # TODO: Replace this with actual API call logic.
    # For now, we simulate fetching data by using the weather model to "fill" it,
    # or we could read from the CSV if the dates overlap with what we have on disk.
    
    logger.info(f"Fetching historical weather from API for {len(input_df)} rows...")
    
    # Mock implementation: 
    # In reality, you would do: response = requests.get(API_URL, params=...)
    # and fill input_df columns with the response data.
    
    # For this prototype, we'll use the model to simulate the API response so the code runs.
    if weather_model_bundle:
        weather_features = ['lat', 'lon', 'year', 'month', 'day']
        preds = weather_model_bundle.predict(input_df[weather_features])
        if isinstance(preds, (pd.DataFrame, pd.Series)):
             input_df = pd.concat([input_df, preds], axis=1)
    
    return input_df

def _predict_rolling_weather(input_df: pd.DataFrame, last_valid_date: datetime) -> pd.DataFrame:
    """
    Predicts future weather using a rolling window approach.
    """
    if weather_model_bundle is None:
        return input_df

    logger.info(f"Running rolling weather prediction for {len(input_df)} rows...")
    
    # 1. We need context (the past 30 days) to start the rolling prediction.
    # In a real implementation, we would fetch the 30 days ending at 'last_valid_date'
    # from our historical CSV or API to initialize the window.
    
    # For this implementation, we will perform a simplified rolling prediction 
    # where we predict one step, potentially use that output for the next (if the model supports it),
    # or just use the time features if the model is non-autoregressive.
    
    # Assuming weather_model_bundle.predict handles the logic based on inputs:
    weather_features = ['lat', 'lon', 'year', 'month', 'day']
    
    # If the model is truly autoregressive (LSTM), we would loop:
    # window = initial_context
    # for row in input_df:
    #    pred = model.predict(window)
    #    update window with pred
    #    store pred
    
    # Since we are using the 'bundle' abstraction and don't see the internal predict code,
    # we will pass the dataframe. If the bundle is smart, it handles it. 
    # If it's a simple regressor, it predicts based on date features.
    
    try:
        preds = weather_model_bundle.predict(input_df[weather_features])
        
        # Merge predictions
        if isinstance(preds, (pd.DataFrame, pd.Series)):
            # Ensure indices align
            preds = preds.reset_index(drop=True)
            input_df = input_df.reset_index(drop=True)
            input_df = pd.concat([input_df, preds], axis=1)
            
    except Exception as e:
        logger.error(f"Rolling prediction failed: {e}")
        
    return input_df

def _predict_weather_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to predict weather features (Irradiance, Temp, etc.) using the loaded weather model.
    """
    if weather_model_bundle is None:
        logger.warning("⚠️ Weather model not loaded. Skipping weather feature generation.")
        return input_df

    # 1. Determine the cutoff date
    last_valid_date = _get_last_valid_weather_date()
    
    # 2. Split the request into Historical (API) and Future (Rolling)
    mask_hist = input_df['Date'] <= last_valid_date
    mask_future = input_df['Date'] > last_valid_date
    
    df_hist = input_df[mask_hist].copy()
    df_future = input_df[mask_future].copy()
    
    # 3. Process Historical
    if not df_hist.empty:
        df_hist = _fetch_weather_from_api(df_hist)
        
    # 4. Process Future
    if not df_future.empty:
        df_future = _predict_rolling_weather(df_future, last_valid_date)
        
    # 5. Combine results
    # Sort by date to ensure order is preserved
    result_df = pd.concat([df_hist, df_future]).sort_values('Date').reset_index(drop=True)
    
    return result_df


def _predict_solar_power(input_df: pd.DataFrame, selected_model, strategy: str) -> np.ndarray:
    """
    Helper to predict solar power using the selected model(s).
    Handles the 'separated' strategy by averaging outputs if selected_model is a dict.
    """
    # Filter columns: Keep PMP and Weather columns. Drop metadata (Lat/Lon/Date).
    # The user specified that solar models are ONLY trained on weather + pmp.
    cols_to_drop = ['Date', 'lat', 'lon', 'year', 'month', 'day', 'Month']
    existing_drop = [c for c in cols_to_drop if c in input_df.columns]
    X_solar = input_df.drop(columns=existing_drop)
    
    if strategy == "seperated" and isinstance(selected_model, dict):
        # Ensemble averaging for separated models:
        # Run prediction on ALL site models and take the mean.
        model_predictions = []
        for _, model in selected_model.items():
            pred = model.predict(X_solar)
            model_predictions.append(pred)
        
        if not model_predictions:
             raise ValueError("No separated models available to predict.")

        return np.mean(model_predictions, axis=0)
    else:
        # Merged strategy or single model object
        return selected_model.predict(X_solar)

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
    pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)"),
    strategy: str = Query("merged", enum=["merged", "seperated"], description="Model strategy to use")
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
        
        selected_model = solar_models.get(strategy)
        if selected_model is None:
            raise HTTPException(status_code=503, detail=f"Solar model for strategy '{strategy}' is not loaded.")

        # Create input for prediction
        date_range = pd.date_range(start=start, end=end)
        input_df = pd.DataFrame({
            'Date': date_range,
            'lat': lat,
            'lon': lon,
            'pmp': pmp
        })

        # Feature Engineering: Extract year, month, day for the model
        # 1. Feature Engineering: Basic Date Features
        input_df['year'] = input_df['Date'].dt.year
        input_df['month'] = input_df['Date'].dt.month
        input_df['day'] = input_df['Date'].dt.day

        # Select features expected by the model (exclude raw Date object)
        X = input_df[['lat', 'lon', 'pmp', 'year', 'month', 'day']]
        # 2. Predict Weather Features (The Missing Link)
        input_df = _predict_weather_features(input_df)

        # Predict using the loaded models
        try:
            predicted_kwh = _predict_solar_power(input_df, selected_model, strategy)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

        predictions = []
        for date, kwh in zip(date_range, predicted_kwh):
            predictions.append(DayPrediction(date=date.strftime("%Y-%m-%d"), kwh=round(float(kwh), 2)))
        
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
    pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)"),
    strategy: str = Query("merged", enum=["merged", "seperated"], description="Model strategy to use")
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
        
        selected_model = solar_models.get(strategy)
        if selected_model is None:
            raise HTTPException(status_code=503, detail=f"Solar model for strategy '{strategy}' is not loaded.")

        # Create daily range for the months to get accurate aggregation
        # (Simplification: predicting for 1st of each month or aggregating days)
        # Here we generate a daily range covering the full months
        # Calculate last day of end month
        next_month = end.replace(day=28) + timedelta(days=4)
        last_day_of_end_month = next_month - timedelta(days=next_month.day)
        
        date_range = pd.date_range(start=start, end=last_day_of_end_month)
        input_df = pd.DataFrame({'Date': date_range, 'lat': lat, 'lon': lon, 'pmp': pmp})
        
        # Feature Engineering
        # 1. Feature Engineering
        input_df['year'] = input_df['Date'].dt.year
        input_df['month'] = input_df['Date'].dt.month
        input_df['day'] = input_df['Date'].dt.day
        X = input_df[['lat', 'lon', 'pmp', 'year', 'month', 'day']]
        
        # 2. Predict Weather
        input_df = _predict_weather_features(input_df)

        try:
            daily_kwh = _predict_solar_power(input_df, selected_model, strategy)
            input_df['kwh'] = daily_kwh
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

        # Aggregate by month
        input_df['Month'] = input_df['Date'].dt.to_period('M')
        monthly_data = input_df.groupby('Month')['kwh'].sum().reset_index()

        predictions = []
        for _, row in monthly_data.iterrows():
            predictions.append(MonthPrediction(date=str(row['Month']), kwh=round(row['kwh'], 2)))
        
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
    pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)"),
    strategy: str = Query("merged", enum=["merged", "seperated"], description="Model strategy to use")
) -> YearPredictionResponse:
    """
    Predict solar generation for a specific year at a given location.
    
    Example: /predict/year?lon=119.588339&lat=23.530236&year=2025&pmp=1000
    """
    selected_model = solar_models.get(strategy)
    if selected_model is None:
        raise HTTPException(status_code=503, detail=f"Solar model for strategy '{strategy}' is not loaded.")

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    date_range = pd.date_range(start=start_date, end=end_date)
    input_df = pd.DataFrame({'Date': date_range, 'lat': lat, 'lon': lon, 'pmp': pmp})

    # Feature Engineering
    # 1. Feature Engineering
    input_df['year'] = input_df['Date'].dt.year
    input_df['month'] = input_df['Date'].dt.month
    input_df['day'] = input_df['Date'].dt.day
    X = input_df[['lat', 'lon', 'pmp', 'year', 'month', 'day']]
    
    # 2. Predict Weather
    input_df = _predict_weather_features(input_df)

    try:
        daily_kwh = _predict_solar_power(input_df, selected_model, strategy)
        total_kwh = round(float(daily_kwh.sum()), 2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
    
    return YearPredictionResponse(
        location=Location(lat=lat, lon=lon),
        year=year,
        pmp=pmp,
        kwh=total_kwh
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)