"""
Solar Power Prediction API
FastAPI application with properly defined DTOs according to API contract.
"""

import os
import time, sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Request
from datetime import datetime, timedelta
from typing import Optional, Tuple
import math
import aiohttp
from io import StringIO

from app.dto import (
    Location,
    DayPrediction,
    MonthPrediction,
    DayPredictionResponse,
    MonthPredictionResponse,
    YearPredictionResponse,
    ErrorResponse
)
from app.model import (
    train_weather_model,
    load_weather_model,
    train_solar_model_merged,
    load_solar_model_merged,
    train_solar_model_seperated,
    load_solar_model_seperated
)
from app.utils import fetch_grid_weather

# Setup logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
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
STARTUP_MODE = os.getenv("STARTUP_MODE", "train_now")  # "train_now" or "load_models"
MODEL_DIR = os.getenv("MODEL_DIR", "./app/models")
WEATHER_DATA_DIR = os.getenv("WEATHER_DATA_DIR", "./app/data/weather-data")
SOLAR_DATA_DIR = os.getenv("SOLAR_DATA_DIR", "./app/data/solar-data")

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

            # Fetch weather data
            logger.info(f"🌍 Fetching weather grid data to {WEATHER_DATA_DIR}...")
            await fetch_grid_weather(output_dir=WEATHER_DATA_DIR)

            # Train weather model
            logger.info("📊 Training weather forecasting model...")
            weather_result = train_weather_model(
                data_dir=WEATHER_DATA_DIR,
                output_dir=MODEL_DIR,
                win=30,  # 30-day window (exact match to reference)
                mode="multi"  # Multi-output model
            )
            logger.info(f"✅ Weather model trained: {weather_result['bundle_path']}")
            weather_pred_file = weather_result['pred_path']

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
                
                # Determine the weather history file to use.
                # The solar trainers expect a single file, so we'll pick one from the weather data dir.
                weather_hist_files = list(Path(WEATHER_DATA_DIR).glob("*.csv"))
                if not weather_hist_files:
                    raise FileNotFoundError(f"No weather history files found in {WEATHER_DATA_DIR} for solar training.")
                weather_hist_file = str(weather_hist_files[0])
                logger.info(f"Using historical weather file for solar training: {weather_hist_file}")


                # Train solar model (Merged)
                logger.info("☀️ Training solar generation model (Merged)...")
                solar_result = train_solar_model_merged(solar_files=solar_files, weather_hist_file=weather_hist_file, weather_pred_file=weather_pred_file, output_dir=MODEL_DIR, test_months=6, valid_months=1)
                logger.info(f"✅ Solar model (Merged) trained: {solar_result['bundle_path']}")
                solar_models["merged"] = load_solar_model_merged(solar_result['bundle_path'])

                # Train solar model (Seperated)
                logger.info("☀️ Training solar generation model (Seperated)...")
                solar_result_sep = train_solar_model_seperated(solar_files=solar_files, weather_hist_file=weather_hist_file, weather_pred_file=weather_pred_file, output_dir=MODEL_DIR, test_months=6, valid_months=1)
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
            solar_bundle_path = Path(MODEL_DIR) / "solar_model_bundle_merged.pkl"
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
        weather_hist_files = list(Path(WEATHER_DATA_DIR).glob("*.csv"))
        if weather_hist_files:
            _update_last_valid_date(str(weather_hist_files[0]))
        else:
            logger.warning(f"No weather history files found in {WEATHER_DATA_DIR}, cannot determine last valid date.")

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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs every request with method, path, status code, and processing time."""
    logger.debug(f"Request received: {request.method} {request.url}")
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(f"📡 Request: {request.method} {request.url} - Response: {response.status_code} [{process_time:.2f}ms]")
    return response

def _shifted_sin_cos(doy: np.ndarray, offset: int, year_lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase-shifted sin/cos features for seasonality.
    """
    shifted = doy - offset
    angles = 2.0 * math.pi * ((shifted % year_lengths) / year_lengths)
    sinv = np.sin(angles)
    cosv = np.cos(angles)
    return sinv.astype("float32"), cosv.astype("float32")


async def _fetch_point_history(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical weather data for a single point and return a DataFrame.
    """
    params = {
        "start": start_date,
        "end": end_date,
        "latitude": lat,
        "longitude": lon,
        "community": "sb",
        "parameters": "T2M,T2M_MAX,TS,CLOUD_AMT_DAY,CLOUD_OD,ALLSKY_SFC_SW_DWN,RH2M,ALLSKY_SFC_SW_DIRH",
        "format": "csv",
        "header": "false",
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("https://power.larc.nasa.gov/api/temporal/daily/point", params=params) as response:
                response.raise_for_status()
                text = await response.text()
                
                # The response has a header, so we can use pandas to read it directly
                # We need to add the LAT and LON columns ourselves.
                df = pd.read_csv(StringIO(text), header=0)
                df['LAT'] = lat
                df['LON'] = lon
                df['Date'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
                return df
        except Exception as e:
            logger.error(f"Failed to fetch historical point data for ({lat}, {lon}): {e}")
            raise RuntimeError(f"Could not fetch seed data for prediction.")


def _update_last_valid_date(weather_hist_file: str):
    """
    Simulates checking the external API return value for the last valid date.
    It reads the historical file and finds the last row that does not contain
    the MISSING_VALUE (-999).
    """
    global last_valid_date
    logger.debug("Updating last valid weather date from data source...")
    try:
        if not os.path.exists(weather_hist_file):
            logger.warning(f"Weather history file not found at {weather_hist_file}. Using fallback.")
            last_valid_date = datetime.now() - timedelta(days=1)
            return

        df = pd.read_csv(weather_hist_file)
        # Construct Date column
        df['Date'] = pd.to_datetime(df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
        
        # Check for missing values (-999) in columns other than Date
        cols_to_check = [c for c in df.columns if c != 'Date']
        valid_mask = (df[cols_to_check] != MISSING_VALUE).all(axis=1)
        
        if valid_mask.any():
            last_valid_idx = df.index[valid_mask].max()
            last_valid_date = df.loc[last_valid_idx, 'Date'].to_pydatetime()
            logger.debug(f"Source data check complete. Last valid date is {last_valid_date.date()}")
        else:
            last_valid_date = datetime.now() - timedelta(days=1)
            logger.warning("No valid data found in source file, using fallback date.")
            
    except Exception as e:
        logger.error(f"Failed to update last valid weather date: {e}", exc_info=True)
        last_valid_date = datetime.now() - timedelta(days=1)

def _get_last_valid_weather_date() -> datetime:
    """
    Returns the cached last valid weather date.
    """
    global last_valid_date
    if last_valid_date is None:
        logger.warning("Last valid date not initialized at startup, updating now...")
        weather_hist_files = list(Path(WEATHER_DATA_DIR).glob("*.csv"))
        if weather_hist_files:
            _update_last_valid_date(str(weather_hist_files[0]))
        else:
            logger.warning(f"No weather history files found in {WEATHER_DATA_DIR}, cannot determine last valid date.")
    return last_valid_date

def _fetch_weather_from_api(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches historical weather data by reading it from the local CSV files.
    """
    logger.debug(f"☁️ [Weather] Fetching historical data for {len(input_df)} rows from CSV...")
    
    weather_hist_files = list(Path(WEATHER_DATA_DIR).glob("*.csv"))
    if not weather_hist_files:
        logger.warning(f"No weather history files found in {WEATHER_DATA_DIR}. Cannot fetch historical weather.")
        return input_df

    # In case there are multiple, we'll read them all.
    hist_df = pd.concat([pd.read_csv(f) for f in weather_hist_files], ignore_index=True)
    hist_df['Date'] = pd.to_datetime(hist_df[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'}))
    
    # These are the features the solar model expects
    weather_features = weather_model_bundle.get('features', [])
    if not weather_features:
        logger.warning("No features found in weather model bundle.")
        return input_df
        
    cols_to_merge = ['Date'] + weather_features
    
    # Ensure columns exist in historical data
    missing_cols = [c for c in cols_to_merge if c not in hist_df.columns]
    if missing_cols:
        # Don't fail, but fill with NaNs which might be handled later.
        for col in missing_cols:
            if col != 'Date':
                hist_df[col] = np.nan

    # Merge with input_df, keeping all rows from input_df (left merge)
    result_df = pd.merge(input_df, hist_df[cols_to_merge], on="Date", how="left")
    
    return result_df

async def _predict_rolling_weather(input_df: pd.DataFrame, last_valid_date: datetime) -> pd.DataFrame:
    """
    Predicts future weather using a rolling window approach by fetching exact historical data for the requested point.
    """
    if weather_model_bundle is None:
        raise RuntimeError("Weather model is not loaded.")

    win = weather_model_bundle['win']
    targets = weather_model_bundle['targets']
    hottest_offset = weather_model_bundle['hottest_offset']
    model = weather_model_bundle['model']
    
    request_lat = input_df['lat'].iloc[0]
    request_lon = input_df['lon'].iloc[0]

    # Define the seed date range
    seed_end_date = last_valid_date
    seed_start_date = seed_end_date - timedelta(days=win - 1)
    
    # Fetch the exact historical data for the point to use as a seed
    logger.info(f"Fetching seed data for ({request_lat}, {request_lon}) from {seed_start_date.date()} to {seed_end_date.date()}")
    seed_df = await _fetch_point_history(
        lat=request_lat,
        lon=request_lon,
        start_date=seed_start_date.strftime("%Y%m%d"),
        end_date=seed_end_date.strftime("%Y%m%d")
    )
    
    if len(seed_df) < win:
        raise RuntimeError(f"Could not fetch enough historical data to form a seed window of {win} days. Found {len(seed_df)} days.")

    # Prepare for rolling forecast
    window_feats = seed_df[targets].to_numpy(dtype="float32")
    
    predictions = []
    
    for _, row in input_df.iterrows():
        current_date = row['Date']
        
        year_len = 366 if current_date.is_leap_year else 365
        doy = current_date.timetuple().tm_yday
        sin_doy, cos_doy = _shifted_sin_cos(np.array([doy]), hottest_offset, np.array([year_len]))

        # Create input for the model
        flat_window = window_feats.flatten()
        extras = np.array([request_lon, request_lat, sin_doy[0], cos_doy[0]], dtype="float32")
        X_row = np.concatenate([flat_window, extras]).reshape(1, -1)

        # Predict
        y_pred = model.predict(X_row)[0]

        # Store prediction
        pred_row_dict = {target: pred for target, pred in zip(targets, y_pred)}
        predictions.append(pred_row_dict)

        # Update window for next prediction
        new_feat_row = y_pred.reshape(1, -1)
        window_feats = np.vstack([window_feats[1:], new_feat_row])

    pred_df = pd.DataFrame(predictions)
    
    # Combine with original input_df
    input_df = input_df.reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)
    result_df = pd.concat([input_df, pred_df], axis=1)

    return result_df

async def _predict_weather_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper to predict weather features (Irradiance, Temp, etc.) using the loaded weather model.
    """
    if weather_model_bundle is None:
        logger.warning("⚠️ Weather model not loaded. Skipping weather feature generation.")
        return input_df
    logger.debug(f"🌦️ [Pipeline] Generating weather features for {len(input_df)} rows...")

    # 1. Determine the cutoff date
    last_valid_date = _get_last_valid_weather_date()
    
    # 2. Split the request into Historical (API) and Future (Rolling)
    mask_hist = input_df['Date'] <= last_valid_date
    mask_future = input_df['Date'] > last_valid_date
    
    df_hist = input_df[mask_hist].copy()
    df_future = input_df[mask_future].copy()
    logger.debug(f"📅 [Pipeline] Date split: {len(df_hist)} historical rows, {len(df_future)} future rows (Cutoff: {last_valid_date.date()})")
    
    # 3. Process Historical
    if not df_hist.empty:
        df_hist = _fetch_weather_from_api(df_hist)
        
    # 4. Process Future
    if not df_future.empty:
        df_future = await _predict_rolling_weather(df_future, last_valid_date)
        
    # 5. Combine results
    # Sort by date to ensure order is preserved
    result_df = pd.concat([df_hist, df_future]).sort_values('Date').reset_index(drop=True)
    
    return result_df


def _predict_solar_power(input_df: pd.DataFrame, selected_model, strategy: str) -> np.ndarray:
    """
    Helper to predict solar power using the selected model(s).
    Handles the 'separated' strategy by averaging outputs if selected_model is a dict.
    """
    logger.debug(f"⚡ [Pipeline] Predicting solar power. Strategy: '{strategy}', Input shape: {input_df.shape}")
    
    # --- Feature Engineering & Column Ordering ---
    # The model expects columns in a specific order. We must enforce it.
    
    # 1. Get the expected feature columns from the model bundle.
    expected_features = selected_model.get("feature_cols")
    if not expected_features:
        raise ValueError(f"Could not find 'feature_cols' in model bundle for strategy '{strategy}'")

    # 2. Filter the input DataFrame to only include expected features.
    # This also handles dropping metadata columns like Date, lat, lon, etc.
    # It will raise a KeyError if a required feature is missing in the input_df.
    try:
        X_solar = input_df[expected_features]
    except KeyError as e:
        missing_cols = set(expected_features) - set(input_df.columns)
        logger.error(f"Missing required columns for solar prediction: {missing_cols}")
        raise ValueError(f"Input data is missing required columns for the solar model: {', '.join(missing_cols)}") from e

    # --- Prediction ---
    if strategy == "seperated" and isinstance(selected_model, dict) and "models" in selected_model:
        # Ensemble averaging for separated models:
        # Run prediction on ALL site models and take the mean.
        logger.debug(f"👥 [Strategy] Averaging output from {len(selected_model['models'])} 'seperated' models.")
        model_predictions = []
        
        # For 'seperated' strategy, all models share the same feature set (without one-hot encoding).
        for model_name, model in selected_model['models'].items():
            logger.debug(f"  Predicting with model: {model_name}")
            pred = model.predict(X_solar)
            model_predictions.append(pred)
        
        if not model_predictions:
             raise ValueError("No separated models available to predict.")

        return np.mean(model_predictions, axis=0)
        
    elif strategy == "merged" and isinstance(selected_model, dict) and "model" in selected_model:
        logger.debug("🎯 [Strategy] Using 'merged' model.")
        # `X_solar` is already ordered correctly for the merged model.
        return selected_model['model'].predict(X_solar)
        
    else:
        # Mismatch between strategy name and model type
        err_msg = f"Strategy mismatch: '{strategy}' selected but model type is {type(selected_model)} for strategy {strategy}"
        logger.error(err_msg)
        raise ValueError(f"Model type mismatch for strategy {strategy}")

@app.get(
    "/predict/day",
    response_model=DayPredictionResponse,
    summary="Predict solar generation for a day range",
    description="Predict solar generation for a specific day or date range at a given location"
)
async def predict_day(
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
    logger.info(f"Processing /predict/day for lat={lat}, lon={lon}, range={startDate}-{endDate}, strategy='{strategy}'")
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
            'PMP': pmp
        })

        # Feature Engineering: Extract year, month, day for the model
        input_df['year'] = input_df['Date'].dt.year
        input_df['month'] = input_df['Date'].dt.month
        input_df['day'] = input_df['Date'].dt.day

        # 2. Predict Weather Features (The Missing Link)
        input_df = await _predict_weather_features(input_df)

        # Add one-hot encoding for merged model strategy
        if strategy == "merged" and isinstance(selected_model, dict) and "datasets" in selected_model:
            datasets = selected_model.get('datasets', [])
            if datasets:
                first_dataset = datasets[0]
                for ds in datasets:
                    input_df[f"ds_{ds}"] = 1 if ds == first_dataset else 0

        # Predict using the loaded models
        try:
            predicted_kwh = _predict_solar_power(input_df, selected_model, strategy)
        except Exception as e:
            logger.error(f"Solar power prediction failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Model prediction failed unexpectedly: {str(e)}")

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
        logger.error(f"Date validation error for /predict/day: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM-DD: {str(e)}")


# @app.get(
#     "/predict/month",
#     response_model=MonthPredictionResponse,
#     summary="Predict solar generation for a month range",
#     description="Predict solar generation for a specific month range at a given location"
# )
# async def predict_month(
#     lon: float = Query(..., ge=-180, le=180, description="Longitude"),
#     lat: float = Query(..., ge=-90, le=90, description="Latitude"),
#     startDate: str = Query(..., description="Start month (YYYY-MM)", alias="startDate"),
#     endDate: str = Query(..., description="End month (YYYY-MM)", alias="endDate"),
#     pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)"),
#     strategy: str = Query("merged", enum=["merged", "seperated"], description="Model strategy to use")
# ) -> MonthPredictionResponse:
#     """
#     Predict solar generation for a specific month range at a given location.
    
#     Example: /predict/month?lon=119.588339&lat=23.530236&startDate=2025-01&endDate=2025-05&pmp=1000
#     """
#     logger.info(f"Processing /predict/month for lat={lat}, lon={lon}, range={startDate}-{endDate}, strategy='{strategy}'")
#     try:
#         # Validate date format
#         start = datetime.strptime(startDate, "%Y-%m")
#         end = datetime.strptime(endDate, "%Y-%m")
        
#         if end < start:
#             raise HTTPException(status_code=400, detail="endDate must be after or equal to startDate")
        
#         selected_model = solar_models.get(strategy)
#         if selected_model is None:
#             raise HTTPException(status_code=503, detail=f"Solar model for strategy '{strategy}' is not loaded.")

#         # Create daily range for the months to get accurate aggregation
#         next_month = end.replace(day=28) + timedelta(days=4)
#         last_day_of_end_month = next_month - timedelta(days=next_month.day)
        
#         date_range = pd.date_range(start=start, end=last_day_of_end_month)
#         input_df = pd.DataFrame({'Date': date_range, 'lat': lat, 'lon': lon, 'PMP': pmp})
        
#         # Feature Engineering
#         input_df['year'] = input_df['Date'].dt.year
#         input_df['month'] = input_df['Date'].dt.month
#         input_df['day'] = input_df['Date'].dt.day
        
#         # 2. Predict Weather
#         input_df = await _predict_weather_features(input_df)

#         # Add one-hot encoding for merged model strategy
#         if strategy == "merged" and isinstance(selected_model, dict) and "datasets" in selected_model:
#             datasets = selected_model.get('datasets', [])
#             if datasets:
#                 first_dataset = datasets[0]
#                 for ds in datasets:
#                     input_df[f"ds_{ds}"] = 1 if ds == first_dataset else 0

#         try:
#             daily_kwh = _predict_solar_power(input_df, selected_model, strategy)
#             input_df['kwh'] = daily_kwh
#         except Exception as e:
#              logger.error(f"Solar power prediction failed: {e}", exc_info=True)
#              raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

#         # Aggregate by month
#         input_df['Month'] = input_df['Date'].dt.to_period('M')
#         monthly_data = input_df.groupby('Month')['kwh'].sum().reset_index()

#         predictions = []
#         for _, row in monthly_data.iterrows():
#             predictions.append(MonthPrediction(date=str(row['Month']), kwh=round(row['kwh'], 2)))
        
#         return MonthPredictionResponse(
#             location=Location(lat=lat, lon=lon),
#             month=start.month,
#             year=start.year,
#             pmp=pmp,
#             predictions=predictions
#         )
    
#     except ValueError as e:
#         logger.error(f"Date validation error for /predict/month: {e}", exc_info=True)
#         raise HTTPException(status_code=400, detail=f"Invalid date format. Use YYYY-MM: {str(e)}")


# @app.get(
#     "/predict/year",
#     response_model=YearPredictionResponse,
#     summary="Predict solar generation for a year",
#     description="Predict solar generation for a specific year at a given location"
# )
# async def predict_year(
#     lon: float = Query(..., ge=-180, le=180, description="Longitude"),
#     lat: float = Query(..., ge=-90, le=90, description="Latitude"),
#     year: int = Query(..., ge=2000, le=2100, description="Year (YYYY)"),
#     pmp: Optional[float] = Query(1000, description="Panel Maximum Power (W)"),
#     strategy: str = Query("merged", enum=["merged", "seperated"], description="Model strategy to use")
# ) -> YearPredictionResponse:
#     """
#     Predict solar generation for a specific year at a given location.
    
#     Example: /predict/year?lon=119.588339&lat=23.530236&year=2025&pmp=1000
#     """
#     logger.info(f"Processing /predict/year for lat={lat}, lon={lon}, year={year}, strategy='{strategy}'")
#     selected_model = solar_models.get(strategy)
#     if selected_model is None:
#         raise HTTPException(status_code=503, detail=f"Solar model for strategy '{strategy}' is not loaded.")

#     start_date = datetime(year, 1, 1)
#     end_date = datetime(year, 12, 31)
    
#     date_range = pd.date_range(start=start_date, end=end_date)
#     input_df = pd.DataFrame({'Date': date_range, 'lat': lat, 'lon': lon, 'PMP': pmp})

#     # Feature Engineering
#     input_df['year'] = input_df['Date'].dt.year
#     input_df['month'] = input_df['Date'].dt.month
#     input_df['day'] = input_df['Date'].dt.day
    
#     # 2. Predict Weather
#     input_df = await _predict_weather_features(input_df)

#     # Add one-hot encoding for merged model strategy
#     if strategy == "merged" and isinstance(selected_model, dict) and "datasets" in selected_model:
#         datasets = selected_model.get('datasets', [])
#         if datasets:
#             first_dataset = datasets[0]
#             for ds in datasets:
#                 input_df[f"ds_{ds}"] = 1 if ds == first_dataset else 0

#     try:
#         daily_kwh = _predict_solar_power(input_df, selected_model, strategy)
#         total_kwh = round(float(daily_kwh.sum()), 2)
#     except Exception as e:
#         logger.error(f"Solar power prediction failed: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
    
#     return YearPredictionResponse(
#         location=Location(lat=lat, lon=lon),
#         year=year,
#         pmp=pmp,
#         kwh=total_kwh
#     )
