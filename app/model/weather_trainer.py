"""
Weather forecasting model trainer - exact match to xgb-weather-forecaster.py reference code.
Trains XGBoost models for multi-day weather prediction using sequence windows.
"""

import math
import pickle
import logging
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import os

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# Fixed XGBoost parameters matching reference code exactly
XGB_FIXED_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "tree_method": "hist",
}


@dataclass
class SequenceSet:
    """Container for sequences with metadata."""
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    lats: np.ndarray
    lons: np.ndarray


def shifted_sin_cos(doy: np.ndarray, offset: int, year_lengths: np.ndarray) -> tuple:
    """
    Compute phase-shifted sin/cos features for seasonality.
    Matches reference implementation exactly.
    """
    shifted = doy - offset
    angles = 2.0 * math.pi * ((shifted % year_lengths) / year_lengths)
    return np.sin(angles), np.cos(angles)


def load_all_stations(data_dir: str) -> List[pd.DataFrame]:
    """
    Load all station CSV files from a directory.
    """
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    frames = [pd.read_csv(p) for p in sorted(files)]
    return frames


def train_weather_model(
    data_dir: str,
    output_dir: str,
    targets: Optional[List[str]] = None,
    win: int = 30,
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
    mode: str = "single",
) -> Dict[str, Any]:
    """
    Train weather forecasting model - exact match to xgb-weather-forecaster.py.
    
    Args:
        data_dir: Path to directory with historical weather CSVs
        output_dir: Directory to save trained model bundle
        targets: Weather variables to predict (default: all 8)
        win: Window size in days (default: 30)
        train_end: Train period end date (YYYY-MM-DD)
        val_end: Validation period end date (YYYY-MM-DD)
        mode: "single" or "multi" - single model per target or one multi-output model
        
    Returns:
        Dict with model bundle path and metadata
    """
    logger.info("=" * 70)
    logger.info("Weather Model Training - Exact Match to Reference")
    logger.info("=" * 70)
    
    # Default targets (all 8 weather variables)
    if targets is None:
        targets = [
            "T2M", "T2M_MAX", "TS", "CLOUD_AMT_DAY", "CLOUD_OD",
            "ALLSKY_SFC_SW_DWN", "RH2M", "ALLSKY_SFC_SW_DIRH"
        ]
    
    logger.info(f"Targets: {targets}")
    logger.info(f"Window size: {win} days")
    logger.info(f"Mode: {mode}")
    
    # --- 1. Load Data ---
    logger.info(f"Loading CSVs from: {data_dir}")
    frames = load_all_stations(data_dir)
    df = pd.concat(frames, ignore_index=True)
    
    # Build Date column if needed
    if "Date" not in df.columns:
        if not {"YEAR", "MO", "DY"}.issubset(df.columns):
            raise ValueError("CSV must have either 'Date' or 'YEAR', 'MO', 'DY'")
        df["Date"] = pd.to_datetime(
            dict(year=df["YEAR"], month=df["MO"], day=df["DY"]), errors="coerce"
        )
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    logger.info(f"Data range: {df['Date'].min()} to {df['Date'].max()}, {len(df)} rows")
    
    # Extract LAT/LON
    if "LAT" in df.columns and "LON" in df.columns:
        lats = df["LAT"].values
        lons = df["LON"].values
    else:
        logger.warning("No LAT/LON columns; using zeros")
        lats = np.zeros(len(df))
        lons = np.zeros(len(df))
    
    # --- 2. Compute Hottest DOY Offset for Seasonality ---
    if "T2M" in df.columns:
        clim = df.groupby(df["Date"].dt.dayofyear)["T2M"].mean()
        hottest_offset = int(clim.idxmax())
        logger.info(f"Hottest day-of-year offset: {hottest_offset}")
    else:
        hottest_offset = 200
        logger.warning(f"No T2M column; using default offset {hottest_offset}")
    
    # --- 3. Define Feature Columns ---
    # Explicitly use targets as features to ensure consistency
    features = targets
    logger.info(f"Feature columns ({len(features)}): {features}")
    
    # --- 4. Build Sequences (sliding window) ---
    logger.info(f"Building sequences with window={win}...")
    
    year_lengths = df["Date"].dt.is_leap_year.map({True: 366, False: 365}).astype(int).values
    doy = df["Date"].dt.dayofyear.values
    sin_doy, cos_doy = shifted_sin_cos(doy, hottest_offset, year_lengths)
    
    feats_arr = df[features].to_numpy(dtype="float32")
    targets_arr = df[targets].to_numpy(dtype="float32")
    dates_arr = df["Date"].values
    
    X_list, y_list, dates_list, lats_list, lons_list = [], [], [], [], []
    
    for t in range(win, len(df)):
        # Flatten past window features
        past_feats = feats_arr[t - win:t].flatten()
        
        # Current day's seasonal + spatial features
        current_feat = np.concatenate([
            past_feats,
            [lons[t], lats[t], sin_doy[t], cos_doy[t]]
        ])
        
        X_list.append(current_feat)
        y_list.append(targets_arr[t])
        dates_list.append(dates_arr[t])
        lats_list.append(lats[t])
        lons_list.append(lons[t])
    
    if not X_list:
        raise ValueError("No sequences generated (data too short?)")
    
    X = np.stack(X_list)
    y = np.stack(y_list)
    dates = np.array(dates_list)
    seq_lats = np.array(lats_list)
    seq_lons = np.array(lons_list)
    
    logger.info(f"Sequence array shapes: X={X.shape}, y={y.shape}")
    
    # --- 5. NaN Handling (remove rows with NaN in target) ---
    valid_mask = np.isfinite(y).all(axis=1)
    n_removed = (~valid_mask).sum()
    if n_removed > 0:
        logger.info(f"Removing {n_removed} sequences with NaN in targets")
        X = X[valid_mask]
        y = y[valid_mask]
        dates = dates[valid_mask]
        seq_lats = seq_lats[valid_mask]
        seq_lons = seq_lons[valid_mask]
    
    logger.info(f"Final data shape: X={X.shape}, y={y.shape}")
    
    # --- 6. Calendar-based Train/Val/Test Splits ---
    if train_end is None:
        # Default: use 70% for training
        n_train = int(0.7 * len(dates))
        train_end_date = pd.Timestamp(dates[n_train])
    else:
        train_end_date = pd.Timestamp(train_end)
    
    if val_end is None:
        # Default: next 15% for validation
        n_val = int(0.15 * len(dates))
        val_end_date = pd.Timestamp(dates[n_train + n_val]) if train_end is None else train_end_date + pd.Timedelta(days=int(0.15 * len(dates) / 0.7))
    else:
        val_end_date = pd.Timestamp(val_end)
    
    train_mask = dates <= train_end_date
    val_mask = (dates > train_end_date) & (dates <= val_end_date)
    test_mask = dates > val_end_date
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"Train split: {train_mask.sum()} samples (up to {train_end_date.date()})")
    logger.info(f"Val split:   {val_mask.sum()} samples (up to {val_end_date.date()})")
    logger.info(f"Test split:  {test_mask.sum()} samples")
    
    # --- 7. Train Model (Single or Multi-Output) ---
    logger.info(f"Training XGBoost model (mode={mode})...")
    
    if mode == "single":
        # One model per target (separate)
        models = {}
        for i, target_name in enumerate(targets):
            logger.info(f"  Training model for target: {target_name}")
            model = XGBRegressor(**XGB_FIXED_PARAMS)
            model.fit(X_train, y_train[:, i])
            models[target_name] = model
        
        bundle = {
            "mode": "single",
            "targets": targets,
            "models": models,
            "win": win,
            "features": features,
            "hottest_offset": hottest_offset,
        }
    
    else:
        # Multi-output (one model for all targets)
        logger.info(f"Training single multi-output model for {len(targets)} targets")
        base_model = XGBRegressor(**XGB_FIXED_PARAMS)
        model = MultiOutputRegressor(base_model, n_jobs=1)
        model.fit(X_train, y_train)
        
        bundle = {
            "mode": "multi",
            "targets": targets,
            "model": model,
            "win": win,
            "features": features,
            "hottest_offset": hottest_offset,
        }
    
    logger.info("Training complete!")
    
    # --- 8. Generate Predictions on Test Set and Save ---
    logger.info("Generating predictions on the test set...")
    if mode == "single":
        y_pred_test = np.zeros_like(y_test)
        for i, target_name in enumerate(targets):
            y_pred_test[:, i] = models[target_name].predict(X_test)
    else:
        y_pred_test = model.predict(X_test)

    # Create a DataFrame for predictions
    pred_df = pd.DataFrame(index=dates[test_mask])
    pred_df['Date'] = dates[test_mask]
    for i, target_name in enumerate(targets):
        pred_df[f"{target_name}_true"] = y_test[:, i]
        pred_df[f"{target_name}_pred"] = y_pred_test[:, i]

    output_path = Path(output_dir)
    pred_filename = output_path / "weather-pred.csv"
    pred_df.to_csv(pred_filename, index=False)
    logger.info(f"Test predictions saved to: {pred_filename}")

    # --- 9. Save Model Bundle ---
    bundle_filename = output_path / "weather_model_bundle.pkl"
    with open(bundle_filename, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("=" * 70)
    logger.info(f"Model bundle saved: {bundle_filename}")
    logger.info(f"  Mode: {bundle['mode']}")
    logger.info(f"  Targets: {bundle['targets']}")
    logger.info(f"  Window: {bundle['win']}")
    logger.info(f"  Features: {len(bundle['features'])}")
    logger.info("=" * 70)
    
    return {
        "bundle_path": str(bundle_filename),
        "pred_path": str(pred_filename),
        "mode": bundle["mode"],
        "targets": bundle["targets"],
        "window_size": bundle["win"],
        "features": bundle["features"],
        "hottest_offset": bundle["hottest_offset"],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
    }

def load_weather_model(bundle_path: str) -> Dict[str, Any]:
    """
    Load trained weather model bundle from pickle file.
    
    Args:
        bundle_path: Path to model bundle pickle file
        
    Returns:
        Model bundle dictionary
    """
    logger.info(f"Loading weather model bundle: {bundle_path}")
    
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    
    logger.info(f"  Mode: {bundle['mode']}")
    logger.info(f"  Targets: {bundle['targets']}")
    logger.info(f"  Window: {bundle['win']}")
    logger.info(f"  Features: {len(bundle['features'])}")
    
    return bundle
