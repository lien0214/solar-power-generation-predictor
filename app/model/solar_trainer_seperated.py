"""
Solar generation forecasting model trainer - separated models.
Trains one XGBoost model per solar dataset.
"""

import pickle
import logging
from typing import Dict, List
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# Required weather features for solar prediction
REQUIRED_FEATURES = [
    "T2M", "T2M_MAX", "TS", "CLOUD_AMT_DAY", "CLOUD_OD",
    "ALLSKY_SFC_SW_DWN", "RH2M", "ALLSKY_SFC_SW_DIRH"
]

# Fixed XGBoost parameters
SOLAR_XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 600,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "objective": "reg:squarederror",
    "random_state": 42,
    "tree_method": "hist",
}


def train_solar_model_seperated(
    solar_files: Dict[str, str],
    weather_hist_file: str,
    weather_pred_file: str,
    output_dir: str,
    test_months: int = 6,
    valid_months: int = 1,
) -> Dict:
    """
    Train a separate solar forecasting model for each dataset.
    
    Args:
        solar_files: Dict mapping dataset names to CSV file paths
        weather_hist_file: Path to historical weather CSV
        weather_pred_file: Path to predicted weather CSV
        output_dir: Directory to save trained model bundle
        test_months: Number of 30-day months for test period
        valid_months: Number of 30-day months for validation period
        
    Returns:
        Dict with model bundle path and metadata
    """
    logger.info("=" * 70)
    logger.info("Solar Model Training - Separated Models")
    logger.info("=" * 70)
    
    days_per_month = 30
    datasets = list(solar_files.keys())
    logger.info(f"Solar datasets: {datasets}")
    
    # --- Load Weather Data ---
    logger.info(f"Loading historical weather: {weather_hist_file}")
    w_hist = pd.read_csv(weather_hist_file)
    w_hist["Date"] = pd.to_datetime(
        dict(year=w_hist["YEAR"], month=w_hist["MO"], day=w_hist["DY"]),
        errors="coerce"
    )
    w_hist_feats = w_hist[["Date"] + REQUIRED_FEATURES]
    
    logger.info(f"Loading predicted weather: {weather_pred_file}")
    w_pred = pd.read_csv(weather_pred_file)
    w_pred["Date"] = pd.to_datetime(w_pred["Date"], errors="coerce")
    
    pred_map = {f: f"{f}_pred" for f in REQUIRED_FEATURES}
    w_pred_only = w_pred[["Date"] + list(pred_map.values())].rename(
        columns={v: k for k, v in pred_map.items()}
    )
    
    trained_models = {}
    all_stats = []

    for ds_name, solar_path in solar_files.items():
        logger.info("-" * 50)
        logger.info(f"Training model for dataset: {ds_name}")
        
        # --- Load Solar Data ---
        df = pd.read_csv(solar_path)
        if "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        solar_df = df[["Date", "PMP", "KWh"]]
        
        # --- Compute Splits ---
        ds_min, ds_max = solar_df["Date"].min(), solar_df["Date"].max()
        wp_min, wp_max = w_pred_only["Date"].min(), w_pred_only["Date"].max()
        overlap_start = max(ds_min, wp_min)
        overlap_end = min(ds_max, wp_max)
        
        test_end = overlap_end.normalize()
        test_start = test_end - timedelta(days=test_months * days_per_month - 1)
        valid_start = test_start
        valid_end = valid_start + timedelta(days=valid_months * days_per_month - 1)
        train_end = valid_start - timedelta(days=1)

        # --- Create Datasets ---
        train_df = solar_df.merge(w_hist_feats, on="Date", how="left")
        train_mask = train_df["Date"] <= train_end
        
        feat_cols = REQUIRED_FEATURES + ["PMP"]
        X_train = train_df.loc[train_mask, feat_cols]
        y_train = train_df.loc[train_mask, "KWh"].astype(float).values
        
        valid_df = solar_df.merge(w_pred_only, on="Date", how="left")
        valid_mask = (valid_df["Date"] >= valid_start) & (valid_df["Date"] <= valid_end)
        X_valid = valid_df.loc[valid_mask, feat_cols]
        y_valid = valid_df.loc[valid_mask, "KWh"].astype(float).values
        
        # --- Train Model ---
        X_trainval = pd.concat([X_train, X_valid], axis=0)
        y_trainval = np.concatenate([y_train, y_valid])
        
        model = XGBRegressor(**SOLAR_XGB_PARAMS)
        model.fit(X_trainval, y_trainval)
        
        trained_models[ds_name] = model
        stats = {
            "dataset": ds_name,
            "train_samples": len(X_train),
            "val_samples": len(X_valid),
        }
        all_stats.append(stats)
        logger.info(f"  Training complete for {ds_name}: {stats}")

    # --- Save Model Bundle ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        "models": trained_models,
        "datasets": datasets,
        "feature_cols": feat_cols,
        "required_features": REQUIRED_FEATURES,
        "xgb_params": SOLAR_XGB_PARAMS,
    }
    
    bundle_filename = output_path / "solar_model_bundle_seperated.pkl"
    with open(bundle_filename, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    logger.info("=" * 70)
    logger.info(f"Separated solar model bundle saved: {bundle_filename}")
    logger.info("=" * 70)
    
    return {
        "bundle_path": str(bundle_filename),
        "datasets": datasets,
        "stats": all_stats,
    }

def load_solar_model_seperated(bundle_path: str) -> Dict:
    """
    Load trained separated solar model bundle from pickle file.
    """
    logger.info(f"Loading separated solar model bundle: {bundle_path}")
    
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    
    logger.info(f"  Datasets: {bundle['datasets']}")
    return bundle
