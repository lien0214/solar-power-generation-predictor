"""
Solar generation forecasting model trainer - exact match to xgb-solar-forecaster-merged.py.
Trains a single XGBoost model for multiple solar datasets with one-hot encoding.
"""

import pickle
import logging
from typing import Dict
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


# Required weather features for solar prediction (exact match to reference)
REQUIRED_FEATURES = [
    "T2M", "T2M_MAX", "TS", "CLOUD_AMT_DAY", "CLOUD_OD",
    "ALLSKY_SFC_SW_DWN", "RH2M", "ALLSKY_SFC_SW_DIRH"
]

# Fixed XGBoost parameters (match best params from reference after tuning)
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


def train_solar_model(
    solar_files: Dict[str, str],
    weather_hist_file: str,
    weather_pred_file: str,
    output_dir: str,
    test_months: int = 6,
    valid_months: int = 1,
) -> Dict:
    """
    Train merged solar forecasting model - exact match to xgb-solar-forecaster-merged.py.
    
    Args:
        solar_files: Dict mapping dataset names to CSV file paths
        weather_hist_file: Path to historical weather CSV
        weather_pred_file: Path to predicted weather CSV (with _true and _pred columns)
        output_dir: Directory to save trained model bundle
        test_months: Number of 30-day months for test period (default: 6)
        valid_months: Number of 30-day months for validation period (default: 1)
        
    Returns:
        Dict with model bundle path and metadata
    """
    logger.info("=" * 70)
    logger.info("Solar Model Training - Merged (Exact Match to Reference)")
    logger.info("=" * 70)
    
    days_per_month = 30
    datasets = list(solar_files.keys())
    logger.info(f"Solar datasets: {datasets}")
    logger.info(f"Test period: {test_months} months ({test_months * days_per_month} days)")
    logger.info(f"Valid period: {valid_months} month ({valid_months * days_per_month} days)")
    
    # --- 1. Load Solar Data ---
    logger.info("Loading solar generation data...")
    solar_dfs = []
    for ds_name, path in solar_files.items():
        df = pd.read_csv(path)
        # Expect 'date' column (lowercase) as per reference
        if "date" in df.columns:
            df["Date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "Date" not in df.columns:
            raise ValueError(f"CSV {path} must have 'date' or 'Date' column")
        else:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        
        df = df.dropna(subset=["Date"])
        df["dataset"] = ds_name
        solar_dfs.append(df[["Date", "PMP", "KWh", "dataset"]])
        logger.info(f"  {ds_name}: {len(df)} records")
    
    solar_all = pd.concat(solar_dfs, ignore_index=True)
    logger.info(f"Total solar records: {len(solar_all)}")
    
    # --- 2. Load Historical Weather ---
    logger.info(f"Loading historical weather: {weather_hist_file}")
    w_hist = pd.read_csv(weather_hist_file)
    w_hist["Date"] = pd.to_datetime(
        dict(year=w_hist["YEAR"], month=w_hist["MO"], day=w_hist["DY"]),
        errors="coerce"
    )
    w_hist_feats = w_hist[["Date"] + REQUIRED_FEATURES]
    
    # --- 3. Load Predicted Weather (True/Pred Columns) ---
    logger.info(f"Loading predicted weather: {weather_pred_file}")
    w_pred = pd.read_csv(weather_pred_file)
    if "Date" not in w_pred.columns:
        raise ValueError("weather-pred.csv must have 'Date' column")
    w_pred["Date"] = pd.to_datetime(w_pred["Date"], errors="coerce")
    
    # Check for _true and _pred columns
    expected_pred = [f"{f}_pred" for f in REQUIRED_FEATURES]
    expected_true = [f"{f}_true" for f in REQUIRED_FEATURES]
    missing_pred = [c for c in expected_pred if c not in w_pred.columns]
    missing_true = [c for c in expected_true if c not in w_pred.columns]
    
    if missing_pred or missing_true:
        raise ValueError(
            f"weather-pred.csv missing columns.\n"
            f"Missing *_pred: {missing_pred}\nMissing *_true: {missing_true}"
        )
    
    # Normalize to match REQUIRED_FEATURES naming
    true_map = {f: f"{f}_true" for f in REQUIRED_FEATURES}
    pred_map = {f: f"{f}_pred" for f in REQUIRED_FEATURES}
    w_true = w_pred[["Date"] + list(true_map.values())].rename(
        columns={v: k for k, v in true_map.items()}
    )
    w_pred_only = w_pred[["Date"] + list(pred_map.values())].rename(
        columns={v: k for k, v in pred_map.items()}
    )
    
    # --- 4. Per-Dataset Split Calculation ---
    logger.info("Computing per-dataset train/val/test splits...")
    
    def split_last_six_months_for_dataset(ds_df, w_pred_df):
        """Calculate test/val periods for a dataset (last 6 months)."""
        ds_min, ds_max = ds_df["Date"].min(), ds_df["Date"].max()
        wp_min, wp_max = w_pred_df["Date"].min(), w_pred_df["Date"].max()
        overlap_start = max(ds_min, wp_min)
        overlap_end = min(ds_max, wp_max)
        
        if pd.isna(overlap_start) or pd.isna(overlap_end) or overlap_end < overlap_start:
            raise ValueError(f"No overlap for dataset {ds_df['dataset'].iloc[0]}")
        
        # Test period: last 6 months (180 days)
        test_end = overlap_end.normalize()
        test_start = test_end - timedelta(days=test_months * days_per_month - 1)
        
        # Validation: first month of test period
        valid_start = test_start
        valid_end = valid_start + timedelta(days=valid_months * days_per_month - 1)
        
        # Adjust if test period goes before overlap
        if test_start < overlap_start:
            shift = overlap_start - test_start
            test_start += shift
            test_end += shift
            valid_start += shift
            valid_end += shift
        
        # Ensure within overlap
        test_start = max(test_start, overlap_start)
        test_end = min(test_end, overlap_end)
        valid_start = max(valid_start, overlap_start)
        valid_end = min(valid_end, overlap_end)
        
        return valid_start, valid_end, test_start, test_end
    
    cutoffs = []
    for ds in datasets:
        ds_df = solar_all.loc[solar_all["dataset"] == ds]
        vs, ve, ts, te = split_last_six_months_for_dataset(ds_df, w_pred_only)
        cutoffs.append({
            "dataset": ds,
            "VALID_START": vs,
            "VALID_END": ve,
            "TEST_START": ts,
            "TEST_END": te,
        })
    
    cutoffs_df = pd.DataFrame(cutoffs)
    logger.info("Splits per dataset:")
    for _, r in cutoffs_df.iterrows():
        logger.info(
            f"  {r['dataset']}: Valid {r['VALID_START'].date()}…{r['VALID_END'].date()}, "
            f"Test {r['TEST_START'].date()}…{r['TEST_END'].date()}"
        )
    
    # --- 5. Create Train/Val/Test Splits with One-Hot Encoding ---
    logger.info("Creating train/val/test datasets...")
    
    # Add one-hot encoding for datasets
    for ds_name in datasets:
        solar_all[f"ds_{ds_name}"] = (solar_all["dataset"] == ds_name).astype(int)
    
    # Training: historical weather up to VALID_START per dataset
    train_base = solar_all.merge(w_hist_feats, on="Date", how="left").merge(
        cutoffs_df, on="dataset", how="left"
    )
    for ds_name in datasets:
        train_base[f"ds_{ds_name}"] = (train_base["dataset"] == ds_name).astype(int)
    
    TRAIN_END_PER = train_base["VALID_START"] - pd.Timedelta(days=1)
    train_mask = train_base["Date"] <= TRAIN_END_PER
    
    feat_cols = REQUIRED_FEATURES + ["PMP"] + [f"ds_{ds}" for ds in datasets]
    X_train = train_base.loc[train_mask, feat_cols]
    y_train = train_base.loc[train_mask, "KWh"].astype(float).values
    
    logger.info(f"Train: {len(X_train)} samples")
    
    # Validation: predicted weather during VALID period
    valid_pred_df = solar_all.merge(w_pred_only, on="Date", how="left").merge(
        cutoffs_df, on="dataset", how="left"
    )
    for ds_name in datasets:
        valid_pred_df[f"ds_{ds_name}"] = (valid_pred_df["dataset"] == ds_name).astype(int)
    
    valid_mask_pred = (
        (valid_pred_df["Date"] >= valid_pred_df["VALID_START"])
        & (valid_pred_df["Date"] <= valid_pred_df["VALID_END"])
    )
    
    X_valid_pred = valid_pred_df.loc[valid_mask_pred, feat_cols]
    y_valid_pred = valid_pred_df.loc[valid_mask_pred, "KWh"].astype(float).values
    
    logger.info(f"Validation: {len(X_valid_pred)} samples")
    
    # Test: predicted weather during TEST period
    test_pred_df = solar_all.merge(w_pred_only, on="Date", how="left").merge(
        cutoffs_df, on="dataset", how="left"
    )
    for ds_name in datasets:
        test_pred_df[f"ds_{ds_name}"] = (test_pred_df["dataset"] == ds_name).astype(int)
    
    test_mask_pred = (
        (test_pred_df["Date"] >= test_pred_df["TEST_START"])
        & (test_pred_df["Date"] <= test_pred_df["TEST_END"])
    )
    
    X_test_pred = test_pred_df.loc[test_mask_pred, feat_cols]
    y_test_pred = test_pred_df.loc[test_mask_pred, "KWh"].astype(float).values
    
    logger.info(f"Test: {len(X_test_pred)} samples")
    
    # --- 6. Train Final Model (Train + Validation) ---
    logger.info("Training XGBoost model...")
    X_trainval = pd.concat([X_train, X_valid_pred], axis=0)
    y_trainval = np.concatenate([y_train, y_valid_pred])
    
    model = XGBRegressor(**SOLAR_XGB_PARAMS)
    model.fit(X_trainval, y_trainval)
    
    logger.info("Training complete!")
    
    # --- 7. Save Model Bundle ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    bundle = {
        "model": model,
        "datasets": datasets,
        "feature_cols": feat_cols,
        "required_features": REQUIRED_FEATURES,
        "xgb_params": SOLAR_XGB_PARAMS,
    }
    
    bundle_filename = output_path / "solar_model_bundle.pkl"
    with open(bundle_filename, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info("=" * 70)
    logger.info(f"Solar model bundle saved: {bundle_filename}")
    logger.info(f"  Datasets: {datasets}")
    logger.info(f"  Features: {len(feat_cols)}")
    logger.info("=" * 70)
    
    return {
        "bundle_path": str(bundle_filename),
        "datasets": datasets,
        "feature_cols": feat_cols,
        "train_samples": len(X_train),
        "val_samples": len(X_valid_pred),
        "test_samples": len(X_test_pred),
    }


def load_solar_model(bundle_path: str) -> Dict:
    """
    Load trained solar model bundle from pickle file.
    
    Args:
        bundle_path: Path to model bundle pickle file
        
    Returns:
        Model bundle dictionary
    """
    logger.info(f"Loading solar model bundle: {bundle_path}")
    
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    
    logger.info(f"  Datasets: {bundle['datasets']}")
    logger.info(f"  Features: {len(bundle['feature_cols'])}")
    
    return bundle
