"""
Prediction Service.
Business logic for solar power generation predictions.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import math
import asyncio
import json
from pathlib import Path

import pandas as pd
import numpy as np

from ..schemas import DayPrediction, MonthPrediction
from .model_manager import get_model_manager

logger = logging.getLogger(__name__)


def shifted_sin_cos(doy: np.ndarray, offset: int, year_lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase-shifted sin/cos features for seasonality.
    """
    shifted = doy - offset
    angles = 2.0 * math.pi * ((shifted % year_lengths) / year_lengths)
    sinv = np.sin(angles)
    cosv = np.cos(angles)
    return sinv.astype("float32"), cosv.astype("float32")


class PredictionService:
    """Service for generating solar power predictions."""

    def __init__(self):
        """Initialize prediction service."""
        self.model_manager = get_model_manager()
        self.prediction_cache = {}

    async def _get_rolling_weather_forecast(
        self, start_date: datetime, end_date: datetime, lon: float, lat: float
    ) -> pd.DataFrame:
        """
        Generate a rolling weather forecast from a seed.
        """
        weather_model_bundle = self.model_manager.get_weather_model()
        if not weather_model_bundle:
            raise RuntimeError("Weather model is not loaded.")

        win = weather_model_bundle['win']
        features = weather_model_bundle['features']
        targets = weather_model_bundle['targets']
        hottest_offset = weather_model_bundle['hottest_offset']
        
        # Load historical weather data to find the seed
        try:
            hist_df = pd.read_csv(self.model_manager.weather_hist_file)
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
        except FileNotFoundError:
            raise RuntimeError("Historical weather data file not found.")

        # Find the last valid date and get the seed
        seed_end_date = start_date - timedelta(days=1)
        seed_start_date = seed_end_date - timedelta(days=win - 1)
        
        seed_df = hist_df[(hist_df['Date'] >= seed_start_date) & (hist_df['Date'] <= seed_end_date)]
        
        if len(seed_df) < win:
            raise RuntimeError(f"Not enough historical data to form a seed window of {win} days.")

        # Prepare for rolling forecast
        window_feats = seed_df[features].to_numpy(dtype="float32")
        last_date = seed_end_date
        
        predictions = []
        
        current_date = start_date
        while current_date <= end_date:
            year_len = 366 if current_date.is_leap_year else 365
            doy = current_date.timetuple().tm_yday
            sin_doy, cos_doy = shifted_sin_cos(np.array([doy]), hottest_offset, np.array([year_len]))

            # Create input for the model
            flat_window = window_feats.flatten()
            extras = np.array([lon, lat, sin_doy[0], cos_doy[0]], dtype="float32")
            X_row = np.concatenate([flat_window, extras]).reshape(1, -1)

            # Predict
            if weather_model_bundle['mode'] == 'single':
                y_pred = np.array([model.predict(X_row)[0] for model in weather_model_bundle['models'].values()]).T
            else:
                y_pred = weather_model_bundle['model'].predict(X_row)[0]

            # Store prediction
            pred_row = {'Date': current_date, 'LAT': lat, 'LON': lon}
            for i, target in enumerate(targets):
                pred_row[f"{target}_pred"] = y_pred[i]
            predictions.append(pred_row)

            # Update window for next prediction
            new_feat_row = window_feats[-1, :].copy()
            name_to_idx = {name: i for i, name in enumerate(features)}
            for i, target in enumerate(targets):
                if target in name_to_idx:
                    new_feat_row[name_to_idx[target]] = y_pred[i]
            
            window_feats = np.vstack([window_feats[1:], new_feat_row])
            
            current_date += timedelta(days=1)

        pred_df = pd.DataFrame(predictions)
        
        # Save predictions to a file
        pred_dir = Path("repo/data/weather_predictions")
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"{lon}_{lat}_{start_date.date()}_{end_date.date()}.json"
        pred_df.to_json(pred_file, orient="records", date_format="iso")
        
        return pred_df

    async def _get_weather_forecast(self, start_date: datetime, end_date: datetime, lon: float, lat: float) -> pd.DataFrame:
        """
        Get weather forecast for a date range.
        """
        # First, check if a pre-computed prediction file exists
        pred_dir = Path("repo/data/weather_predictions")
        pred_file = pred_dir / f"{lon}_{lat}_{start_date.date()}_{end_date.date()}.json"
        if pred_file.exists():
            return pd.read_json(pred_file, orient="records")

        # If not, check the main weather prediction file
        try:
            df = pd.read_csv(self.model_manager.weather_pred_file)
            df['Date'] = pd.to_datetime(df['Date'])
            
            min_date, max_date = df['Date'].min(), df['Date'].max()
            if start_date >= min_date and end_date <= max_date:
                mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
                return df.loc[mask]
            else:
                logger.warning("Requested date range is outside the pre-computed weather predictions.")
                return await self._get_rolling_weather_forecast(start_date, end_date, lon, lat)

        except FileNotFoundError:
            logger.warning(f"Weather prediction file not found. Falling back to rolling forecast.")
            return await self._get_rolling_weather_forecast(start_date, end_date, lon, lat)
        except Exception as e:
            logger.error(f"Could not get weather forecast: {e}")
            raise RuntimeError("Weather forecast data not available.")

    async def _prepare_prediction_data(self, dates: pd.Series, pmp: float, lon: float, lat: float) -> pd.DataFrame:
        """
        Prepare a DataFrame with the required features for prediction.
        """
        pred_df = pd.DataFrame({'Date': dates})
        
        weather_data = await self._get_weather_forecast(pred_df['Date'].min(), pred_df['Date'].max(), lon, lat)
        
        pred_df = pd.merge(pred_df, weather_data, on='Date', how='left')

        pred_df['PMP'] = pmp
        
        weather_model_bundle = self.model_manager.get_weather_model()
        if not weather_model_bundle:
            raise RuntimeError("Weather model is not loaded.")

        for col in weather_model_bundle['targets']:
            pred_col = f"{col}_pred"
            if pred_col not in pred_df.columns:
                 raise RuntimeError(f"Missing weather prediction column: {pred_col}")
            pred_df[col] = pred_df[pred_col]

        return pred_df
    
    async def predict_day_range(
        self,
        lon: float,
        lat: float,
        start_date: str,
        end_date: str,
        pmp: float,
        tactic: str
    ) -> List[DayPrediction]:
        """
        Predict solar generation for a date range.
        """
        cache_key = (lon, lat, start_date, end_date, pmp, tactic)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        if not self.model_manager.is_ready():
            raise RuntimeError("Models not loaded. Please wait for initialization.")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if end < start:
            raise ValueError("end_date must be after or equal to start_date")
        
        dates = pd.to_datetime(pd.date_range(start, end, freq='D'))
        pred_df = await self._prepare_prediction_data(dates, pmp, lon, lat)
        
        if tactic == "merged":
            model_bundle = self.model_manager.get_solar_model_merged()
            if not model_bundle:
                raise RuntimeError("Merged solar model is not loaded.")
            model = model_bundle['model']
            
            first_dataset = model_bundle['datasets'][0]
            for ds in model_bundle['datasets']:
                pred_df[f"ds_{ds}"] = 1 if ds == first_dataset else 0
                
            features = model_bundle['feature_cols']
            X = pred_df[features]
            kwh_predictions = model.predict(X)
            
        elif tactic == "seperated":
            model_bundle = self.model_manager.get_solar_model_seperated()
            if not model_bundle:
                raise RuntimeError("Separated solar model is not loaded.")

            first_dataset = list(model_bundle['models'].keys())[0]
            model = model_bundle['models'][first_dataset]
            
            features = model_bundle['feature_cols']
            X = pred_df[features]
            kwh_predictions = model.predict(X)
        else:
            raise ValueError(f"Unknown solar model tactic: {tactic}")

        predictions = []
        for i, row in pred_df.iterrows():
            predictions.append(
                DayPrediction(
                    date=row['Date'].strftime("%Y-%m-%d"),
                    kwh=round(max(0, kwh_predictions[i]), 2)
                )
            )
        
        self.prediction_cache[cache_key] = predictions
        return predictions

    async def predict_month_range(
        self,
        lon: float,
        lat: float,
        start_date: str,
        end_date: str,
        pmp: float,
        tactic: str
    ) -> List[MonthPrediction]:
        """
        Predict solar generation for a month range.
        """
        start = datetime.strptime(start_date, "%Y-%m")
        end = datetime.strptime(end_date, "%Y-%m")
        
        end_of_month = end.replace(day=1) + pd.offsets.MonthEnd(0)
        
        daily_predictions = await self.predict_day_range(
            lon=lon,
            lat=lat,
            start_date=start.strftime("%Y-%m-01"),
            end_date=end_of_month.strftime("%Y-%m-%d"),
            pmp=pmp,
            tactic=tactic
        )
        
        monthly_sum = {}
        for pred in daily_predictions:
            month = pred.date[:7]
            if month not in monthly_sum:
                monthly_sum[month] = 0
            monthly_sum[month] += pred.kwh
            
        predictions = []
        for month, total_kwh in monthly_sum.items():
            if month >= start_date and month <= end_date:
                predictions.append(
                    MonthPrediction(
                        date=month,
                        kwh=round(total_kwh, 2)
                    )
                )
        
        return predictions

    async def predict_year(
        self,
        lon: float,
        lat: float,
        year: int,
        pmp: float,
        tactic: str
    ) -> float:
        """
        Predict total solar generation for a year.
        """
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        daily_predictions = await self.predict_day_range(
            lon=lon,
            lat=lat,
            start_date=start_date,
            end_date=end_date,
            pmp=pmp,
            tactic=tactic
        )
        
        total_kwh = sum(p.kwh for p in daily_predictions)
        
        return round(total_kwh, 2)


# Global service instance
prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """Get prediction service instance."""
    return prediction_service
