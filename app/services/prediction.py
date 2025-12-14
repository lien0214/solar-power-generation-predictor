"""
Prediction Service.
Business logic for solar power generation predictions.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from ..schemas import DayPrediction, MonthPrediction
from .model_manager import get_model_manager

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for generating solar power predictions."""
    
    def __init__(self):
        """Initialize prediction service."""
        self.model_manager = get_model_manager()
    
    async def predict_day_range(
        self,
        lon: float,
        lat: float,
        start_date: str,
        end_date: str,
        pmp: float
    ) -> List[DayPrediction]:
        """
        Predict solar generation for a date range.
        
        Args:
            lon: Longitude
            lat: Latitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            pmp: Panel Maximum Power (W)
            
        Returns:
            List of daily predictions
        """
        # Validate models are ready
        if not self.model_manager.is_ready():
            raise RuntimeError("Models not loaded. Please wait for initialization.")
        
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        if end < start:
            raise ValueError("end_date must be after or equal to start_date")
        
        # Generate predictions for each day
        predictions = []
        current = start
        
        while current <= end:
            # TODO: Replace with actual model prediction
            # For now, using mock data
            kwh = round(5.0 + (hash(current.strftime("%Y-%m-%d")) % 100) / 100, 2)
            
            predictions.append(
                DayPrediction(
                    date=current.strftime("%Y-%m-%d"),
                    kwh=kwh
                )
            )
            current += timedelta(days=1)
        
        return predictions
    
    async def predict_month_range(
        self,
        lon: float,
        lat: float,
        start_date: str,
        end_date: str,
        pmp: float
    ) -> List[MonthPrediction]:
        """
        Predict solar generation for a month range.
        
        Args:
            lon: Longitude
            lat: Latitude
            start_date: Start month (YYYY-MM)
            end_date: End month (YYYY-MM)
            pmp: Panel Maximum Power (W)
            
        Returns:
            List of monthly predictions
        """
        # Validate models are ready
        if not self.model_manager.is_ready():
            raise RuntimeError("Models not loaded. Please wait for initialization.")
        
        # Parse month dates
        start = datetime.strptime(start_date, "%Y-%m")
        end = datetime.strptime(end_date, "%Y-%m")
        
        if end < start:
            raise ValueError("end_date must be after or equal to start_date")
        
        # Generate predictions for each month
        predictions = []
        current = start
        
        while current <= end:
            # TODO: Replace with actual model prediction
            # For now, using mock data
            kwh = round(150.0 + (hash(current.strftime("%Y-%m")) % 50), 2)
            
            predictions.append(
                MonthPrediction(
                    date=current.strftime("%Y-%m"),
                    kwh=kwh
                )
            )
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return predictions
    
    async def predict_year(
        self,
        lon: float,
        lat: float,
        year: int,
        pmp: float
    ) -> float:
        """
        Predict total solar generation for a year.
        
        Args:
            lon: Longitude
            lat: Latitude
            year: Year
            pmp: Panel Maximum Power (W)
            
        Returns:
            Total predicted kWh for the year
        """
        # Validate models are ready
        if not self.model_manager.is_ready():
            raise RuntimeError("Models not loaded. Please wait for initialization.")
        
        # TODO: Replace with actual model prediction
        # For now, using mock data
        total_kwh = round(1200.0 + (hash(f"{year}{lat}{lon}") % 500), 2)
        
        return total_kwh


# Global service instance
prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """Get prediction service instance."""
    return prediction_service
