"""
Service Layer Tests for Prediction Business Logic.

Following TDD: These tests verify the business logic layer.
"""

import pytest
from app.services.prediction import PredictionService


@pytest.mark.service
@pytest.mark.unit
class TestPredictionService:
    """Tests for PredictionService business logic."""
    
    @pytest.mark.asyncio
    async def test_predict_day_range_success(self, prediction_service):
        """
        RED → GREEN: Successfully predict for a day range.
        
        Verifies:
        - Returns predictions for all days in range
        - Each prediction has date and kwh
        """
        result = await prediction_service.predict_day_range(
            lon=119.588339,
            lat=23.530236,
            start_date="2025-01-01",
            end_date="2025-01-10",
            pmp=1000.0
        )
        
        assert len(result) == 10
        assert all(pred.date and pred.kwh is not None for pred in result)
    
    @pytest.mark.asyncio
    async def test_predict_day_range_single_day(self, prediction_service):
        """
        RED → GREEN: Handle single day prediction (start == end).
        """
        result = await prediction_service.predict_day_range(
            lon=119.588339,
            lat=23.530236,
            start_date="2025-01-01",
            end_date="2025-01-01",
            pmp=1000.0
        )
        
        assert len(result) == 1
        assert result[0].date == "2025-01-01"
    
    @pytest.mark.asyncio
    async def test_predict_day_range_models_not_ready(self, prediction_service, mock_model_manager_not_ready):
        """
        RED → GREEN: Raise exception when models are not ready.
        """
        prediction_service.model_manager = mock_model_manager_not_ready
        
        with pytest.raises(RuntimeError, match="Models not loaded"):
            await prediction_service.predict_day_range(
                lon=119.588339,
                lat=23.530236,
                start_date="2025-01-01",
                end_date="2025-01-10",
                pmp=1000.0
            )
    
    @pytest.mark.asyncio
    async def test_predict_month_range_success(self, prediction_service):
        """
        RED → GREEN: Successfully predict for a month range.
        """
        result = await prediction_service.predict_month_range(
            lon=119.588339,
            lat=23.530236,
            start_date="2025-01",
            end_date="2025-06",
            pmp=1000.0
        )
        
        assert len(result) == 6
        assert all(pred.date and pred.kwh is not None for pred in result)
    
    @pytest.mark.asyncio
    async def test_predict_year_success(self, prediction_service):
        """
        RED → GREEN: Successfully predict for a full year.
        """
        result = await prediction_service.predict_year(
            lon=119.588339,
            lat=23.530236,
            year=2025,
            pmp=1000.0
        )
        
        assert isinstance(result, (int, float))
        assert result > 0
