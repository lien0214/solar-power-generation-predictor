"""
Schema Validation Tests for Pydantic Models.

Following TDD: These tests verify Pydantic validation rules.
"""

import pytest
from pydantic import ValidationError
from app.schemas.prediction import (
    Location,
    DayPrediction,
    MonthPrediction,
    DayPredictionResponse,
    MonthPredictionResponse,
    YearPredictionResponse
)


@pytest.mark.schema
@pytest.mark.unit
class TestLocation:
    """Tests for Location schema validation."""
    
    def test_valid_coordinates(self):
        """
        RED → GREEN: Accept valid latitude and longitude.
        """
        location = Location(lat=23.530236, lon=119.588339)
        assert location.lat == 23.530236
        assert location.lon == 119.588339
    
    def test_latitude_too_high(self):
        """
        RED → GREEN: Reject latitude > 90.
        """
        with pytest.raises(ValidationError) as exc_info:
            Location(lat=95.0, lon=119.588339)
        assert "lat" in str(exc_info.value).lower()
    
    def test_latitude_too_low(self):
        """
        RED → GREEN: Reject latitude < -90.
        """
        with pytest.raises(ValidationError):
            Location(lat=-95.0, lon=119.588339)
    
    def test_longitude_too_high(self):
        """
        RED → GREEN: Reject longitude > 180.
        """
        with pytest.raises(ValidationError):
            Location(lat=23.530236, lon=200.0)
    
    def test_longitude_too_low(self):
        """
        RED → GREEN: Reject longitude < -180.
        """
        with pytest.raises(ValidationError):
            Location(lat=23.530236, lon=-200.0)


@pytest.mark.schema
@pytest.mark.unit
class TestDayPrediction:
    """Tests for DayPrediction schema validation."""
    
    def test_valid_day_prediction(self):
        """
        RED → GREEN: Accept valid day prediction.
        """
        pred = DayPrediction(date="2025-01-01", kwh=1500.5)
        assert pred.date == "2025-01-01"
        assert pred.kwh == 1500.5
    
    def test_negative_kwh_rejected(self):
        """
        RED → GREEN: Reject negative kWh values.
        """
        with pytest.raises(ValidationError) as exc_info:
            DayPrediction(date="2025-01-01", kwh=-100.0)
        assert "kwh" in str(exc_info.value).lower()
    
    def test_zero_kwh_accepted(self):
        """
        RED → GREEN: Accept zero kWh (e.g., night time or bad weather).
        """
        pred = DayPrediction(date="2025-01-01", kwh=0.0)
        assert pred.kwh == 0.0


@pytest.mark.schema
@pytest.mark.unit
class TestDayPredictionResponse:
    """Tests for DayPredictionResponse schema validation."""
    
    def test_valid_response(self):
        """
        RED → GREEN: Accept valid day prediction response.
        """
        response = DayPredictionResponse(
            location=Location(lat=23.530236, lon=119.588339),
            startDate="2025-01-01",
            endDate="2025-01-10",
            pmp=1000.0,
            predictions=[
                DayPrediction(date="2025-01-01", kwh=1500.0),
                DayPrediction(date="2025-01-02", kwh=1600.0)
            ]
        )
        assert len(response.predictions) == 2
        assert response.pmp == 1000.0
    
    def test_negative_pmp_rejected(self):
        """
        RED → GREEN: Reject negative PMP values.
        """
        with pytest.raises(ValidationError):
            DayPredictionResponse(
                location=Location(lat=23.530236, lon=119.588339),
                startDate="2025-01-01",
                endDate="2025-01-10",
                pmp=-1000.0,
                predictions=[]
            )


@pytest.mark.schema
@pytest.mark.unit
class TestMonthPredictionResponse:
    """Tests for MonthPredictionResponse schema validation."""
    
    def test_valid_response(self):
        """
        RED → GREEN: Accept valid month prediction response.
        """
        response = MonthPredictionResponse(
            location=Location(lat=23.530236, lon=119.588339),
            startDate="2025-01",
            endDate="2025-06",
            pmp=1000.0,
            predictions=[
                MonthPrediction(date="2025-01", kwh=45000.0),
                MonthPrediction(date="2025-02", kwh=42000.0)
            ]
        )
        assert len(response.predictions) == 2


@pytest.mark.schema
@pytest.mark.unit
class TestYearPredictionResponse:
    """Tests for YearPredictionResponse schema validation."""
    
    def test_valid_response(self):
        """
        RED → GREEN: Accept valid year prediction response.
        """
        response = YearPredictionResponse(
            location=Location(lat=23.530236, lon=119.588339),
            year=2025,
            pmp=1000.0,
            kwh=500000.0
        )
        assert response.year == 2025
        assert response.kwh == 500000.0
    
    def test_year_too_low_rejected(self):
        """
        RED → GREEN: Reject year < 2000.
        """
        with pytest.raises(ValidationError):
            YearPredictionResponse(
                location=Location(lat=23.530236, lon=119.588339),
                year=1999,
                pmp=1000.0,
                kwh=500000.0
            )
    
    def test_year_too_high_rejected(self):
        """
        RED → GREEN: Reject year > 2100.
        """
        with pytest.raises(ValidationError):
            YearPredictionResponse(
                location=Location(lat=23.530236, lon=119.588339),
                year=2101,
                pmp=1000.0,
                kwh=500000.0
            )
