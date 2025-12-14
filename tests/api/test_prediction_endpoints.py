"""
API Endpoint Tests for Prediction Endpoints.
Tests all three prediction endpoints with various scenarios.

Following TDD: These tests verify the HTTP layer behavior.
"""

import pytest
from fastapi import status


@pytest.mark.api
@pytest.mark.unit
class TestPredictDayEndpoint:
    """Tests for GET /v1/predict/day endpoint."""
    
    def test_predict_day_success(self, test_client, valid_day_prediction_request):
        """
        RED → GREEN: Successfully predict solar generation for a day range.
        
        Expected:
        - Status 200
        - Response contains location, dates, pmp, and predictions list
        - Each prediction has date and kwh fields
        """
        response = test_client.get("/v1/predict/day", params=valid_day_prediction_request)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "location" in data
        assert data["location"]["lat"] == 23.530236
        assert data["location"]["lon"] == 119.588339
        assert data["startDate"] == "2025-01-01"
        assert data["endDate"] == "2025-01-10"
        assert data["pmp"] == 1000.0
        assert "predictions" in data
        assert len(data["predictions"]) == 10  # 10 days
        assert all("date" in p and "kwh" in p for p in data["predictions"])
    
    def test_predict_day_invalid_date_format(self, test_client):
        """
        RED → GREEN: Return 400 for invalid date format.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2025/01/01",  # Wrong format
            "endDate": "2025-01-10",
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/day", params=params)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_predict_day_end_before_start(self, test_client):
        """
        RED → GREEN: Return 400 when end_date is before start_date.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2025-01-10",
            "endDate": "2025-01-01",  # Before start
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/day", params=params)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "end_date must be after" in response.json()["detail"].lower()
    
    def test_predict_day_invalid_latitude(self, test_client):
        """
        RED → GREEN: Return 422 for latitude out of range.
        """
        params = {
            "lon": 119.588339,
            "lat": 95.0,  # > 90
            "startDate": "2025-01-01",
            "endDate": "2025-01-10",
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/day", params=params)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_day_invalid_longitude(self, test_client):
        """
        RED → GREEN: Return 422 for longitude out of range.
        """
        params = {
            "lon": 200.0,  # > 180
            "lat": 23.530236,
            "startDate": "2025-01-01",
            "endDate": "2025-01-10",
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/day", params=params)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_day_models_not_ready(self, test_client_models_not_ready):
        """
        RED → GREEN: Return 503 when models are not loaded.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2025-01-01",
            "endDate": "2025-01-10",
            "pmp": 1000.0
        }
        
        response = test_client_models_not_ready.get("/v1/predict/day", params=params)
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "not loaded" in response.json()["detail"].lower()
    
    def test_predict_day_default_pmp(self, test_client):
        """
        RED → GREEN: Use default PMP value when not provided.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2025-01-01",
            "endDate": "2025-01-01"
            # pmp not provided
        }
        
        response = test_client.get("/v1/predict/day", params=params)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["pmp"] == 1000.0  # Default


@pytest.mark.api
@pytest.mark.unit
class TestPredictMonthEndpoint:
    """Tests for GET /v1/predict/month endpoint."""
    
    def test_predict_month_success(self, test_client, valid_month_prediction_request):
        """
        RED → GREEN: Successfully predict solar generation for a month range.
        """
        response = test_client.get("/v1/predict/month", params=valid_month_prediction_request)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "location" in data
        assert data["location"]["lat"] == 23.530236
        assert data["startDate"] == "2025-01"
        assert data["endDate"] == "2025-06"
        assert "predictions" in data
        assert len(data["predictions"]) == 6  # 6 months
        assert all("date" in p and "kwh" in p for p in data["predictions"])
    
    def test_predict_month_invalid_format(self, test_client):
        """
        RED → GREEN: Return 400 when month format is invalid.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2025/01",  # Wrong separator (should be 2025-01)
            "endDate": "2025-06",
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/month", params=params)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_predict_month_end_before_start(self, test_client):
        """
        RED → GREEN: Return 400 when end month is before start month.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2025-06",
            "endDate": "2025-01",  # Before start
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/month", params=params)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_predict_month_cross_year_boundary(self, test_client):
        """
        RED → GREEN: Handle month ranges that cross year boundary.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2024-11",
            "endDate": "2025-02",
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/month", params=params)
        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()["predictions"]) == 4  # Nov, Dec, Jan, Feb


@pytest.mark.api
@pytest.mark.unit
class TestPredictYearEndpoint:
    """Tests for GET /v1/predict/year endpoint."""
    
    def test_predict_year_success(self, test_client, valid_year_prediction_request):
        """
        RED → GREEN: Successfully predict solar generation for a year.
        """
        response = test_client.get("/v1/predict/year", params=valid_year_prediction_request)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "location" in data
        assert data["year"] == 2025
        assert "kwh" in data
        assert isinstance(data["kwh"], (int, float))
        assert data["kwh"] > 0
    
    def test_predict_year_invalid_year_too_low(self, test_client):
        """
        RED → GREEN: Return 422 for year below 2000.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "year": 1999,
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/year", params=params)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_year_invalid_year_too_high(self, test_client):
        """
        RED → GREEN: Return 422 for year above 2100.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "year": 2101,
            "pmp": 1000.0
        }
        
        response = test_client.get("/v1/predict/year", params=params)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_year_models_not_ready(self, test_client_models_not_ready):
        """
        RED → GREEN: Return 503 when models are not loaded.
        """
        params = {
            "lon": 119.588339,
            "lat": 23.530236,
            "year": 2025,
            "pmp": 1000.0
        }
        
        response = test_client_models_not_ready.get("/v1/predict/year", params=params)
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
