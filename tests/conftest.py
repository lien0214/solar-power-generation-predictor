"""
Pytest configuration and shared fixtures.
Provides reusable test fixtures for all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ========== Application Fixtures ==========

@pytest.fixture
def test_client():
    """
    FastAPI TestClient with mocked model manager.
    Mocks at the model manager level so service validation logic still runs.
    This allows us to test API/service behavior without loading actual ML models.
    """
    from app.main import app
    from app.services.model_manager import model_manager_service
    from unittest.mock import Mock, MagicMock
    import numpy as np
    
    # Save original methods
    original_is_ready = model_manager_service.is_ready
    original_get_weather = model_manager_service.get_weather_model
    original_get_solar = model_manager_service.get_solar_model
    
    # Mock model manager to return fake models
    model_manager_service.is_ready = Mock(return_value=True)
    
    # Create fake model bundles that return predictable values
    fake_weather_model = MagicMock()
    fake_weather_model.predict = Mock(side_effect=lambda x: np.random.rand(len(x), 8) * 10 + 20)
    
    fake_solar_model = MagicMock()
    fake_solar_model.predict = Mock(side_effect=lambda x: np.random.rand(len(x)) * 2000 + 1000)
    
    fake_weather_bundle = {
        "mode": "multi",
        "model": fake_weather_model,
        "scaler": MagicMock(),
        "targets": ["T2M", "T2M_MAX", "TS", "CLOUD_AMT_DAY", "CLOUD_OD", "ALLSKY_SFC_SW_DWN", "RH2M", "ALLSKY_SFC_SW_DIRH"],
        "win": 30,
        "hottest_doy": 200
    }
    
    fake_solar_bundle = {
        "model": fake_solar_model,
        "scaler": MagicMock(),
        "datasets": ["CT安集01"],
        "features": ["T2M", "RH2M", "ALLSKY_SFC_SW_DWN"]
    }
    
    model_manager_service.get_weather_model = Mock(return_value=fake_weather_bundle)
    model_manager_service.get_solar_model = Mock(return_value=fake_solar_bundle)
    
    client = TestClient(app)
    yield client
    
    # Restore original methods
    model_manager_service.is_ready = original_is_ready
    model_manager_service.get_weather_model = original_get_weather
    model_manager_service.get_solar_model = original_get_solar


@pytest.fixture
def test_client_models_not_ready():
    """
    TestClient with models NOT ready (for testing 503 errors).
    """
    from app.main import app
    from app.services import get_model_manager
    
    mock_model_mgr = Mock()
    mock_model_mgr.is_ready.return_value = False
    mock_model_mgr.weather_model_bundle = None
    mock_model_mgr.solar_model_bundle = None
    
    app.dependency_overrides[get_model_manager] = lambda: mock_model_mgr
    
    client = TestClient(app)
    yield client
    
    app.dependency_overrides.clear()


# ========== Service Fixtures ==========

@pytest.fixture
def mock_model_manager():
    """Mock ModelManagerService for testing."""
    mock = Mock()
    mock.is_ready.return_value = True
    mock.weather_model_bundle = {
        "mode": "multi",
        "targets": ["T2M", "T2M_MAX", "TS"],
        "win": 30,
        "features": ["T2M", "T2M_MAX"],
        "hottest_offset": 200
    }
    mock.solar_model_bundle = {
        "datasets": ["CT安集01", "CT安集02"],
        "feature_cols": ["T2M", "PMP", "ds_CT安集01"],
        "required_features": ["T2M", "T2M_MAX"]
    }
    mock.initialize = AsyncMock()
    mock.get_weather_model.return_value = mock.weather_model_bundle
    mock.get_solar_model.return_value = mock.solar_model_bundle
    return mock


@pytest.fixture
def mock_model_manager_not_ready():
    """Mock ModelManagerService that's not ready."""
    mock = Mock()
    mock.is_ready.return_value = False
    mock.weather_model_bundle = None
    mock.solar_model_bundle = None
    mock.initialize = AsyncMock()
    mock.get_weather_model.return_value = None
    mock.get_solar_model.return_value = None
    return mock


@pytest.fixture
def prediction_service(mock_model_manager):
    """Create PredictionService with mocked model manager."""
    from app.services.prediction import PredictionService
    service = PredictionService()
    service.model_manager = mock_model_manager
    return service


# ========== File System Fixtures ==========

@pytest.fixture
def temp_model_dir():
    """Temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_weather_csv(temp_model_dir):
    """Create a small synthetic weather CSV for testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = {
        "YEAR": dates.year,
        "MO": dates.month,
        "DY": dates.day,
        "LAT": [23.530236] * 100,
        "LON": [119.588339] * 100,
        "T2M": np.random.uniform(15, 30, 100),
        "T2M_MAX": np.random.uniform(20, 35, 100),
        "TS": np.random.uniform(15, 30, 100),
        "CLOUD_AMT_DAY": np.random.uniform(0, 100, 100),
        "CLOUD_OD": np.random.uniform(0, 50, 100),
        "ALLSKY_SFC_SW_DWN": np.random.uniform(100, 300, 100),
        "RH2M": np.random.uniform(40, 90, 100),
        "ALLSKY_SFC_SW_DIRH": np.random.uniform(50, 250, 100),
    }
    df = pd.DataFrame(data)
    
    csv_path = temp_model_dir / "test_weather.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def sample_solar_csv(temp_model_dir):
    """Create a small synthetic solar CSV for testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = {
        "date": dates.strftime("%Y-%m-%d"),
        "PMP": [1000.0] * 100,
        "KWh": np.random.uniform(3, 8, 100),
    }
    df = pd.DataFrame(data)
    
    csv_path = temp_model_dir / "test_solar.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def sample_weather_pred_csv(temp_model_dir):
    """Create synthetic weather prediction CSV with _true and _pred columns."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2024-06-01", periods=30, freq="D")
    data = {"Date": dates.strftime("%Y-%m-%d")}
    
    # Add _true and _pred columns for each feature
    features = ["T2M", "T2M_MAX", "TS", "CLOUD_AMT_DAY", "CLOUD_OD",
                "ALLSKY_SFC_SW_DWN", "RH2M", "ALLSKY_SFC_SW_DIRH"]
    
    for feat in features:
        data[f"{feat}_true"] = np.random.uniform(10, 30, 30)
        data[f"{feat}_pred"] = np.random.uniform(10, 30, 30)
    
    df = pd.DataFrame(data)
    csv_path = temp_model_dir / "test_weather_pred.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


# ========== Mock Data Fixtures ==========

@pytest.fixture
def valid_day_prediction_request():
    """Valid request data for day prediction endpoint."""
    return {
        "lon": 119.588339,
        "lat": 23.530236,
        "startDate": "2025-01-01",
        "endDate": "2025-01-10",
        "pmp": 1000.0
    }


@pytest.fixture
def valid_month_prediction_request():
    """Valid request data for month prediction endpoint."""
    return {
        "lon": 119.588339,
        "lat": 23.530236,
        "startDate": "2025-01",
        "endDate": "2025-06",
        "pmp": 1000.0
    }


@pytest.fixture
def valid_year_prediction_request():
    """Valid request data for year prediction endpoint."""
    return {
        "lon": 119.588339,
        "lat": 23.530236,
        "year": 2025,
        "pmp": 1000.0
    }


# ========== Marker Helpers ==========

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "service: Service layer tests")
    config.addinivalue_line("markers", "model: ML model tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
