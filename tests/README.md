# Test Suite - Quick Reference

**Status**: 🟢 40/40 tests passing (100%)  
**Coverage**: 43%  
**Execution Time**: ~17 seconds

> 📖 **For complete documentation**, see [doc/testing.md](../doc/testing.md)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
cd repo
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m api           # API endpoint tests
pytest -m service       # Service layer tests
pytest -m schema        # Schema validation tests
```

---

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures (244 lines)
├── pytest.ini                       # Pytest configuration
├── api/
│   └── test_prediction_endpoints.py # 15 tests ✅
├── services/
│   ├── test_prediction_service.py   # 5 tests ✅
│   └── test_model_manager_service.py # 4 tests ✅
├── schemas/
│   └── test_prediction_schemas.py   # 13 tests ✅
├── models/
│   ├── test_weather_trainer.py      # 5 tests (skipped)
│   └── test_solar_trainer.py        # 6 tests (skipped)
└── integration/
    └── test_app_lifecycle.py        # 6 tests (skipped)
```

**Active**: 40 tests (all passing)  
**Skipped**: 17 tests (ML training + integration, deferred to e2e phase)

---

## Common Commands

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Stop on first failure
pytest -x

# Run specific file
pytest tests/api/test_prediction_endpoints.py

# Run specific test
pytest tests/api/test_prediction_endpoints.py::TestPredictDayEndpoint::test_success

# Pattern matching
pytest -k "predict_day"

# With coverage report
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Skip slow tests
pytest -m "not slow"
```

---

## Test Markers

```bash
pytest -m unit          # Fast unit tests with mocks
pytest -m api           # API endpoint tests
pytest -m service       # Service layer tests
pytest -m schema        # Schema validation tests
pytest -m model         # Model tests (skipped by default)
pytest -m integration   # Integration tests (skipped by default)
pytest -m slow          # Slow tests (training, etc.)
```

---

## Common Fixtures

Available in `conftest.py`:

- `test_client` - FastAPI test client with mocked models
- `mock_model_manager` - Mocked model manager (ready state)
- `prediction_service` - Service instance with mocked dependencies
- `temp_model_dir` - Temporary directory for test files
- `sample_weather_csv` - Synthetic weather data
- `valid_day_prediction_request` - Valid API request params

---

## Writing New Tests

### 1. Choose the Right Layer

- **API tests** (`tests/api/`): HTTP request/response behavior
- **Service tests** (`tests/services/`): Business logic
- **Schema tests** (`tests/schemas/`): Data validation
- **Model tests** (`tests/models/`): Training interfaces (optional)

### 2. Follow the Pattern

```python
@pytest.mark.unit
@pytest.mark.api
class TestMyFeature:
    """Tests for my new feature."""
    
    def test_success_case(self, test_client):
        """Test successful operation."""
        # Arrange
        params = {"key": "value"}
        
        # Act
        response = test_client.get("/endpoint", params=params)
        
        # Assert
        assert response.status_code == 200
    
    def test_error_case(self, test_client):
        """Test error handling."""
        response = test_client.get("/endpoint", params={"invalid": "data"})
        assert response.status_code == 400
```

### 3. Use Fixtures

```python
def test_with_fixtures(test_client, mock_model_manager, temp_model_dir):
    # test_client: FastAPI client with mocked dependencies
    # mock_model_manager: Mocked ML model manager
    # temp_model_dir: Temporary directory for test files
    pass
```

---

## Debugging

```bash
# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Run only failed tests
pytest --lf

# Show local variables on failure
pytest -l

# Very verbose
pytest -vv
```

---

## Coverage

```bash
# Terminal report with missing lines
pytest --cov=app --cov-report=term-missing

# HTML report
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

**Current Coverage**: 43%
- Schemas: 100% ✅
- API: 77% ✅
- Services: 67-98% ✅
- Models: 9-16% (intentionally low - tested in ML pipeline)

---

## Getting Help

1. 📖 **Full Documentation**: [doc/testing.md](../doc/testing.md)
2. 🔍 **See All Fixtures**: `pytest --fixtures`
3. 📝 **Check Examples**: Look at existing tests in each directory
4. 🐛 **Debugging**: Use `pytest -v -s --pdb`

---

**For comprehensive testing guide, see [doc/testing.md](../doc/testing.md)**
