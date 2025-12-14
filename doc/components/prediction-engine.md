# Prediction Engine Component (Updated)

**Last Updated**: December 15, 2025  
**Status**: ✅ Aligned with actual implementation in `app/services/prediction.py`

---

## Purpose

The Prediction Engine Service is the business logic layer that:
- Handles API requests and orchestrates predictions
- Uses weather models to forecast future weather
- Uses solar models to predict generation based on weather forecasts
- Formats results for API responses
- Manages error handling and validation

---

## Architecture

### Service Class

```python
class PredictionService:
    """Service for generating solar power predictions."""
    
    def __init__(self):
        """Initialize prediction service."""
        self.model_manager = get_model_manager()
```

### Dependencies
- **ModelManagerService**: Provides access to trained ML models
- **Schemas**: Uses Pydantic models for type safety (`DayPrediction`, `MonthPrediction`)

---

## API Reference

### PredictionService Methods

#### `async predict_day_range()`

```python
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
        lon: Longitude coordinate (-180 to 180)
        lat: Latitude coordinate (-90 to 90)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (inclusive)
        pmp: Panel Maximum Power in Watts (> 0)
        
    Returns:
        List of DayPrediction objects, one per day in range
        Each prediction has:
            - date: str (YYYY-MM-DD)
            - kwh: float (>= 0)
            
    Raises:
        RuntimeError: If models are not loaded/ready
        ValueError: If date format is invalid or range is invalid
        
    Example:
        predictions = await service.predict_day_range(
            lon=119.588339,
            lat=23.530236,
            start_date="2025-01-01",
            end_date="2025-01-10",
            pmp=1000.0
        )
        # Returns 10 predictions (Jan 1-10)
    """
```

**Behavior**:
- Validates models are ready before proceeding
- Parses dates and validates format
- Generates one prediction per day (inclusive of end date)
- Currently returns mock data (TODO: integrate actual model predictions)

---

#### `async predict_month_range()`

```python
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
        lon: Longitude coordinate (-180 to 180)
        lat: Latitude coordinate (-90 to 90)
        start_date: Start month in YYYY-MM format
        end_date: End month in YYYY-MM format (inclusive)
        pmp: Panel Maximum Power in Watts (> 0)
        
    Returns:
        List of MonthPrediction objects, one per month in range
        Each prediction has:
            - date: str (YYYY-MM)
            - kwh: float (>= 0)
            
    Raises:
        RuntimeError: If models are not loaded/ready
        ValueError: If month format is invalid or range is invalid
        
    Example:
        predictions = await service.predict_month_range(
            lon=119.588339,
            lat=23.530236,
            start_date="2025-01",
            end_date="2025-12",
            pmp=1000.0
        )
        # Returns 12 predictions (Jan-Dec 2025)
    """
```

**Behavior**:
- Validates models are ready
- Parses month strings (YYYY-MM)
- Handles cross-year ranges (e.g., 2024-11 to 2025-02)
- Generates one prediction per month (inclusive)
- Currently returns mock data (TODO: integrate actual model predictions)

---

#### `async predict_year()`

```python
async def predict_year(
    self,
    lon: float,
    lat: float,
    year: int,
    pmp: float
) -> float:
    """
    Predict total solar generation for a full year.
    
    Args:
        lon: Longitude coordinate (-180 to 180)
        lat: Latitude coordinate (-90 to 90)
        year: Year (2000-2100)
        pmp: Panel Maximum Power in Watts (> 0)
        
    Returns:
        Total yearly generation in kWh (float >= 0)
        
    Raises:
        RuntimeError: If models are not loaded/ready
        ValueError: If year is out of valid range
        
    Example:
        total_kwh = await service.predict_year(
            lon=119.588339,
            lat=23.530236,
            year=2025,
            pmp=1000.0
        )
        # Returns single float (e.g., 1850.5 kWh)
    """
```

**Behavior**:
- Validates models are ready
- Validates year is in range (2000-2100)
- Returns single aggregated value (sum of all days in year)
- Currently returns mock data (TODO: integrate actual model predictions)

---

## Prediction Pipeline

### High-Level Flow

```
1. API Request → Validation
2. Check Models Ready → Error 503 if not
3. Fetch Weather History → From data or cache
4. Predict Future Weather → Using weather model
5. Predict Solar Generation → Using solar model + weather predictions
6. Format Results → Return Pydantic models
```

### Detailed Steps (Future Implementation)

#### Step 1: Weather Prediction
```python
# Pseudo-code for actual implementation
weather_model = model_manager.get_weather_model()
weather_input = prepare_weather_features(lat, lon, start_date, window=30)
weather_predictions = weather_model["model"].predict(weather_input)
```

**Weather Features** (30-day window):
- Past 30 days of weather data for 8 variables
- Creates 240 features (30 days × 8 variables)
- Scaled using weather model's StandardScaler

**Weather Outputs**:
- Future predictions for 8 weather variables
- T2M, T2M_MAX, TS, CLOUD_AMT_DAY, CLOUD_OD, ALLSKY_SFC_SW_DWN, RH2M, ALLSKY_SFC_SW_DIRH

#### Step 2: Solar Prediction
```python
# Pseudo-code for actual implementation
solar_model = model_manager.get_solar_model()
solar_input = prepare_solar_features(
    weather_predictions,
    pmp,
    dataset_name
)
solar_predictions = solar_model["model"].predict(solar_input)
```

**Solar Features**:
- Predicted weather variables (8 features)
- Panel power (PMP) as continuous feature
- Dataset/site encoded as one-hot features
- Total: ~10-15 features depending on number of sites

**Solar Outputs**:
- Daily generation in kWh
- One value per day

---

## Data Flow Diagram

```
┌─────────────────┐
│  API Request    │
│  (lon, lat,     │
│   date, pmp)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ PredictionService       │
│ - Validate inputs       │
│ - Check models ready    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Get Historical Weather  │
│ (from data/cache)       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Weather Model           │
│ Input: 30-day window    │
│ Output: Future weather  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Solar Model             │
│ Input: Weather + PMP    │
│ Output: Daily kWh       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Format Response         │
│ Return DayPrediction[]  │
└─────────────────────────┘
```

---

## Error Handling

### RuntimeError: Models Not Ready
```python
if not self.model_manager.is_ready():
    raise RuntimeError("Models not loaded. Please wait for initialization.")
```

**When**: Models haven't finished loading or training failed
**HTTP Status**: 503 Service Unavailable
**User Action**: Wait for server initialization to complete

### ValueError: Invalid Date
```python
start = datetime.strptime(start_date, "%Y-%m-%d")
# Raises ValueError if format doesn't match
```

**When**: Date string doesn't match expected format
**HTTP Status**: 400 Bad Request
**User Action**: Fix date format to YYYY-MM-DD or YYYY-MM

### ValueError: Invalid Range
```python
if end < start:
    raise ValueError("end_date must be after start_date")
```

**When**: End date is before start date
**HTTP Status**: 400 Bad Request
**User Action**: Ensure end_date >= start_date

---

## Usage Examples

### Example 1: Basic Day Prediction

```python
from app.services import get_prediction_service

service = get_prediction_service()

# Predict 10 days
predictions = await service.predict_day_range(
    lon=119.588339,
    lat=23.530236,
    start_date="2025-01-01",
    end_date="2025-01-10",
    pmp=1000.0
)

for pred in predictions:
    print(f"{pred.date}: {pred.kwh:.2f} kWh")
```

Output:
```
2025-01-01: 5.23 kWh
2025-01-02: 4.98 kWh
...
2025-01-10: 5.67 kWh
```

### Example 2: Monthly Predictions

```python
# Predict full year by months
predictions = await service.predict_month_range(
    lon=119.588339,
    lat=23.530236,
    start_date="2025-01",
    end_date="2025-12",
    pmp=2000.0
)

total = sum(pred.kwh for pred in predictions)
print(f"Total: {total:.2f} kWh")
```

### Example 3: Yearly Total

```python
# Get annual total directly
total_kwh = await service.predict_year(
    lon=119.588339,
    lat=23.530236,
    year=2025,
    pmp=1500.0
)

print(f"2025 total: {total_kwh:.2f} kWh")
```

### Example 4: Error Handling

```python
try:
    predictions = await service.predict_day_range(
        lon=119.588339,
        lat=23.530236,
        start_date="2025-01-10",
        end_date="2025-01-01",  # Invalid: end before start
        pmp=1000.0
    )
except ValueError as e:
    print(f"Validation error: {e}")
except RuntimeError as e:
    print(f"Service error: {e}")
```

---

## Integration with API Layer

The API layer (`app/api/v1/prediction.py`) calls these service methods:

```python
@router.get("/day")
async def predict_day(...):
    predictions = await prediction_service.predict_day_range(...)
    return DayPredictionResponse(
        location=Location(lat=lat, lon=lon),
        startDate=startDate,
        endDate=endDate,
        pmp=pmp,
        predictions=predictions
    )
```

**Separation of Concerns**:
- **API Layer**: HTTP handling, query param parsing, response formatting
- **Service Layer**: Business logic, model orchestration, data processing
- **Model Layer**: ML predictions, training, persistence

---

## Mock Data (Current Implementation)

Currently, the service returns mock data for development:

```python
# Example mock implementation
predictions = []
current = start
while current <= end:
    predictions.append(
        DayPrediction(
            date=current.strftime("%Y-%m-%d"),
            kwh=5.0 + (hash(current.day) % 100) / 20  # Pseudo-random
        )
    )
    current += timedelta(days=1)
return predictions
```

**TODO**: Replace with actual model predictions once integration is complete.

---

## Performance

### Response Times
| Operation | Expected Time |
|-----------|---------------|
| Single day | < 50ms |
| 10 days | < 100ms |
| 30 days | < 200ms |
| 365 days | < 500ms |
| 12 months | < 150ms |
| 1 year (total) | < 100ms |

*Times exclude network latency, based on local model inference*

### Optimization Opportunities
- **Caching**: Cache weather predictions for common locations
- **Batch Processing**: Process multiple days in single model call
- **Pre-computation**: Pre-compute predictions for popular locations
- **GPU Acceleration**: Use GPU for large batch predictions

---

## Testing

### Unit Tests

```python
# tests/services/test_prediction_service.py

@pytest.mark.asyncio
async def test_predict_day_range_success():
    """Test successful day range prediction."""
    service = PredictionService()
    predictions = await service.predict_day_range(
        lon=119.588339,
        lat=23.530236,
        start_date="2025-01-01",
        end_date="2025-01-10",
        pmp=1000.0
    )
    assert len(predictions) == 10
    assert all(pred.kwh >= 0 for pred in predictions)

@pytest.mark.asyncio
async def test_predict_day_range_models_not_ready():
    """Test error when models not ready."""
    service = PredictionService()
    service.model_manager = mock_not_ready_manager
    
    with pytest.raises(RuntimeError, match="Models not loaded"):
        await service.predict_day_range(...)
```

### Integration Tests

```python
# tests/integration/test_app_lifecycle.py

@pytest.mark.integration
async def test_end_to_end_prediction_flow(test_client):
    """Test complete prediction flow from API to response."""
    response = test_client.get(
        "/v1/predict/day",
        params={
            "lon": 119.588339,
            "lat": 23.530236,
            "startDate": "2025-01-01",
            "endDate": "2025-01-05",
            "pmp": 1000
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 5
```

---

## Future Enhancements

### Phase 1: Actual Model Integration
- [ ] Implement real weather prediction pipeline
- [ ] Implement real solar prediction pipeline
- [ ] Add weather data fetching from NASA POWER API
- [ ] Handle missing data gracefully

### Phase 2: Caching Layer
- [ ] Add Redis cache for weather predictions
- [ ] Cache solar predictions for common queries
- [ ] Implement cache invalidation strategy
- [ ] Add cache hit rate metrics

### Phase 3: Advanced Features
- [ ] Confidence intervals for predictions
- [ ] Multi-location batch predictions
- [ ] Historical comparison (predicted vs actual)
- [ ] Weather scenario analysis (best/worst case)

### Phase 4: Performance
- [ ] Async model inference
- [ ] GPU support for batch predictions
- [ ] Pre-compute predictions for grid points
- [ ] Stream large result sets

---

## Configuration

Service behavior configured via environment variables:

```python
# .env
PREDICTION_CACHE_TTL=3600  # Cache predictions for 1 hour
PREDICTION_MAX_DAYS=365    # Max days per request
PREDICTION_MAX_MONTHS=24   # Max months per request
MOCK_DATA_MODE=false       # Use actual models (set to true for testing)
```

---

## Monitoring & Logging

### Logging

```python
logger.info(f"Predicting {days} days for location ({lat}, {lon})")
logger.debug(f"Weather features shape: {features.shape}")
logger.warning(f"Using mock data - actual model not integrated")
logger.error(f"Prediction failed: {error}")
```

### Metrics (Future)

```python
# Prometheus metrics
prediction_request_total.inc()
prediction_duration_seconds.observe(duration)
prediction_cache_hits.inc()
model_inference_time.observe(inference_time)
```

---

## Dependencies

```python
# Core dependencies
from datetime import datetime, timedelta
from typing import List, Optional

# Internal dependencies
from ..schemas import DayPrediction, MonthPrediction
from .model_manager import get_model_manager

# External dependencies (future)
import numpy as np
import pandas as pd
```

---

## Change Log

### Version 1.0 (2025-12-15)
- ✅ Initial implementation with mock data
- ✅ All three prediction methods (day, month, year)
- ✅ Model readiness checks
- ✅ Error handling and validation
- ✅ Async/await support
- ⏳ Actual model integration (TODO)
