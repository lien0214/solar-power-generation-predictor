# API Contract (Updated)

This document describes the standardized API endpoints, input/output schemas, and example requests/responses for the solar power prediction engine.

**Last Updated**: December 15, 2025  
**Status**: ✅ Aligned with actual implementation in `app/api/v1/prediction.py`

---

## Base URL

- **Development**: `http://localhost:8000`
- **API Version**: `v1`
- **Base Path**: `/v1/predict`

---

## Endpoints

### 1. GET /v1/predict/day

**Description**: Predict solar generation for a specific day or date range at a given location.

**Query Parameters**:
| Parameter | Type | Required | Validation | Description |
|-----------|------|----------|------------|-------------|
| `lon` | float | Yes | -180 to 180 | Longitude coordinate |
| `lat` | float | Yes | -90 to 90 | Latitude coordinate |
| `startDate` | string | Yes | YYYY-MM-DD format | Start date of prediction range |
| `endDate` | string | Yes | YYYY-MM-DD format | End date of prediction range (inclusive) |
| `pmp` | float | No (default: 1000) | > 0 | Panel Maximum Power in Watts |

**Success Response** (200 OK):
```json
{
  "location": {
    "lat": 23.530236,
    "lon": 119.588339
  },
  "startDate": "2025-01-01",
  "endDate": "2025-01-31",
  "pmp": 1000.0,
  "predictions": [
    {
      "date": "2025-01-01",
      "kwh": 5.23
    },
    {
      "date": "2025-01-02",
      "kwh": 4.98
    }
    // ... one entry per day (endDate included)
  ]
}
```

**Schema**: `DayPredictionResponse`
- `location`: `Location` object with validated lat/lon
- `startDate`: Echo of request start date
- `endDate`: Echo of request end date
- `pmp`: Panel power (request value or default)
- `predictions`: Array of `DayPrediction` objects
  - `date`: ISO date string (YYYY-MM-DD)
  - `kwh`: Non-negative float representing daily generation

**Example Request**:
```bash
curl "http://localhost:8000/v1/predict/day?lon=119.588339&lat=23.530236&startDate=2025-01-01&endDate=2025-01-10&pmp=1000"
```

---

### 2. GET /v1/predict/month

**Description**: Predict solar generation for a month range at a given location.

**Query Parameters**:
| Parameter | Type | Required | Validation | Description |
|-----------|------|----------|------------|-------------|
| `lon` | float | Yes | -180 to 180 | Longitude coordinate |
| `lat` | float | Yes | -90 to 90 | Latitude coordinate |
| `startDate` | string | Yes | YYYY-MM format | Start month of prediction range |
| `endDate` | string | Yes | YYYY-MM format | End month of prediction range (inclusive) |
| `pmp` | float | No (default: 1000) | > 0 | Panel Maximum Power in Watts |

**Success Response** (200 OK):
```json
{
  "location": {
    "lat": 23.530236,
    "lon": 119.588339
  },
  "startDate": "2025-01",
  "endDate": "2025-12",
  "pmp": 1000.0,
  "predictions": [
    {
      "date": "2025-01",
      "kwh": 150.5
    },
    {
      "date": "2025-02",
      "kwh": 145.2
    }
    // ... one entry per month (endDate included)
  ]
}
```

**Schema**: `MonthPredictionResponse`
- `location`: `Location` object with validated lat/lon
- `startDate`: Echo of request start month (YYYY-MM)
- `endDate`: Echo of request end month (YYYY-MM)
- `pmp`: Panel power (request value or default)
- `predictions`: Array of `MonthPrediction` objects
  - `date`: ISO month string (YYYY-MM)
  - `kwh`: Non-negative float representing monthly generation (sum of all days in month)

**Example Request**:
```bash
curl "http://localhost:8000/v1/predict/month?lon=119.588339&lat=23.530236&startDate=2025-01&endDate=2025-12&pmp=1000"
```

---

### 3. GET /v1/predict/year

**Description**: Predict total solar generation for a specific year at a given location.

**Query Parameters**:
| Parameter | Type | Required | Validation | Description |
|-----------|------|----------|------------|-------------|
| `lon` | float | Yes | -180 to 180 | Longitude coordinate |
| `lat` | float | Yes | -90 to 90 | Latitude coordinate |
| `year` | int | Yes | 2000 to 2100 | Year for prediction |
| `pmp` | float | No (default: 1000) | > 0 | Panel Maximum Power in Watts |

**Success Response** (200 OK):
```json
{
  "location": {
    "lat": 23.530236,
    "lon": 119.588339
  },
  "year": 2025,
  "pmp": 1000.0,
  "kwh": 1850.5
}
```

**Schema**: `YearPredictionResponse`
- `location`: `Location` object with validated lat/lon
- `year`: Echo of request year
- `pmp`: Panel power (request value or default)
- `kwh`: Non-negative float representing total yearly generation (sum of all days in year)

**Example Request**:
```bash
curl "http://localhost:8000/v1/predict/year?lon=119.588339&lat=23.530236&year=2025&pmp=1000"
```

---

## Error Responses

All endpoints follow FastAPI standard error format.

### 400 Bad Request
**Cause**: Invalid input parameters (bad date format, invalid range, etc.)

```json
{
  "detail": "Invalid date format. Expected YYYY-MM-DD."
}
```

### 422 Unprocessable Entity
**Cause**: Validation error (lat/lon out of bounds, year out of range, etc.)

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["query", "lat"],
      "msg": "Input should be less than or equal to 90",
      "input": "95.0"
    }
  ]
}
```

### 503 Service Unavailable
**Cause**: Models not loaded yet (during startup or after error)

```json
{
  "detail": "Models not loaded. Please wait for initialization."
}
```

### 500 Internal Server Error
**Cause**: Unexpected server error

```json
{
  "detail": "Internal server error: <error message>"
}
```

---

## Data Types & Validation

### Location
```python
class Location(BaseModel):
    lat: float  # -90 to 90
    lon: float  # -180 to 180
```

### DayPrediction
```python
class DayPrediction(BaseModel):
    date: str   # YYYY-MM-DD format
    kwh: float  # >= 0
```

### MonthPrediction
```python
class MonthPrediction(BaseModel):
    date: str   # YYYY-MM format
    kwh: float  # >= 0
```

---

## General Notes

### Date Formats
- **Day predictions**: ISO 8601 date format `YYYY-MM-DD` (e.g., `2025-01-15`)
- **Month predictions**: Year-month format `YYYY-MM` (e.g., `2025-01`)
- **Year predictions**: 4-digit year `YYYY` (e.g., `2025`)

### Coordinate Validation
- **Latitude**: Must be between -90 and 90 (inclusive)
- **Longitude**: Must be between -180 and 180 (inclusive)
- **Precision**: Accepts up to 6 decimal places

### PMP (Panel Maximum Power)
- **Default value**: 1000 W (1 kW)
- **Units**: Watts (W)
- **Validation**: Must be positive (> 0)
- **Usage**: Scales the base prediction to panel capacity

### Energy Units
- All `kwh` values represent **kilowatt-hours (kWh)**
- Values are always non-negative floats
- Zero values indicate no generation (e.g., nighttime, extreme weather)

### Date Range Behavior
- **Inclusive**: Both start and end dates/months are included in results
- **Single day/month**: Set `startDate` equal to `endDate`
- **Order validation**: `endDate` must be >= `startDate`
- **Cross-year ranges**: Supported for month predictions (e.g., `2024-11` to `2025-02`)

---

## Response Times

Expected response times under normal conditions:

| Endpoint | Time Range | Expected Response Time |
|----------|-----------|------------------------|
| `/day` | 1-30 days | < 500ms |
| `/day` | 31-365 days | < 1000ms |
| `/month` | 1-12 months | < 500ms |
| `/year` | 1 year | < 300ms |

*Note: First request after server start may be slower due to model warm-up*

---

## API Versioning

- Current version: **v1**
- All endpoints are prefixed with `/v1`
- Future versions will be `/v2`, `/v3`, etc.
- Version compatibility: Breaking changes require new version number

---

## Authentication

**Current Status**: No authentication required (development phase)

**Future**: Will implement API key authentication
```bash
curl -H "X-API-Key: your-api-key" "http://localhost:8000/v1/predict/day?..."
```

---

## Rate Limiting

**Current Status**: No rate limiting (development phase)

**Future**: Rate limits will be applied
- **Free tier**: 100 requests/hour
- **Pro tier**: 1000 requests/hour

---

## Examples

### Example 1: Single Day Prediction
```bash
curl "http://localhost:8000/v1/predict/day?lon=119.588339&lat=23.530236&startDate=2025-01-15&endDate=2025-01-15&pmp=1500"
```

### Example 2: 10-Day Forecast
```bash
curl "http://localhost:8000/v1/predict/day?lon=119.588339&lat=23.530236&startDate=2025-01-01&endDate=2025-01-10&pmp=1000"
```

### Example 3: Full Year by Months
```bash
curl "http://localhost:8000/v1/predict/month?lon=119.588339&lat=23.530236&startDate=2025-01&endDate=2025-12&pmp=2000"
```

### Example 4: Cross-Year Months
```bash
curl "http://localhost:8000/v1/predict/month?lon=119.588339&lat=23.530236&startDate=2024-11&endDate=2025-02&pmp=1000"
```

### Example 5: Annual Total
```bash
curl "http://localhost:8000/v1/predict/year?lon=119.588339&lat=23.530236&year=2025&pmp=1000"
```

---

## Testing

### Health Check
```bash
curl "http://localhost:8000/health"
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### Interactive API Docs
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Change Log

### Version 1.0 (2025-12-15)
- ✅ Aligned with actual implementation
- ✅ Added validation rules and constraints
- ✅ Added error response examples
- ✅ Clarified date format requirements
- ✅ Added `MonthPrediction` schema (separate from `DayPrediction`)
- ✅ Documented default PMP value (1000 W)
- ✅ Added response time expectations
- ✅ Added testing section with health check
