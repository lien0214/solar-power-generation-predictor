# API Contract

This document describes the API endpoints, input/output schemas, and example requests/responses for the solar power prediction engine.

---

## Endpoints

### 1. GET /predict/day
- **Description:** Predict solar generation for a specific day or date range at a given location.
- **Query Parameters:**
  - `lon` (float, required): Longitude
  - `lat` (float, required): Latitude
  - `startDate` (string, required): Start date (YYYY-MM-DD)
  - `endDate` (string, required): End date (YYYY-MM-DD)
  - `pmp` (float, optional): Panel Maximum Power (W)
- **Response:**
```json
{
  "location": {"lat": 23.530236, "lon": 119.588339},
  "startDate": "2025-01-01",
  "endDate": "2025-01-31",
  "pmp": 1000,
  "predictions": [
    {"date": "2025-01-01", "kwh": 5.23},
    {"date": "2025-01-02", "kwh": 4.98}
    // ... (endDate included)
  ]
}
```

### 2. GET /predict/month
- **Description:** Predict solar generation for a specific month at a given location.
- **Query Parameters:**
  - `lon` (float, required): Longitude
  - `lat` (float, required): Latitude
  - `startDate`: (string, required): Start date (YYYY-MM)
  - `endDate`: (string, required): End date (YYYY-MM)
  - `pmp` (float, optional): Panel Maximum Power (W)
- **Response:**
```json
{
  "location": {"lat": 23.530236, "lon": 119.588339},
  "month": 1,
  "year": 2025,
  "pmp": 1000,
  "predictions": [
    {"date": "2025-01", "kwh": 105.23},
    // ...
    {"date": "2025-05", "kwh": 304.98}
  ]
}
```

### 3. GET /predict/year
- **Description:** Predict solar generation for a specific year at a given location.
- **Query Parameters:**
  - `lon` (float, required): Longitude
  - `lat` (float, required): Latitude
  - `year` (int, required): Year (YYYY)
  - `pmp` (float, optional): Panel Maximum Power (W)
- **Response:**
```json
{
  "location": {"lat": 23.530236, "lon": 119.588339},
  "year": 2025,
  "pmp": 1000,
  "kwh": 1243.53
}
```

---

## General Notes
- All endpoints return results in JSON format.
- Dates in predictions are in ISO format (YYYY-MM-DD or YYYY-MM or YYYY).
- If a prediction is not available for a date, error response.
- Error responses follow standard FastAPI error format.

## Example Error Response
```json
{
  "detail": "Invalid latitude/longitude or date range."
}
```
