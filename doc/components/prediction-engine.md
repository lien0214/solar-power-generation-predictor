# Prediction Engine Component

## Purpose
Handles API requests, runs weather prediction, then solar prediction, and returns results to the user.

## Key Functions
- `predict_day(lon: float, lat: float, start_date: str, end_date: str, pmp: float, models: dict, cache: CacheLayer) -> dict`
  - Predicts solar generation for a day range, returns results.
- `predict_month(lon: float, lat: float, month: int, year: int, pmp: float, models: dict, cache: CacheLayer) -> dict`
  - Predicts solar generation for a month, returns results.
- `predict_year(lon: float, lat: float, year: int, pmp: float, models: dict, cache: CacheLayer) -> dict`
  - Predicts solar generation for a year, returns results.

## Example Usage
```python
result = predict_day(119.588339, 23.530236, '2025-01-01', '2025-01-31', 1000, models, cache)
```

## Inputs/Outputs
- **Inputs**: Location, date range, PMP, models, cache
- **Outputs**: Prediction results (dict)
