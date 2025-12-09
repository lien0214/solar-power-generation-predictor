# API Server Component

## Purpose
Exposes FastAPI endpoints for day, month, and year predictions, serves Swagger UI for client interaction.

## Key Functions
- `get_day_prediction(...)`
  - Handles GET /predict/day requests, parses query params, calls prediction engine.
- `get_month_prediction(...)`
  - Handles GET /predict/month requests, parses query params, calls prediction engine.
- `get_year_prediction(...)`
  - Handles GET /predict/year requests, parses query params, calls prediction engine.
- `startup_event()`
  - FastAPI startup event to trigger engine initialization.

## Example Usage
```python
@app.get('/predict/day')
def get_day_prediction(...):
    # Parse params, call prediction engine
    return result
```

## Inputs/Outputs
- **Inputs**: API query parameters
- **Outputs**: JSON prediction results
