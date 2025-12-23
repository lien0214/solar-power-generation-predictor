# Prediction Engine Component

## Overview

The Prediction Engine is the core logic layer that transforms user requests (Location + Date) into solar energy estimates (kWh). It orchestrates the flow of data between the Weather Fetcher, Feature Engineering, and the Solar ML Models.

## Prediction Flow

The process follows a strict pipeline to ensure data consistency:

1.  **Input Validation**:
    *   Validates Lat/Lon boundaries.
    *   Parses and validates Date ranges.
    *   Ensures the requested Strategy is loaded.

2.  **Date Expansion**:
    *   Converts start/end dates into a daily sequence (Pandas DataFrame).
    *   Adds metadata columns (`lat`, `lon`, `pmp`).

3.  **Feature Engineering (Temporal)**:
    *   Extracts `year`, `month`, `day` from the date sequence.
    *   These are critical features for capturing seasonality.

4.  **Weather Enrichment**:
    *   Calls the **Weather Fetcher** (see `weather-fetcher.md`) to add meteorological features (`Global_Solar_Radiation`, `Temperature`, etc.) to the DataFrame.
    *   *Critical*: Solar models cannot predict without this weather context.

5.  **Solar Inference**:
    *   Passes the enriched DataFrame to the selected Solar Model.
    *   Applies the selected **Strategy** (Merged vs Separated).

6.  **Aggregation (Optional)**:
    *   For `/predict/month` and `/predict/year`, aggregates daily kWh sums.

## Prediction Strategies

The engine supports two distinct modes of operation via the `strategy` parameter:

### 1. Merged Strategy (`merged`)
*   **Logic**: Uses a single, generalized XGBoost model.
*   **Use Case**: Best for general-purpose predictions or when the specific site characteristics are unknown.
*   **Implementation**: `solar_models["merged"].predict(X)`

### 2. Separated Strategy (`seperated`)
*   **Logic**: Uses an **Ensemble Mean** approach.
*   **Mechanism**:
    1.  The input is fed into *every* available site-specific model (e.g., Site A, Site B, Site C).
    2.  Each model generates a prediction.
    3.  The engine calculates the mathematical mean of all predictions.
*   **Use Case**: Provides a more robust, stable prediction by averaging out outliers from individual site models.
*   **Implementation**:
    ```python
    preds = [model.predict(X) for model in solar_models["seperated"].values()]
    final_pred = np.mean(preds, axis=0)
    ```

## Code Structure

The logic is implemented in helper functions within `repo/main.py`:

*   `predict_day(...)`: Entry point for daily predictions.
*   `_predict_solar_power(...)`: Handles the strategy logic (Merged vs Ensemble).
*   `_predict_weather_features(...)`: Orchestrates weather data retrieval.

## Error Handling

*   **Model Not Loaded**: Returns `503 Service Unavailable` if the requested strategy is missing.
*   **Prediction Failure**: Returns `500 Internal Server Error` if the XGBoost model fails (e.g., input shape mismatch).
*   **Weather Failure**: If weather prediction fails, the system logs an error. (Note: Current implementation may fail hard or return partial data depending on the exception).