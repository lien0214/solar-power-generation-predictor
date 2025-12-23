# Model Manager Component

## Overview

The Model Manager is responsible for the lifecycle of the machine learning models within the application. It handles initialization, training, loading, and serving of both weather and solar prediction models.

In the current architecture (v1.0), this logic is encapsulated within the **FastAPI Lifespan** event handler in `repo/main.py` and utilizes global state for high-performance access.

## Responsibilities

1.  **Startup Initialization**: Determines whether to train new models or load existing ones based on configuration.
2.  **Dynamic Data Discovery**: Automatically finds and registers solar site data for training.
3.  **Model State Management**: Maintains the global `weather_model_bundle` and `solar_models` dictionary.
4.  **Strategy Support**: Loads both "merged" and "separated" model strategies into memory.

## Startup Modes

The behavior is controlled by the `STARTUP_MODE` environment variable:

### 1. `train_now` (Training Mode)
Used for development or when data updates are required.

1.  **Weather Training**: Trains the XGBoost weather model using `WEATHER_HIST_FILE`.
2.  **Solar Discovery**: Scans `SOLAR_DATA_DIR` (default: `repo/app/data/`) for all `.csv` files.
3.  **Solar Training**:
    *   **Merged**: Trains a single model on all discovered CSVs.
    *   **Separated**: Trains individual models for each CSV.
4.  **Persistence**: Saves all trained bundles (`.pkl`) to `MODEL_DIR`.
5.  **Loading**: Loads the newly trained models into memory.

### 2. `load_models` (Production Mode)
Used for fast startup in production environments.

1.  **Weather Loading**: Loads `weather_model_bundle.pkl` from `MODEL_DIR`.
2.  **Solar Loading**:
    *   Loads `solar_model_bundle.pkl` (Merged).
    *   Loads `solar_model_bundle_seperated.pkl` (Separated).
3.  **Validation**: Logs warnings if specific models are missing but allows the server to start (partial degradation).

## Global State

Models are stored in global variables in `repo/main.py` for zero-latency access during request processing:

```python
# Holds the Weather XGBoost model + Scalers
weather_model_bundle = None 

# Dictionary holding solar models by strategy
# {
#   "merged": <ModelBundle>,
#   "seperated": { "site_name": <ModelBundle>, ... }
# }
solar_models = {}
```

## Dynamic Solar Data Discovery

Instead of hardcoding site names, the system dynamically discovers training data:

```python
solar_data_path = Path(SOLAR_DATA_DIR)
solar_files = {p.stem: str(p) for p in solar_data_path.glob("*.csv")}
```

*   **Key Benefit**: To add a new solar site, simply drop the CSV file into the data folder and restart with `STARTUP_MODE=train_now`. No code changes are needed.