# Startup Manager Component

## Purpose
Handles initial data fetch, model training/loading, and orchestrates the startup sequence for the engine.

## Key Functions
- `startup_sequence(config: dict) -> None`
  - Loads config, triggers weather data fetch, model training/loading, and starts API server.
- `fetch_training_data(config: dict) -> str`
  - Downloads grid weather data, backs up locally. Returns path to data.
- `train_or_load_models(config: dict, data_path: str) -> dict`
  - Trains models if `train_now`, loads models if `load_models`. Returns model objects.

## Example Usage
```python
config = load_config('config.yaml')
data_path = fetch_training_data(config)
models = train_or_load_models(config, data_path)
startup_sequence(config)
```

## Inputs/Outputs
- **Inputs**: Config dict, local file paths
- **Outputs**: Model objects in memory, local backup files
