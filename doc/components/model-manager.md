# Model Manager Component

## Purpose
Trains or loads weather and solar generation models, keeps them in memory for fast prediction.

## Key Functions
- `train_models(weather_data_paths: List[str], solar_data_paths: List[str], config: dict) -> dict`
  - Trains weather and solar models, returns model objects.
- `load_models(model_dir: str) -> dict`
  - Loads pre-trained models from disk, returns model objects.
- `save_models(models: dict, model_dir: str) -> None`
  - Saves trained models to disk for future use.

## Example Usage
```python
models = train_models(weather_files, solar_files, config)
models = load_models('./models')
save_models(models, './models')
```

## Inputs/Outputs
- **Inputs**: Weather/solar data file paths, config dict, model directory
- **Outputs**: Model objects in memory, saved model files
