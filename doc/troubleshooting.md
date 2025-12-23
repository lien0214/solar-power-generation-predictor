# Troubleshooting Guide

## Common Issues

### 1. "Solar model not loaded" Error
**Symptom**: API returns 503 Service Unavailable with detail "Solar model ... is not loaded".
**Cause**: The application started in `load_models` mode but could not find the `.pkl` files in `models/`, or `train_now` mode failed to find CSVs.
**Solution**:
- If using `load_models`: Ensure `solar_model_bundle.pkl` exists in the `models/` directory.
- If using `train_now`: Check logs for "No solar data CSV files found". Ensure CSVs are in `repo/app/data/`.

### 2. "Weather history file not found"
**Symptom**: Logs show warning about missing weather file and "Using fallback".
**Cause**: `WEATHER_HIST_FILE` path is incorrect or file is missing.
**Solution**: Verify `WEATHER_HIST_FILE` env var points to a valid CSV (default: `../code/data/23.530236_119.588339.csv`).

### 3. Predictions are flat or constant
**Symptom**: The API returns the same value for every day.
**Cause**: The model might be failing to predict weather features and falling back to means, or the solar model is over-regularized.
**Solution**: Check logs for "Weather prediction failed". Ensure `weather_model_bundle` is loaded correctly.

### 4. Startup is very slow
**Symptom**: Server takes > 5 minutes to start.
**Cause**: `STARTUP_MODE` is set to `train_now`.
**Solution**: Switch to `STARTUP_MODE=load_models` after the first successful training run.

## Debugging

To enable verbose logging, ensure your environment is set up to capture standard output.

```bash
export LOG_LEVEL=DEBUG
uvicorn main:app --reload
```