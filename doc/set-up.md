# Product Setup Guide

## Installation
1. **Clone the repository** to your local machine.
2. **Install Python dependencies** (see requirements.txt).
3. **Install Redis** (optional, for caching; can be local or remote).
4. **Install PyInstaller** if packaging as an executable.

## Configuration
- All product constants and settings are managed in a config file `config.yaml`.
- Key settings:
  - Startup mode: `train_now` or `load_models`
  - Redis connection (host, port)
  - Data paths: `WEATHER_HIST_FILE` (historical weather) and `SOLAR_DATA_DIR` (solar sites folder)
  - API server settings (host, port)

## Startup Modes
- **Train Now**: Scans `SOLAR_DATA_DIR` for CSV files, trains both merged and separated models, and saves them.
- **Load Models**: Loads pre-trained models from disk for immediate use.

## Starting the Engine
1. Ensure config file is set up correctly.
2. Run the main entry point (e.g., `python main.py` or the packaged executable).
3. The engine will fetch data, train/load models, and start the FastAPI server.
4. Access the Swagger UI at `/docs` for API exploration.

## Backup & Caching
- Training data is backed up locally after fetch.
- Predictions and results are cached in Redis for fast repeated access.

## Example Config File
```yaml
startup_mode: train_now
redis:
  host: localhost
  port: 6379
  db: 0
data_paths:
  weather_data: ./data/grid-weather
  solar_data: app/data
  models: ./models
api:
  host: 0.0.0.0
  port: 8000
```
