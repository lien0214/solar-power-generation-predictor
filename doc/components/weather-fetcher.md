# Weather Fetcher Component

## Purpose
Downloads grid weather data from NASA POWER API for training and prediction, and backs up locally.

## Key Functions
- `fetch_grid_weather(lat_min, lat_max, lon_min, lon_max, grid_size, output_dir) -> List[str]`
  - Fetches weather data for grid points, saves CSVs, returns list of file paths.
- `fetch_point_weather(lat, lon, output_dir) -> str`
  - Fetches weather data for a single point, saves CSV, returns file path.

## Example Usage
```python
files = fetch_grid_weather(23.199836, 23.758598, 119.312190, 119.692245, 4, './data/grid-weather')
file = fetch_point_weather(23.530236, 119.588339, './data')
```

## Inputs/Outputs
- **Inputs**: Latitude/longitude bounds, grid size, output directory
- **Outputs**: CSV file paths with weather data
