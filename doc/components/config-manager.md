# Config Manager Component

## Purpose
Loads, validates, and provides access to all product configuration settings from a config file.

## Key Functions
- `load_config(path: str) -> dict`
  - Loads config from YAML file, validates required fields.
- `get_setting(config: dict, key: str, default=None) -> Any`
  - Retrieves a specific setting from config, with optional default.

## Example Usage
```python
config = load_config('config.yaml')
redis_host = get_setting(config, 'redis.host', 'localhost')
```

## Inputs/Outputs
- **Inputs**: Config file path
- **Outputs**: Config dict for use by other components
